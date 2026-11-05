"""Audit cached pose errors by data conditions, without changing weights or labels."""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def describe_conditions(target, quality):
    root = target[:, [11, 12]].mean(1)
    shoulder = target[:, [5, 6]].mean(1)
    torso = shoulder-root
    torso_cos = np.abs(torso[:, 2])/np.maximum(np.linalg.norm(torso, axis=1), 1e-6)
    legs = (np.linalg.norm(target[:, 11]-target[:, 13], axis=-1)
            + np.linalg.norm(target[:, 13]-target[:, 15], axis=-1)
            + np.linalg.norm(target[:, 12]-target[:, 14], axis=-1)
            + np.linalg.norm(target[:, 14]-target[:, 16], axis=-1))*.5
    ankle = target[:, [15, 16], 2].mean(1)
    extended = (root[:, 2]-ankle)/np.maximum(legs, 1e-6)
    posture = np.full(len(target), 'uncertain', dtype=object)
    posture[torso_cos < .4] = 'horizontal_torso'
    posture[(torso_cos >= .7) & (extended >= .7)] = 'upright_extended_legs'
    posture[(torso_cos >= .7) & (extended < .55)] = 'upright_bent_legs'
    motion = np.r_[np.nan, np.linalg.norm(np.diff(target, axis=0), axis=-1).mean(-1)]
    count = np.expm1(quality[:, 0]*np.log(201))
    count_bin = np.select([count < 9.5, count < 39.5, count < 99.5], ['lt10', '10_39', '40_99'], default='ge100')
    duration = np.zeros(len(target))
    for i in range(1, len(target)):
        if motion[i] < .01:
            duration[i] = duration[i-1]+quality[i, -1]
    static = np.select([~np.isfinite(motion), motion >= .01, duration < 2, duration < 5, duration < 15],
                       ['segment_start', 'moving', 'static_lt2s', 'static_2_5s', 'static_5_15s'], default='static_ge15s')
    normal = np.cross(target[:, 6]-target[:, 5], torso)[:, :2]
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    direction = root[:, :2]/np.maximum(np.linalg.norm(root[:, :2], axis=-1, keepdims=True), 1e-6)
    facing = np.abs((normal*direction).sum(-1))
    orientation = np.select([facing > .8, facing < .35], ['front_or_back', 'side'], default='oblique')
    x = np.select([root[:, 0] < 2, root[:, 0] < 4], ['x_lt2', 'x_2_4'], default='x_ge4')
    y = np.select([root[:, 1] < -1, root[:, 1] > 1], ['y_lt_minus1', 'y_gt1'], default='y_center')
    return dict(points=count_bin, motion=static, posture=posture, orientation=orientation,
                position=np.char.add(np.char.add(x, '/'), y),
                posture_motion=np.asarray([f'{p}/{m}' for p, m in zip(posture, static)]))


def head_foot_direction(poses):
    direction = poses[:, 0, :2]-poses[:, [15, 16], :2].mean(1)
    return direction/np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)


def upright_transition_response(target, predictions, quality):
    """GT-geometric rising candidates; report misses instead of silently dropping them."""
    def vertical(poses):
        torso = poses[:, [5, 6]].mean(1)-poses[:, [11, 12]].mean(1)
        return np.abs(torso[:, 2])/np.maximum(np.linalg.norm(torso, axis=-1), 1e-6)
    truth = vertical(target)
    predicted = [vertical(p) for p in predictions]
    result = np.zeros((2, 3))  # events, matched within 24 frames, total matched delay seconds
    for frame in range(1, len(target)-3):
        if not (truth[frame] > .7 and truth[frame-1] <= .7
                and np.all(truth[frame:frame+4] > .7)
                and np.any(truth[max(0, frame-24):frame] < .4)):
            continue
        for index, values in enumerate(predicted):
            result[index, 0] += 1
            for candidate in range(frame, min(len(target)-2, frame+24)):
                if np.all(values[candidate:candidate+3] > .7):
                    result[index, 1] += 1
                    result[index, 2] += quality[frame+1:candidate+1, -1].sum()
                    break
    return result


def summarize_streams(streams, predict=None):
    totals = defaultdict(lambda: np.zeros(5))
    joint = np.zeros((2, 17))
    frames = 0
    direction_flips = dict(adjacent_horizontal_pairs=0, gt=0, baseline=0)
    episodes = []
    rising = np.zeros((2, 3))
    for stream in streams:
        target, base = [stream[k].float().numpy() for k in ('target', 'baseline')]
        predicted = base if predict is None else predict(stream)
        quality = stream['quality'].double().numpy()
        rising += upright_transition_response(target, (base, predicted), quality)
        errors = np.stack((np.linalg.norm(base-target, axis=-1), np.linalg.norm(predicted-target, axis=-1)))
        roots = target[:, [11, 12]].mean(1, keepdims=True)
        relative = np.stack([np.linalg.norm((p-p[:, [11, 12]].mean(1, keepdims=True))-(target-roots), axis=-1).mean(-1)
                             for p in (base, predicted)])
        joint += errors.sum(1)
        frames += len(target)
        conditions = describe_conditions(target, quality)
        pd, gd = head_foot_direction(base), head_foot_direction(target)
        cosine = (pd*gd).sum(-1)
        horizontal = conditions['posture'] == 'horizontal_torso'
        conditions['head_foot'] = np.select([~horizontal, cosine < -.5, cosine > .5],
                                            ['not_horizontal', 'horizontal_opposite', 'horizontal_same'],
                                            default='horizontal_oblique')
        conditions['posture_points'] = np.asarray([f'{p}/{n}' for p, n in zip(conditions['posture'], conditions['points'])])
        pair = horizontal[1:] & horizontal[:-1]
        direction_flips['adjacent_horizontal_pairs'] += int(pair.sum())
        direction_flips['gt'] += int((pair & ((gd[1:]*gd[:-1]).sum(-1) < -.5)).sum())
        direction_flips['baseline'] += int((pair & ((pd[1:]*pd[:-1]).sum(-1) < -.5)).sum())
        conditions['group'] = np.full(len(target), stream['group'])
        conditions['all'] = np.full(len(target), 'all')
        means = errors.mean(-1)
        for kind, values in conditions.items():
            for label in np.unique(values):
                mask = values == label
                totals[f'{kind}:{label}'] += np.r_[mask.sum(), means[:, mask].sum(-1), relative[:, mask].sum(-1)]
        bad = means[0] > .2
        changes = np.flatnonzero(np.diff(np.r_[False, bad, False]))
        for begin, end in zip(changes[::2], changes[1::2]):
            if end-begin < 4:
                continue
            timestamps = stream.get('timestamp')
            episodes.append(dict(group=stream['group'], start_index=int(begin), end_index=int(end-1),
                                 timestamp=float(timestamps[begin]) if timestamps is not None else None,
                                 frames=int(end-begin), baseline_mpjpe=float(means[0, begin:end].mean()),
                                 refined_mpjpe=float(means[1, begin:end].mean()),
                                 error_sum=float(means[0, begin:end].sum()),
                                 point_bin=str(conditions['points'][begin]),
                                 posture=str(conditions['posture'][begin])))
    if not frames:
        raise ValueError('No cached frames to audit')
    result = []
    total_error = totals['all:all'][1]
    for key, values in totals.items():
        n, base, refined, rel_base, rel_refined = values
        result.append(dict(condition=key, frames=int(n), fraction=n/frames,
                           baseline_mpjpe=base/n, refined_mpjpe=refined/n, delta=(refined-base)/n,
                           relative_baseline=rel_base/n, relative_refined=rel_refined/n,
                           baseline_error_share=base/total_error,
                           excess_above_10cm=max(0., base-.1*n)/frames))
    return dict(frames=frames, direction_flips=direction_flips,
                upright_transition_candidates={name: dict(events=int(v[0]), detected=int(v[1]),
                    missed=int(v[0]-v[1]), matched_mean_delay_seconds=float(v[2]/v[1]) if v[1] else None)
                    for name, v in zip(('baseline', 'refined'), rising)},
                conditions=sorted(result, key=lambda r:r['excess_above_10cm'], reverse=True),
                joint_mpjpe=dict(baseline=(joint[0]/frames).tolist(), refined=(joint[1]/frames).tolist()),
                high_error_episodes=sorted(episodes, key=lambda r:r['error_sum'], reverse=True)[:100])


def load_predictor(experiment, checkpoint, device):
    saved = torch.load(experiment/checkpoint, map_location='cpu', weights_only=False)
    if saved.get('training_mode') == 'joint_finetune':
        if torch.device(device).type != 'cuda':
            raise ValueError('Joint P4 analysis requires CUDA pointnet2_ops; select --device cuda:0 or cuda:1')
        from run.utils.p4_finetune_data import load_saved_joint_model, predict_saved_joint, OnlinePoints
        model = load_saved_joint_model(experiment, saved, device).eval()
        readers = {}
        def predict_joint(stream):
            mode = stream['online']['mode']
            if mode not in readers:
                readers[mode] = OnlinePoints(stream['online'])
            return predict_saved_joint(model, stream, device, saved['config']['chunk_length'], readers[mode])
        return predict_joint
    source = experiment/'P4JointMemory.py'
    spec = importlib.util.spec_from_file_location('audited_pose_model', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.P4JointMemory(**saved['architecture']).to(device).eval()
    model.load_state_dict(saved['model_state_dict'])
    @torch.inference_mode()
    def predict(stream):
        hidden, output = None, []
        for start in range(0, len(stream['features']), 256):
            inputs = {k: stream[k][None, start:start+256].to(device)
                      for k in ('tokens', 'xyz', 'features', 'quality', 'rotation', 'translation')}
            pose, hidden = model(inputs, hidden)
            output.append(pose[0].float().cpu())
        return torch.cat(output).numpy()
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best.pth')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    predict = load_predictor(args.experiment, args.checkpoint, args.device)
    manifest = json.loads((args.experiment/'evidence_manifest.json').read_text())
    report = dict(checkpoint=args.checkpoint, definitions=dict(
        motion='GT mean joint displacement <1cm/frame; duration resets at stream start',
        posture='GT geometry heuristic for audit, not annotated action labels',
        orientation='GT torso plane relative to low-coordinate XY origin; front/back are not distinguished',
        points='current-frame pre-resampling crop count reconstructed from cached quality'), splits={})
    for split in ('train', 'val'):
        def streams():
            for info in manifest[split].values():
                if 'index' in info:
                    from run.utils.p4_finetune_data import load_streams
                    yield from load_streams({'group': info})
                else:
                    yield from torch.load(info['cache'], map_location='cpu', weights_only=False)
        report['splits'][split] = summarize_streams(streams(), predict if split == 'val' else None)
    path = args.experiment/f'diagnosis_{Path(args.checkpoint).stem}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Diagnosis: {path}')
    for row in report['splits']['val']['conditions']:
        if row['condition'].startswith(('posture:', 'motion:', 'points:')):
            print(row)


if __name__ == '__main__':
    main()
