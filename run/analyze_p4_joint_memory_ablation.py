"""Fixed-checkpoint inference interventions; no retraining or training-code changes."""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from run.analyze_pose_refiner import describe_conditions, head_foot_direction


FIELDS = ('tokens', 'xyz', 'features', 'quality', 'rotation', 'translation')
VARIANTS = {
    'full': 'Unmodified trained decoder',
    'paired_permutation': 'Roll tokens and XYZ together: permutation-invariance control',
    'original_head_only': 'Original pretrained head on cached pooled features (FP32)',
    'trained_head_only': 'Zero the complete feature_update output, including bias',
    'no_history': 'Zero previous hidden state in both memory_query and GRU every frame',
    'reset_every_16': 'Reset decoder state every 16 evidence frames',
    'reset_every_64': 'Reset decoder state every 64 evidence frames',
    'no_token_features': 'Zero token_norm output; retain explicit position and global context',
    'no_xyz_encoding': 'Zero position MLP output; implicit geometry in P4 features remains',
    'mean_token_features': 'Replace spatial tokens by their per-frame mean; retain XYZ',
    'misaligned_xyz': 'Roll only XYZ, breaking its correspondence with spatial tokens',
    'no_attention_evidence': 'Zero attention output; retain quality-driven GRU and global context',
    'no_quality': 'Zero quality encoder output, including bias',
    'no_joint_queries': 'Zero learned joint ID vectors; retain joint-specific output rows',
    'no_global_features': 'Zero pooled features in query and output paths: distribution-shift stress test',
}


@contextmanager
def intervene(model, variant):
    handles = []
    try:
        if variant in ('trained_head_only', 'original_head_only'):
            handles.append(model.feature_update.register_forward_hook(lambda m, a, o: torch.zeros_like(o)))
        elif variant == 'no_history':
            handles.append(model.memory_query.register_forward_pre_hook(lambda m, a: (torch.zeros_like(a[0]),)))
            handles.append(model.memory.register_forward_pre_hook(lambda m, a: (a[0], torch.zeros_like(a[1]))))
        elif variant in ('no_token_features', 'no_xyz_encoding', 'no_quality'):
            module = {'no_token_features': model.token_norm, 'no_xyz_encoding': model.position,
                      'no_quality': model.quality}[variant]
            handles.append(module.register_forward_hook(lambda m, a, o: torch.zeros_like(o)))
        elif variant == 'no_attention_evidence':
            handles.append(model.attention.register_forward_hook(lambda m, a, o: (torch.zeros_like(o[0]), o[1])))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def predict(model, stream, variant, device):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown intervention: {variant}')
    chunk = {'reset_every_16': 16, 'reset_every_64': 64}.get(variant, 256)
    state, outputs = None, []
    with intervene(model, variant):
        for start in range(0, len(stream['features']), chunk):
            inputs = {k: stream[k][None, start:start+chunk].to(device) for k in FIELDS}
            if variant == 'paired_permutation':
                inputs['tokens'] = inputs['tokens'].roll(1, 2)
                inputs['xyz'] = inputs['xyz'].roll(1, 2)
            elif variant == 'misaligned_xyz':
                inputs['xyz'] = inputs['xyz'].roll(1, 2)
            elif variant == 'mean_token_features':
                inputs['tokens'] = inputs['tokens'].float().mean(2, keepdim=True).expand_as(inputs['tokens'])
            elif variant == 'no_global_features':
                inputs['features'] = torch.zeros_like(inputs['features'])
            if variant.startswith('reset_every_'):
                state = None
            pose, state = model(inputs, state)
            outputs.append(pose[0].float().cpu().numpy())
    return np.concatenate(outputs)


def masks_for(stream, warmup):
    target = stream['target'].float().numpy()
    conditions = describe_conditions(target, stream['quality'].double().numpy())
    horizontal = conditions['posture'] == 'horizontal_torso'
    vector = target[:, 0, :2]-target[:, [15, 16], :2].mean(1)
    masks = dict(all=np.ones(len(target), bool), cold=np.arange(len(target)) < warmup,
                 warm=np.arange(len(target)) >= warmup, horizontal=horizontal,
                 nonhorizontal=~horizontal)
    masks.update({f'{key}:{label}': values == label for key, values in conditions.items()
                  if key in ('points', 'motion', 'posture') for label in np.unique(values)})
    return target, masks, horizontal & (np.linalg.norm(vector, axis=-1) > .5)


def accumulate(result, stream, pose, full, target, masks, horizontal):
    base = stream['baseline'].float().numpy()
    root = lambda x: x[:, [11, 12]].mean(1, keepdims=True)
    errors = np.linalg.norm(pose-target, axis=-1).mean(-1)
    relative = np.linalg.norm((pose-root(pose))-(target-root(target)), axis=-1).mean(-1)
    roots = np.linalg.norm(root(pose)-root(target), axis=-1).ravel()
    change = np.linalg.norm(pose-full, axis=-1).mean(-1)
    full_errors = np.linalg.norm(full-target, axis=-1).mean(-1)
    for name, mask in masks.items():
        item = result['conditions'].setdefault(name, np.zeros(6, np.float64))
        item += [mask.sum(), errors[mask].sum(dtype=np.float64), relative[mask].sum(dtype=np.float64),
                 roots[mask].sum(dtype=np.float64), change[mask].sum(dtype=np.float64),
                 (errors[mask]-full_errors[mask]).sum(dtype=np.float64)]
    truth_direction = head_foot_direction(target)
    opposite = lambda x: horizontal & ((head_foot_direction(x)*truth_direction).sum(-1) < -.5)
    wrong, base_wrong, full_wrong = opposite(pose), opposite(base), opposite(full)
    result['direction'] += np.array([horizontal.sum(), wrong.sum(), (base_wrong & ~wrong).sum(),
                                    (~base_wrong & wrong).sum(), (full_wrong & ~wrong).sum(),
                                    (~full_wrong & wrong).sum()])
    group = result['groups'].setdefault(stream['group'], np.zeros(3, np.float64))
    group += [len(target), errors.sum(dtype=np.float64), full_errors.sum(dtype=np.float64)]
    result['joint_error'] += np.linalg.norm(pose-target, axis=-1).sum(0, dtype=np.float64)
    result['max_prediction_difference'] = max(result['max_prediction_difference'], float(np.abs(pose-full).max()))


def finish(result):
    conditions = {name: dict(count=int(v[0]), mpjpe=v[1]/v[0], relative_mpjpe=v[2]/v[0],
                            root_error=v[3]/v[0], prediction_change_from_full=v[4]/v[0],
                            delta_mpjpe_from_full=v[5]/v[0]) for name, v in result['conditions'].items() if v[0]}
    groups = np.stack(list(result['groups'].values()))
    generator = np.random.default_rng(42)
    picks = generator.integers(len(groups), size=(2000, len(groups)))
    selected = groups[picks]
    delta = (selected[..., 1]-selected[..., 2]).sum(1)/selected[..., 0].sum(1)
    return dict(conditions=conditions, direction=dict(zip(
        ('horizontal_frames', 'opposite_frames', 'removed_vs_p4', 'introduced_vs_p4',
         'removed_vs_full', 'introduced_vs_full'), result['direction'].tolist())),
        delta_mpjpe_group_bootstrap_ci95=np.quantile(delta, [.025, .975]).tolist(),
        joint_mpjpe=(result['joint_error']/conditions['all']['count']).tolist(),
        groups={name: dict(count=int(v[0]), mpjpe=v[1]/v[0], delta_from_full=(v[1]-v[2])/v[0])
                for name, v in result['groups'].items()},
        max_prediction_difference=result['max_prediction_difference'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    # Pin an immutable snapshot while the independent training job may update best.pth.
    checkpoint = torch.load(experiment/'best.pth', map_location='cpu', weights_only=False)
    if checkpoint.get('training_mode') == 'joint_finetune':
        raise ValueError('Joint fine-tuning requires online P4 encoding; frozen-token ablations do not apply. '
                         'Use run.analyze_pose_refiner for the matching online evaluation.')
    output = experiment/f"ablation_epoch{checkpoint['epoch']}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.smoke:
        output = output.with_name('smoke_'+output.name)
    output.mkdir(exist_ok=False)
    pinned = {k:v for k,v in checkpoint.items() if k != 'optimizer_state_dict'}
    torch.save(pinned, output/'checkpoint.pth')
    shutil.copy2(experiment/'P4JointMemory.py', output/'P4JointMemory.py')
    shutil.copy2(Path(__file__), output/'analyze_p4_joint_memory_ablation.py')
    shutil.copy2(experiment/'evidence_manifest.json', output/'evidence_manifest.json')
    spec = importlib.util.spec_from_file_location('ablated_joint_memory', output/'P4JointMemory.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    device = torch.device(args.device)
    model = module.P4JointMemory(**checkpoint['architecture']).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    original = copy.deepcopy(model)
    config = checkpoint['config']
    baseline_path = Path(config['baseline_experiment'])
    if not baseline_path.is_absolute():
        baseline_path = Path(__file__).resolve().parents[1]/baseline_path
    original.load_p4_head(torch.load(baseline_path/'checkpoint'/config['checkpoint'], map_location=device,
                                   weights_only=False)['model_state_dict'])
    no_joint_id = copy.deepcopy(model)
    with torch.no_grad():
        no_joint_id.joint_queries.zero_()
    results = {name: dict(conditions={}, direction=np.zeros(6, np.int64), groups={},
                          joint_error=np.zeros(17), max_prediction_difference=0.) for name in VARIANTS}
    results['p4_cached'] = dict(conditions={}, direction=np.zeros(6, np.int64), groups={},
                                joint_error=np.zeros(17), max_prediction_difference=0.)
    manifest = json.loads((output/'evidence_manifest.json').read_text())
    started = time.monotonic()
    print('Snapshot',checkpoint['epoch'],'Output',output,flush=True)
    for index, (group, info) in enumerate(manifest['val'].items()):
        streams = torch.load(info['cache'], map_location='cpu', weights_only=False)
        if args.smoke:
            streams = [{k:v[:64] if torch.is_tensor(v) else v for k,v in s.items()} for s in streams[:1]]
        for stream in streams:
            full = predict(model, stream, 'full', device)
            target, masks, horizontal = masks_for(stream, config['warmup_frames'])
            for variant in (*VARIANTS, 'p4_cached'):
                selected = original if variant == 'original_head_only' else no_joint_id if variant == 'no_joint_queries' else model
                pose = (stream['baseline'].float().numpy() if variant == 'p4_cached' else full if variant == 'full'
                        else predict(selected, stream, variant, device))
                accumulate(results[variant], stream, pose, full, target, masks, horizontal)
        if index % 10 == 0:
            print(index+1,'groups',group,'seconds',round(time.monotonic()-started,1),flush=True)
        if args.smoke:
            break
    reports = {name: finish(value) for name, value in results.items()}
    reference = reports['full']['conditions']['all']
    if reports['paired_permutation']['max_prediction_difference'] > 1e-4:
        raise AssertionError('Paired permutation control failed')
    if not args.smoke:
        history = json.loads((experiment/'history.json').read_text())
        epoch = next(v for v in history if v['epoch'] == checkpoint['epoch'])
        if abs(reference['mpjpe']-epoch['all']['decoded']['mpjpe']) > 1e-5:
            raise AssertionError('Unmodified snapshot does not reproduce recorded validation MPJPE')
    report = dict(epoch=checkpoint['epoch'], smoke=args.smoke, device=str(device),
                  checkpoint_sha256=hashlib.sha256((output/'checkpoint.pth').read_bytes()).hexdigest(),
                  definitions=dict(variants={**VARIANTS, 'p4_cached':'Original saved BF16 P4 predictions'},
                      protocol='All same validation frames, no augmentation; decoder causal history resets per stream',
                      limitation='Inference interventions can shift distributions; not separately retrained architecture ablations',
                      confidence_interval='Paired bootstrap over complete groups, frame-weighted; 2000 resamples, seed42'),
                  elapsed_seconds=time.monotonic()-started, results=reports)
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['| Intervention | MPJPE cm | Delta vs full mm | Horizontal cm | Opposite frames | Mean prediction change mm |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in reports.items():
        all_frames = result['conditions']['all']
        horizontal = result['conditions'].get('horizontal',{}).get('mpjpe',float('nan'))
        lines.append(f"| {name} | {100*all_frames['mpjpe']:.4f} | {1000*all_frames['delta_mpjpe_from_full']:+.4f} | "
                     f"{100*horizontal:.4f} | {result['direction']['opposite_frames']} | {1000*all_frames['prediction_change_from_full']:.4f} |")
    text='\n'.join(lines)+'\n'
    (output/'summary.md').write_text(text)
    print(text,flush=True)
    print('Complete',output,round(report['elapsed_seconds'],1),'seconds',flush=True)


if __name__ == '__main__':
    main()
