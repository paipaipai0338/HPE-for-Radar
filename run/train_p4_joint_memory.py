"""Train point-token joint memory before coordinate decoding; no pose-flip postprocessing."""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from models.P4JointMemory.P4JointMemory import P4JointMemory, reset_state, detach_state
from run.utils.load_config import load_config
from run.utils.set_seed import set_seed
from run.utils.p4_sequence_data import HPE_Dataset, StatefulBatches, extract_group, group_fingerprint, load_frontend

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('tokens', 'xyz', 'features', 'quality', 'rotation', 'translation', 'baseline', 'target')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def prepare(config, device):
    experiment = resolve(config['baseline_experiment'])
    checkpoint = experiment/'checkpoint'/config['checkpoint']
    paths = [checkpoint, experiment/'config/P4Transformer.py', experiment/'config/model_config.yaml',
             experiment/'config/config.yaml', ROOT/'models/P4Transformer/transformer_helper.py',
             ROOT/'models/P4Transformer/pointnet2_utils.py', ROOT/'run/utils/process_one_epoch.py',
             ROOT/'data2datasets/dataset_for_all_task.py',
             ROOT/'run/utils/p4_sequence_data.py']
    binding = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    training = load_config(experiment/'config/config.yaml')
    if training['data']['pointcloud_source'] != 'fftgen' or training['data']['split_method'] != 'person':
        raise ValueError('Expected FFTGen P4 with the current person split')
    streams, manifest = {}, {}
    frontend = None
    for mode in ('train', 'val'):
        d = training['data']
        dataset = HPE_Dataset(root_path=d['root_path'], packed_data_root=d['packed_data_root'],
                              sensor_config=d['sensor_config'], mode=mode, base_source=d['base_source'],
                              split_method='person', ratio=d['ratio'], T=d['T'], acc_frame=d['acc_frame'],
                              preload_cache=False, enable_action=False, pointcloud_source='fftgen',
                              val_person_id=d.get('val_person_id', '4'), window_stride=1,
                              train_groups=d.get('train_groups') if mode == 'train' else None)
        groups = defaultdict(list)
        for index, group in enumerate(dataset.sample_groups):
            groups[group].append(index)
        if config.get('smoke'):
            groups = dict(list(groups.items())[:1])
        streams[mode], manifest[mode] = [], {}
        for group, indices in tqdm(groups.items(), desc=f'{mode} spatial tokens'):
            if config.get('smoke'):
                indices = indices[:128]
            fingerprint = group_fingerprint(dataset, indices, binding, dict(
                data=d, task=training['task'], gap=config['max_frame_gap_seconds'],
                center=config['track_max_center_distance'], joints=config['track_max_joint_distance']))
            cache = resolve(config['cache_dir'])/mode/group/f'{fingerprint}.pt'
            if cache.is_file():
                values = torch.load(cache, map_location='cpu', weights_only=False)
            else:
                if frontend is None:
                    frontend, _, _ = load_frontend(experiment, checkpoint, device)
                values = extract_group(frontend, dataset, indices, training, config, device)
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix('.tmp')
                torch.save(values, temporary)
                temporary.replace(cache)
            streams[mode].extend(values)
            manifest[mode][group] = dict(cache=str(cache), frames=sum(len(s['features']) for s in values),
                                         streams=len(values), windows=len(indices))
        del dataset
        if not streams[mode]:
            raise ValueError(f'No {mode} evidence; inspect alignment/validity thresholds')
    if set(manifest['train']) & set(manifest['val']):
        raise ValueError('Training and validation groups overlap')
    del frontend
    return streams, manifest, binding


def augment_tokens(batch, valid, config):
    inputs = dict(batch)
    batch_size, length, count = batch['tokens'].shape[:3]
    device = batch['tokens'].device
    keep = (torch.rand(batch_size, length, count, device=device) >= config['point_token_dropout']) & valid[..., None]
    # Keep the first eight frames available for establishing semantic identities.
    eligible = torch.arange(length, device=device)[None] >= 8
    scores = torch.rand(batch_size, length, device=device).masked_fill(~(valid & eligible), -1.)
    start = scores.argmax(-1, keepdim=True)
    duration = torch.randint(1, config['missing_block_max_frames']+1, (batch_size, 1), device=device)
    selected = torch.rand(batch_size, 1, device=device) < config['missing_block_probability']
    positions = torch.arange(length, device=device)[None]
    missing = selected & eligible & valid & (positions >= start) & (positions < start+duration)
    keep &= ~missing[..., None]
    inputs['token_mask'] = keep
    inputs['quality'] = batch['quality'].clone()
    inputs['quality'][~keep.any(-1)] = 0.
    inputs['quality'][..., -1] = batch['quality'][..., -1]
    return inputs, int(missing.sum())


def losses(pose, target, valid, config):
    torso = target[..., [5, 6], :].mean(-2)-target[..., [11, 12], :].mean(-2)
    horizontal = torso[..., 2].abs()/torso.norm(dim=-1).clamp_min(1e-6) < .4
    weights = torch.where(horizontal, config['horizontal_pose_weight'], 1.)*valid
    errors = (pose-target).norm(dim=-1).mean(-1)
    mpjpe = (errors*weights).sum()/weights.sum().clamp_min(1)
    predicted = pose[..., 0, :]-pose[..., [15, 16], :].mean(-2)
    truth = target[..., 0, :]-target[..., [15, 16], :].mean(-2)
    direction_mask = valid & (truth.norm(dim=-1) > .3)
    direction_error = 1-F.cosine_similarity(predicted, truth, dim=-1, eps=1e-6)
    direction_loss = direction_error.masked_fill(~direction_mask, 0).sum()/direction_mask.sum().clamp_min(1)
    return mpjpe+config['direction_loss_weight']*direction_loss, errors[valid].sum(), int(valid.sum())


def pose_errors(pose, target):
    root, truth_root = [x[..., [11, 12], :].mean(-2, keepdim=True) for x in (pose, target)]
    return torch.stack(((pose-target).norm(dim=-1).mean(-1),
                        ((pose-root)-(target-truth_root)).norm(dim=-1).mean(-1),
                        (root-truth_root).norm(dim=-1).squeeze(-1)), -1)


@torch.inference_mode()
def evaluate(model, streams, device, warmup):
    model.eval()
    totals = {name: torch.zeros(2, 3, device=device) for name in ('all', 'cold', 'warm')}
    counts = dict.fromkeys(totals, 0)
    direction_counts = torch.zeros(5, dtype=torch.long, device=device)
    groups = {}
    for stream in streams:
        state = None
        for start in range(0, len(stream['features']), 256):
            inputs = {k: stream[k][None, start:start+256].to(device) for k in FIELDS}
            pose, state = model(inputs, state)
            target = inputs['target']
            errors = torch.stack((pose_errors(inputs['baseline'], target), pose_errors(pose, target)))[:, 0]
            positions = torch.arange(start, start+errors.shape[1], device=device)
            masks = dict(all=torch.ones_like(positions, dtype=torch.bool), cold=positions<warmup, warm=positions>=warmup)
            for name, mask in masks.items():
                totals[name] += errors[:, mask].sum(1)
                counts[name] += int(mask.sum())
            group = groups.setdefault(stream['group'], [torch.zeros(2, 3, device=device), 0])
            group[0] += errors.sum(1)
            group[1] += errors.shape[1]
            torso = target[..., [5, 6], :].mean(-2)-target[..., [11, 12], :].mean(-2)
            gt_vector = target[..., 0, :2]-target[..., [15, 16], :2].mean(-2)
            horizontal = ((torso[..., 2].abs()/torso.norm(dim=-1).clamp_min(1e-6) < .4)
                          & (gt_vector.norm(dim=-1) > .5))
            wrong = []
            for current in (inputs['baseline'], pose):
                vector = current[..., 0, :2]-current[..., [15, 16], :2].mean(-2)
                wrong.append(horizontal & (F.cosine_similarity(vector, gt_vector, dim=-1) < -.5))
            a, b = wrong
            direction_counts += torch.stack((horizontal.sum(), a.sum(), b.sum(), (a & ~b).sum(), (~a & b).sum()))
    def report(total, count):
        return dict(count=count, **{name: dict(zip(('mpjpe', 'relative_mpjpe', 'root_error'), (total[i]/count).tolist()))
                                   for i, name in enumerate(('baseline', 'decoded'))}) if count else dict(count=0)
    return {**{key: report(value, counts[key]) for key, value in totals.items()},
            'direction': dict(zip(('horizontal_frames', 'baseline_opposite', 'decoded_opposite',
                                   'removed_opposite', 'introduced_opposite'), direction_counts.tolist())),
            'groups': {key: report(*value) for key, value in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'run/p4_joint_memory.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if any(type(config[k]) is not int or config[k] < 1 for k in ('batch_size', 'chunk_length', 'epochs', 'missing_block_max_frames')):
        raise ValueError('Require positive integer batch/chunk/epoch/occlusion lengths')
    if any(not 0 <= config[k] <= 1 for k in ('point_token_dropout', 'missing_block_probability')):
        raise ValueError('Dropout probabilities must be in [0,1]')
    if not 0 <= config['warmup_frames'] < config['chunk_length'] or config['horizontal_pose_weight'] < 1:
        raise ValueError('Invalid warmup or horizontal pose weight')
    if config['direction_loss_weight'] < 0 or min(config['learning_rate'], config['head_learning_rate']) <= 0:
        raise ValueError('Invalid loss weight or learning rate')
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    config['smoke'] = args.smoke
    if args.smoke:
        config['epochs'] = 1
    streams, manifest, binding = prepare(config, device)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    architecture = load_config(ROOT/'models/P4JointMemory/model_config.yaml')
    model = P4JointMemory(**architecture).to(device)
    checkpoint = resolve(config['baseline_experiment'])/'checkpoint'/config['checkpoint']
    model.load_p4_head(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
    head_names = ('head_norm.', 'head_hidden.', 'joint_weight', 'joint_bias')
    heads = [p for name, p in model.named_parameters() if name.startswith(head_names)]
    others = [p for name, p in model.named_parameters() if not name.startswith(head_names)]
    optimizer = torch.optim.AdamW([dict(params=heads, lr=config['head_learning_rate']),
                                   dict(params=others, lr=config['learning_rate'])], weight_decay=config['weight_decay'])
    output = resolve(config['output_dir'])/(('smoke_' if args.smoke else '')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output/'model_config.yaml').write_text(yaml.safe_dump(architecture))
    (output/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(ROOT/'models/P4JointMemory/P4JointMemory.py', output/'P4JointMemory.py')
    shutil.copy2(Path(__file__), output/'train_p4_joint_memory.py')
    history = [dict(epoch=0, **evaluate(model, streams['val'], device, config['warmup_frames']))]
    best, stale = history[0]['all']['decoded']['mpjpe'], 0
    def save(name, epoch):
        torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        config=config, architecture=architecture, baseline_binding=binding, best_metric=best), output/name)
    save('best.pth', 0)
    print(f"Output: {output}\nInitial same-frame metrics: {history[0]['all']}", flush=True)
    for epoch in range(1, config['epochs']+1):
        model.train()
        state = None
        total, count, missing = 0., 0, 0
        for batch, valid, reset in tqdm(StatefulBatches(streams['train'], config['chunk_length'], config['batch_size'], FIELDS), desc=f'joint memory {epoch}'):
            state = reset_state(state, reset.to(device))
            batch = {k: v.to(device) for k, v in batch.items()}
            valid = valid.to(device)
            observed, n = augment_tokens(batch, valid, config)
            pose, state = model(observed, state)
            state = detach_state(state)
            loss, error, size = losses(pose, batch['target'], valid, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint-memory loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(error.detach()); count += size; missing += n
        report = dict(epoch=epoch, train_augmented_mpjpe=total/count, train_frames=count, missing_train_frames=missing,
                      **evaluate(model, streams['val'], device, config['warmup_frames']))
        history.append(report)
        score = report['all']['decoded']['mpjpe']
        if score < best:
            best, stale = score, 0
            save('best.pth', epoch)
        else:
            stale += 1
        save('last.pth', epoch)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}: train_augmented={total/count:.6f} | all={report['all']} | direction={report['direction']}", flush=True)
        if stale >= config['early_stopping_patience']:
            break


if __name__ == '__main__':
    main()
