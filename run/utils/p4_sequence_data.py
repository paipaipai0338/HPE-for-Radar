"""P4 spatial token extraction, conservative GT association and continuous stream batches."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data2datasets.dataset_for_all_task import HPE_Dataset, PackedFrame, collate_fn
from run.utils.load_config import load_config
from run.utils.process_one_epoch import prepare_person_pointcloud, get_autocast_dtype


def match_people(previous, current, center_limit, joint_limit):
    """Associate GT poses; reject ambiguous crossings rather than mix memories."""
    if not len(previous) or not len(current):
        return []
    centers = np.linalg.norm(previous[:, None, [11, 12]].mean(2) - current[None, :, [11, 12]].mean(2), axis=-1)
    joints = np.linalg.norm(previous[:, None] - current[None], axis=-1).mean(-1)
    valid = (centers <= center_limit) & (joints <= joint_limit)
    cost = centers + joints
    rows, cols = linear_sum_assignment(np.where(valid, cost, 1e6))
    matches = []
    for row, col in zip(rows, cols):
        # shortcut: discard ambiguous close crossings; a detector tracker is needed for deployment.
        if valid[row, col] and valid[row].sum() == 1 and valid[:, col].sum() == 1:
            matches.append((int(row), int(col)))
    return matches


def align_window(samples, config):
    """Follow the first-frame GT identities within each T16 window."""
    target = samples['gt_for_high']
    poses, masks = target['padded'].numpy(), target['mask'].numpy()
    aligned = {key: torch.zeros_like(target[key]) for key in ('padded', 'mask', 'bbox')}
    for batch in range(len(poses)):
        previous = np.flatnonzero(masks[batch, 0])
        slots = previous.copy()
        for key in aligned:
            aligned[key][batch, 0] = target[key][batch, 0]
        for frame in range(1, poses.shape[1]):
            current = np.flatnonzero(masks[batch, frame])
            pairs = match_people(poses[batch, frame-1, previous], poses[batch, frame, current],
                                 config['track_max_center_distance'], config['track_max_joint_distance'])
            new_previous, new_slots = [], []
            for old, new in pairs:
                slot, index = int(slots[old]), int(current[new])
                for key in aligned:
                    aligned[key][batch, frame, slot] = target[key][batch, frame, index]
                new_previous.append(index)
                new_slots.append(slot)
            previous, slots = np.asarray(new_previous, dtype=int), np.asarray(new_slots, dtype=int)
    return {**samples, 'gt_for_high': aligned}


class StreamBuilder:
    def __init__(self, config):
        self.config, self.streams, self.active = config, [], []

    def append(self, time, poses, observations):
        candidates = [(stream, stamp, pose) for stream, stamp, pose in self.active
                      if 0 < time - stamp <= self.config['max_frame_gap_seconds']]
        pairs = match_people(np.asarray([x[2] for x in candidates]).reshape(-1, 17, 3), poses,
                             self.config['track_max_center_distance'], self.config['track_max_joint_distance'])
        assigned = {new: candidates[old] for old, new in pairs}
        active = []
        for index, (pose, observation) in enumerate(zip(poses, observations)):
            item = {key: value.clone() for key, value in observation.items()}
            item['timestamp'] = torch.tensor(time, dtype=torch.float64)
            if index in assigned:
                stream, previous, _ = assigned[index]
                item['quality'][-1] = time - previous
            else:
                stream = len(self.streams)
                self.streams.append([])
                item['quality'][-1] = 0.
            self.streams[stream].append(item)
            active.append((stream, time, pose.copy()))
        self.active = active

    def finish(self, group):
        return [{**{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
                 'group': group} for rows in self.streams]


def load_frontend(experiment, checkpoint, device):
    training = load_config(experiment / 'config/config.yaml')
    if (training['model']['name'] != 'P4Transformer' or training['task'].get('coordinate') != 'high'
            or training['task'].get('center_on_pointcloud') or training['task'].get('center_xy_in_low')
            or training['task']['input'] != 'radar_high_pc' or training['task']['output'] != 'gt_for_high'):
        raise ValueError('Refiner requires an uncentered high-coordinate P4Transformer frontend')
    source = experiment / 'config/P4Transformer.py'
    spec = importlib.util.spec_from_file_location('frozen_p4_frontend', source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeError('P4 frontend requires pointnet2_ops._ext built for this Python/PyTorch/CUDA. '
                           'Use the original working P4 environment; --help and refiner unit tests do not require this extension.') from error
    architecture = load_config(experiment / 'config/model_config.yaml')
    model = module.P4Transformer(**architecture).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    return model, training, architecture


def group_fingerprint(dataset, indices, binding, config):
    paths, refs = set(), []
    for key, windows in dataset.data_path_list.items():
        for index in indices:
            for ref in windows[index]:
                refs.append((key, repr(ref)))
                paths.add(Path(ref.pack_path if isinstance(ref, PackedFrame) else ref))
        if key in dataset.acc_path_list:
            for index in indices:
                for ref in dataset.acc_path_list[key][index]:
                    paths.add(Path(ref.pack_path if isinstance(ref, PackedFrame) else ref))
                    refs.append((key, repr(ref)))
    date = dataset.sample_groups[indices[0]].split('/')[0]
    for root in (dataset.root_path, dataset.packed_data_root):
        if root is not None:
            paths.update((root / date / 'calib').glob('*.npz'))
    stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(paths)]
    payload = dict(version=1, binding=binding, refs=refs, sources=stats, data=config)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def extract_group(model, dataset, indices, training, config, device):
    from functools import partial
    group = dataset.sample_groups[indices[0]]
    loader = DataLoader(Subset(dataset, indices), batch_size=config['extraction_batch_size'],
                        shuffle=False, num_workers=config['num_workers'], pin_memory=device.type == 'cuda',
                        collate_fn=partial(collate_fn, max_people=training['data']['max_people']))
    captured, spatial, anchors = [], [], []
    handle = model.mlp_head_pose.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    spatial_handle = model.transformer.register_forward_hook(lambda module, args, output: spatial.append(output.detach()))
    anchor_handle = model.tube_embedding.register_forward_hook(lambda module, args, output: anchors.append(output[0].detach()))
    builder = StreamBuilder(config)
    offset = 0
    data = {**training['data'], 'augmentation': {}}
    task = training['task']
    precision = get_autocast_dtype(task['precision']) if device.type == 'cuda' else None
    try:
        for samples in tqdm(loader, desc=group, leave=False):
            source_indices = indices[offset:offset+len(samples['group_id'])]
            offset += len(source_indices)
            times = [[dataset._frame_timestamp_ns(ref) * 1e-9 for ref in dataset.data_path_list['radar_high_pc'][i]]
                     for i in source_indices]
            samples = align_window(samples, config)
            inputs, gt, valid, cropped, mask, _ = prepare_person_pointcloud(samples, task, data, device, 'val')
            complete = inputs['mask'].any(-1).all(-1)
            slots = valid.nonzero()
            complete &= gt['mask'].permute(0, 2, 1)[valid].all(-1)
            for index, timeline in enumerate(times):
                delta = np.diff(timeline)
                if np.any(delta <= 0) or np.any(delta > config['max_frame_gap_seconds']):
                    complete[slots[:, 0] == index] = False
            observations = [[] for _ in times]
            targets = [[] for _ in times]
            if complete.any():
                inputs = {key: value[complete] for key, value in inputs.items()}
                with torch.autocast(device.type, dtype=precision, enabled=precision is not None):
                    pose = model(inputs)['pose'][:, -1, 0].float()
                features = captured.pop()[:, -1].float()
                anchor_xyz = anchors.pop()[:, -1].float()
                token_features = spatial.pop().reshape(len(features), inputs['input'].shape[1], anchor_xyz.shape[1], -1)[:, -1]
                selected = slots[complete]
                for row, (b, person) in enumerate(selected.tolist()):
                    rotation = samples['high_to_low_R'][b, -1].to(device)
                    translation = samples['high_to_low_t'][b, -1].to(device)
                    baseline = pose[row] @ rotation.T + translation
                    target = gt['padded'][b, -1, person] @ rotation.T + translation
                    points = cropped[b, person, -1][mask[b, person, -1]]
                    count = len(points)
                    velocity = points[:, 3].abs()
                    # Current-frame count before temporal accumulation, using the same GT crop.
                    ref = dataset.data_path_list['radar_high_pc'][source_indices[b]][-1]
                    raw = dataset._load_packed_frame(ref) if isinstance(ref, PackedFrame) else np.load(ref)
                    bounds = gt['bbox'][b, -1, person].cpu().numpy()
                    current = ((raw[:, :3] >= bounds[:3]) & (raw[:, :3] <= bounds[3:])).all(-1).sum()
                    extent = points[:, :3].amax(0) - points[:, :3].amin(0)
                    quality = torch.tensor([np.log1p(current)/np.log(201), np.log1p(count)/np.log(201),
                                            float(velocity.mean()), float(velocity.max()), float(extent.norm()), 0.], dtype=torch.float32)
                    observations[b].append(dict(features=features[row].cpu(), baseline=baseline.cpu(),
                                                 target=target.cpu(), quality=quality,
                                                 tokens=token_features[row].to(torch.bfloat16).cpu(),
                                                 xyz=(anchor_xyz[row] @ rotation.T + translation).cpu(),
                                                 rotation=rotation.cpu(), translation=translation.cpu()))
                    targets[b].append(target.cpu().numpy())
            for time, poses, evidence in zip(times, targets, observations):
                builder.append(time[-1], np.asarray(poses).reshape(-1, 17, 3), evidence)
    finally:
        handle.remove()
        spatial_handle.remove()
        anchor_handle.remove()
    return builder.finish(group)


class StatefulBatches:
    """One ordered trajectory per batch lane; padding is never supervised."""
    def __init__(self, streams, length, batch_size, keys):
        self.streams, self.length, self.batch_size, self.keys = streams, length, batch_size, keys
        if not streams or min(length, batch_size) < 1:
            raise ValueError('Require nonempty streams and positive batch/chunk lengths')

    def __iter__(self):
        pending = iter(torch.randperm(len(self.streams)).tolist())
        lanes = [None] * self.batch_size
        while True:
            reset = torch.ones(self.batch_size, dtype=torch.bool)
            valid = torch.zeros(self.batch_size, self.length, dtype=torch.bool)
            batch = {key: self.streams[0][key].new_zeros(self.batch_size, self.length, *self.streams[0][key].shape[1:])
                     for key in self.keys}
            for lane, current in enumerate(lanes):
                if current is None:
                    stream = next(pending, None)
                    current = None if stream is None else (stream, 0)
                if current is None:
                    continue
                stream, start = current
                values = self.streams[stream]
                size = min(self.length, len(values['features'])-start)
                reset[lane] = start == 0
                valid[lane, :size] = True
                for key in batch:
                    batch[key][lane, :size] = values[key][start:start+size]
                lanes[lane] = (stream, start+size) if start+size < len(values['features']) else None
            if not valid.any():
                return
            yield batch, valid, reset
