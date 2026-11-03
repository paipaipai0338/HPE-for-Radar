"""Collect complete-window last frames, apply Guard, then score each group."""
import json
from pathlib import Path

import numpy as np

from run.utils.temporal_guard import frame_mpjpe, guard_pose_records
from utils.skeleton import get_pose_layout


def guard_frame_metadata(points, poses, path, frame_index):
    """Count original current-frame GT-crop points before accumulation or sampling."""
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError('Guard requires original [N,6] xyz/v/power/TLV points')
    points = points[np.isfinite(points).all(axis=1)]
    total, dynamic = {}, {}
    for person, pose in enumerate(poses):
        lo, hi = pose.min(0) - .3, pose.max(0) + .3
        tags = points[((points[:, :3] >= lo) & (points[:, :3] <= hi)).all(-1), -1]
        total[person] = len(tags)
        dynamic[person] = int(np.isclose(tags, 1.).sum())
    return dict(path=str(path), frame_index=int(frame_index),
                total_point_counts=total, dynamic_point_counts=dynamic)


class GroupGuard:
    def __init__(self, pose_format, window_stride=1):
        self.pose_format = pose_format
        self.joints = len(get_pose_layout(pose_format)['names'])
        self.window_stride = window_stride
        self.records = {}
        self.labels = json.loads((Path(__file__).resolve().parents[2] /
                                  'data2datasets/group_labels.json').read_text())['groups']

    def add_batch(self, samples, pre, valid_mask, task, pointcloud_center=None):
        from utils.MOCAP21 import coco_to_mocap

        metadata = samples.get('guard_metadata')
        group_ids = samples.get('group_id')
        if metadata is None or group_ids is None or len(metadata) != len(group_ids):
            raise ValueError('Guard validation requires group_id and original current-frame guard_metadata')
        truth = samples[task['output']]['padded'][:, -1].detach().cpu().contiguous()
        if self.pose_format == 'mocap':
            truth = coco_to_mocap(truth)
        truth_mask = samples[task['output']]['mask'][:, -1].cpu().numpy().copy()
        truth = truth.numpy()
        gt_low = samples['gt_for_low']['padded'][:, -1].cpu().numpy().copy()
        prediction = None
        complete = np.zeros_like(truth_mask)
        if pre is not None:
            prediction = pre['pose'][:, -1].detach().float().cpu()
            if prediction.shape != truth.shape:
                raise ValueError('GT-cropped Guard validation requires aligned person output slots')
            if pointcloud_center is not None:
                prediction = prediction + pointcloud_center[:, -1].detach().cpu().unsqueeze(-2)
            prediction = prediction.numpy()
            complete = valid_mask.detach().cpu().bool().all(dim=1).numpy() & truth_mask
        for b, (group, meta) in enumerate(zip(group_ids, metadata)):
            ids = np.flatnonzero(complete[b])
            record = dict(meta, gt=truth[b], gt_low=gt_low[b],
                gt_crop_poses_raw=prediction[b, ids].copy() if prediction is not None else np.empty((0, self.joints, 3), np.float32),
                gt_crop_pose_indices=ids, gt_crop_track_ids=np.flatnonzero(truth_mask[b]),
                gt_training_window=bool(len(ids)))
            records = self.records.setdefault(group, {})
            index = meta['frame_index']
            if index in records:
                raise ValueError(f'Duplicate Guard validation frame: {group}/{index}')
            records[index] = record

    def epoch_end(self):
        raw_groups, guarded_groups = {}, {}
        for group, frames in self.records.items():
            records, previous = [], None
            for index, record in sorted(frames.items()):
                if previous is not None and index != previous + self.window_stride:
                    records.append(dict(path=record['path'], gt_crop_poses_raw=np.empty((0, self.joints, 3)),
                        gt_crop_pose_indices=[], gt_crop_track_ids=[], dynamic_point_counts={}, total_point_counts={}))
                records.append(record)
                previous = index
            guard_pose_records(records, 'gt_crop_', self.pose_format)
            sums = {stage: np.zeros(6, np.float64) for stage in ('raw', 'guard')}
            statuses = dict(accepted=0, confirmed=0, held=0, excluded=0)
            for record in records:
                if not record.get('gt_training_window', False):
                    continue
                for status in record['gt_crop_pose_status'].values():
                    statuses[status] += 1
                roots = record['gt_low'][:, [11, 12], :2].mean(1)
                inside = np.square((roots - [2., 0.]) / [1.6, 2.4]).sum(-1) <= 1.
                for stage, poses_key, ids_key in (
                        ('raw', 'gt_crop_poses_raw', 'gt_crop_pose_indices'),
                        ('guard', 'gt_crop_poses', 'gt_crop_pose_track_ids')):
                    people = np.asarray(record[ids_key], dtype=int)
                    poses = record[poses_key]
                    for inside_only in (False, True):
                        selected = people[inside[people]] if inside_only else people
                        pred = poses[np.isin(people, selected)]
                        gt = record['gt'][selected]
                        error, count, *_ = frame_mpjpe(pred, gt, pose_format=self.pose_format)
                        centered, *_ = frame_mpjpe(pred, gt, centered=True, pose_format=self.pose_format)
                        if count:
                            offset = 3 if inside_only else 0
                            sums[stage][offset:offset + 3] += [error / 1000 * count, centered / 1000 * count, count]
            for stage, output in (('raw', raw_groups), ('guard', guarded_groups)):
                total, centered, count, inside, centered_inside, count_inside = sums[stage]
                output[group] = dict(label=self.labels.get(group, {}).get('label', 'unlabelled'),
                    mpjpe=total / count if count else None, centered_mpjpe=centered / count if count else None,
                    count=int(count), mpjpe_inside=inside / count_inside if count_inside else None,
                    centered_mpjpe_inside=centered_inside / count_inside if count_inside else None,
                    count_inside=int(count_inside))
            guarded_groups[group]['statuses'] = statuses
        self.records.clear()
        return raw_groups, guarded_groups
