"""读取一个未参与训练/验证的组，模拟部署时的数据输入。"""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from scipy import signal
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from data2datasets.dataset_for_all_task import HPE_Dataset, collate_fn
from metrics.detection import (
    get_acc,
    get_bbox_iou,
    get_bbox_l1,
    get_objectness,
    pairwise_axis_aligned_iou_3d,
    get_precision,
    get_recall,
    paired_axis_aligned_iou_3d,
)
from metrics.pose import get_bone_length, get_mpjpe, get_pampjpe
from preprocess.actionprocess import LABEL_NAMES, classify_actions
from run.utils.build_model import build_model
from run.utils.checkpoint import load_model_checkpoint
from run.utils.plot_fig import plt_fig


ROOT_PATH = Path("/mnt/huawei")
DATE = "20260722"
GROUP = "group_026"
STAGE = "analysis"  # inference / analysis
T = 8
BATCH_SIZE = 8
IOU_THRESHOLD = 0.7
POINT_CLOUD_RANGE = [0.0, -3.0, -2.0, 6.0, 3.0, 2.0]
RESULT_PATH = Path("/home/pai/Huawei/deploy/result_selected_group.pth")
VIDEO_PATH = Path("/home/pai/Huawei/deploy/selected_group.mp4")
VIDEO_FPS = 10

# 姿态时序平滑与骨长约束参数。
SMOOTH_ALPHA = 0.35
MAX_MATCH_DISTANCE = 0.50
MAX_MISSING = 8
MIN_VALID_JOINTS = 5
MAX_INTERP_GAP = 8
MEDFILT_KERNEL = 7
VELOCITY_THRESHOLD = 0.35
BONE_LENGTH_WEIGHT = 0.65
BONE_ITERS = 2

DIRECTED_BONES = [
    (11, 12),
    (11, 5),
    (12, 6),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 0),
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
]

model_pose_path = Path(
    "/home/pai/Huawei/experiments/P4Transformer/20260728_203033"
)


@dataclass
class TrackState:
    track_id: int
    pose: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    missing_count: int = 0
    age: int = 0


def valid_joint_mask(pose):
    return np.all(np.isfinite(pose), axis=1)


def pose_center(pose):
    mask = valid_joint_mask(pose)
    if not np.any(mask):
        return None
    return np.median(pose[mask], axis=0)


def pose_distance(a, b):
    mask = valid_joint_mask(a) & valid_joint_mask(b)
    if not np.any(mask):
        center_a = pose_center(a)
        center_b = pose_center(b)
        if center_a is None or center_b is None:
            return np.inf
        return float(np.linalg.norm(center_a - center_b))
    return float(np.median(np.linalg.norm(a[mask] - b[mask], axis=1)))


def assign_detections(tracks, detections, max_match_distance):
    if not tracks or not detections:
        return []

    cost = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    for track_idx, track in enumerate(tracks):
        predicted_pose = track.pose.copy()
        mask = valid_joint_mask(predicted_pose)
        predicted_pose[mask] += track.velocity
        for detection_idx, detection in enumerate(detections):
            center = pose_center(detection)
            if center is None:
                continue
            center_cost = np.linalg.norm(track.center + track.velocity - center)
            cost[track_idx, detection_idx] = (
                0.6 * center_cost + 0.4 * pose_distance(predicted_pose, detection)
            )

    # linear_sum_assignment 不接受整行/整列均为 inf 的矩阵。
    finite_rows = np.any(np.isfinite(cost), axis=1)
    finite_cols = np.any(np.isfinite(cost), axis=0)
    if not np.any(finite_rows) or not np.any(finite_cols):
        return []
    row_map = np.flatnonzero(finite_rows)
    col_map = np.flatnonzero(finite_cols)
    finite_cost = cost[np.ix_(finite_rows, finite_cols)]
    large_cost = max_match_distance + 1e6
    rows, cols = linear_sum_assignment(
        np.where(np.isfinite(finite_cost), finite_cost, large_cost)
    )
    assignments = []
    for row, col in zip(row_map[rows], col_map[cols]):
        if cost[row, col] <= max_match_distance:
            assignments.append((int(row), int(col)))
    return assignments


def create_track(track_id, pose):
    center = pose_center(pose)
    return TrackState(
        track_id=track_id,
        pose=pose.copy(),
        center=center.copy(),
        velocity=np.zeros(3, dtype=np.float64),
        missing_count=0,
        age=1,
    )


def get_next_track_id(active_tracks, max_tracks):
    used_ids = {track.track_id for track in active_tracks}
    for track_id in range(max_tracks):
        if track_id not in used_ids:
            return track_id
    return None


def build_tracked_tensor(frames, max_tracks):
    """将每帧无序 query 关联为稳定轨迹，并记录轨迹对应的 query。"""
    tracked = np.full(
        (len(frames), max_tracks, 17, 3), np.nan, dtype=np.float64
    )
    track_query_indices = np.full(
        (len(frames), max_tracks), -1, dtype=np.int64
    )
    active_tracks = []

    for frame_idx, frame in enumerate(frames):
        query_indices = [
            idx
            for idx, pose in enumerate(frame)
            if int(valid_joint_mask(pose).sum()) >= MIN_VALID_JOINTS
        ]
        detections = [frame[idx] for idx in query_indices]
        assignments = assign_detections(
            active_tracks, detections, MAX_MATCH_DISTANCE
        )
        assigned_rows = {row for row, _ in assignments}
        assigned_cols = {col for _, col in assignments}
        detection_for_row = {row: col for row, col in assignments}
        current_query_for_track = {}
        new_active = []

        for row, track in enumerate(active_tracks):
            if row not in assigned_rows:
                track.missing_count += 1
                if track.missing_count <= MAX_MISSING:
                    new_active.append(track)
                continue

            detection_idx = detection_for_row[row]
            detection = detections[detection_idx]
            previous_center = track.center.copy()
            center = pose_center(detection)
            track.velocity = center - previous_center
            track.center = center
            track.pose = detection.copy()
            track.missing_count = 0
            track.age += 1
            current_query_for_track[track.track_id] = query_indices[detection_idx]
            new_active.append(track)

        active_tracks = new_active
        for detection_idx, detection in enumerate(detections):
            if detection_idx in assigned_cols:
                continue
            track_id = get_next_track_id(active_tracks, max_tracks)
            if track_id is None:
                break
            active_tracks.append(create_track(track_id, detection))
            current_query_for_track[track_id] = query_indices[detection_idx]

        for track in active_tracks:
            if track.missing_count != 0:
                continue
            tracked[frame_idx, track.track_id] = track.pose
            track_query_indices[frame_idx, track.track_id] = (
                current_query_for_track[track.track_id]
            )

    return tracked, track_query_indices


def interpolate_1d(values, max_gap):
    out = np.asarray(values, dtype=np.float64).copy()
    missing = np.isnan(out)
    if np.all(missing):
        return out

    indices = np.arange(len(out))
    valid_indices = indices[~missing]
    out[missing] = np.interp(indices[missing], valid_indices, out[~missing])

    if max_gap >= 0:
        padded = np.pad(missing.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            if end - start > max_gap:
                out[start:end] = np.nan
    return out


def median_filter_1d(values, kernel_size):
    if kernel_size <= 1:
        return values.copy()
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size + 4
    padded = np.pad(values, (pad, pad), mode="reflect")
    return signal.medfilt(padded, kernel_size=kernel_size)[pad:-pad]


def exponential_smooth_1d(values, alpha):
    out = values.copy()
    for idx in range(1, len(out)):
        out[idx] = (1.0 - alpha) * out[idx - 1] + alpha * out[idx]
    return out


def suppress_velocity_outliers(sequence, threshold):
    if threshold <= 0:
        return sequence.copy()
    out = sequence.copy()
    speed = np.linalg.norm(np.diff(out, axis=0), axis=2)
    for time_idx, joint_idx in np.argwhere(speed > threshold):
        out[time_idx + 1, joint_idx] = np.nan
    return out


def measure_bone_lengths(pose):
    lengths = np.full(len(DIRECTED_BONES), np.nan, dtype=np.float64)
    for idx, (parent, child) in enumerate(DIRECTED_BONES):
        if np.all(np.isfinite(pose[[parent, child]])):
            lengths[idx] = np.linalg.norm(pose[child] - pose[parent])
    return lengths


def compute_bone_template(sequence):
    lengths = np.stack(
        [measure_bone_lengths(pose) for pose in sequence], axis=0
    )
    template = np.full(lengths.shape[1], np.nan, dtype=np.float64)
    for bone_idx in range(lengths.shape[1]):
        valid = np.isfinite(lengths[:, bone_idx])
        if np.any(valid):
            template[bone_idx] = np.median(lengths[valid, bone_idx])
    return template


def enforce_bone_lengths(pose, template):
    adjusted = pose.copy()
    if BONE_LENGTH_WEIGHT <= 0:
        return adjusted
    for _ in range(max(1, BONE_ITERS)):
        for idx, (parent, child) in enumerate(DIRECTED_BONES):
            target = template[idx]
            if not np.isfinite(target):
                continue
            if not np.all(np.isfinite(adjusted[[parent, child]])):
                continue
            vector = adjusted[child] - adjusted[parent]
            length = np.linalg.norm(vector)
            if length <= 1e-8:
                continue
            desired = adjusted[parent] + vector / length * target
            adjusted[child] = (
                (1.0 - BONE_LENGTH_WEIGHT) * adjusted[child]
                + BONE_LENGTH_WEIGHT * desired
            )
    return adjusted


def process_single_track(track_sequence):
    sequence = suppress_velocity_outliers(
        track_sequence, VELOCITY_THRESHOLD
    )
    valid_mask = np.all(np.isfinite(sequence), axis=2)
    for joint_idx in range(sequence.shape[1]):
        if valid_mask[:, joint_idx].sum() < 2:
            continue
        for dim in range(3):
            values = interpolate_1d(
                sequence[:, joint_idx, dim], MAX_INTERP_GAP
            )
            finite = np.isfinite(values)
            if finite.sum() < 2:
                sequence[:, joint_idx, dim] = values
                continue
            filled = values.copy()
            if np.any(~finite):
                indices = np.arange(len(filled))
                filled[~finite] = np.interp(
                    indices[~finite], indices[finite], filled[finite]
                )
            filtered = median_filter_1d(filled, MEDFILT_KERNEL)
            filtered = exponential_smooth_1d(filtered, SMOOTH_ALPHA)
            filtered[~finite] = np.nan
            sequence[:, joint_idx, dim] = filtered

    template = compute_bone_template(sequence)
    for frame_idx in range(sequence.shape[0]):
        sequence[frame_idx] = enforce_bone_lengths(
            sequence[frame_idx], template
        )
    return sequence


def process_tracked_tensor(tracked):
    output = tracked.copy()
    for track_id in range(tracked.shape[1]):
        output[:, track_id] = process_single_track(tracked[:, track_id])
    return output


def get_unique_frame_locations(radar_paths):
    """返回路径到唯一帧编号的映射，以及每帧的首次滑窗位置。"""
    path_to_frame = {}
    first_locations = []
    for window_idx, window_paths in enumerate(radar_paths):
        for time_idx, path in enumerate(window_paths):
            key = str(path)
            if key not in path_to_frame:
                path_to_frame[key] = len(first_locations)
                first_locations.append((window_idx, time_idx))
    return path_to_frame, first_locations


def match_poses(reference_poses, current_poses, max_distance):
    reference_indices = [
        idx
        for idx, pose in enumerate(reference_poses)
        if pose_center(pose) is not None
    ]
    current_indices = [
        idx
        for idx, pose in enumerate(current_poses)
        if pose_center(pose) is not None
    ]
    if not reference_indices or not current_indices:
        return []
    cost = np.asarray(
        [
            [
                pose_distance(reference_poses[i], current_poses[j])
                for j in current_indices
            ]
            for i in reference_indices
        ]
    )
    rows, cols = linear_sum_assignment(cost)
    return [
        (reference_indices[row], current_indices[col])
        for row, col in zip(rows, cols)
        if cost[row, col] <= max_distance
    ]


def postprocess_pose_sequence(pose, detection_mask, radar_paths):
    """在唯一物理帧上平滑姿态，再映射回所有重叠滑窗的 query。"""
    pose_np = pose.numpy().astype(np.float64, copy=True)
    mask_np = detection_mask.numpy().astype(bool, copy=False)
    path_to_frame, first_locations = get_unique_frame_locations(radar_paths)

    unique_frames = []
    for window_idx, time_idx in first_locations:
        frame = pose_np[window_idx, time_idx].copy()
        frame[~mask_np[window_idx, time_idx]] = np.nan
        unique_frames.append(frame)

    tracked, _ = build_tracked_tensor(unique_frames, max_tracks=pose.shape[2])
    processed = process_tracked_tensor(tracked)
    output = pose_np.copy()

    for window_idx, window_paths in enumerate(radar_paths):
        for time_idx, path in enumerate(window_paths):
            frame_idx = path_to_frame[str(path)]
            current = pose_np[window_idx, time_idx].copy()
            current[~mask_np[window_idx, time_idx]] = np.nan
            for track_idx, query_idx in match_poses(
                tracked[frame_idx], current, MAX_MATCH_DISTANCE
            ):
                smoothed_pose = processed[frame_idx, track_idx]
                finite = np.isfinite(smoothed_pose)
                output[window_idx, time_idx, query_idx][finite] = (
                    smoothed_pose[finite]
                )

    return torch.from_numpy(output).to(dtype=pose.dtype)


def load_pose_model(device):
    pose_model = build_model("P4Transformer").to(device)

    load_model_checkpoint(
        model_pose_path / "checkpoint" / "best.pth",
        pose_model,
        device,
    )

    pose_model.eval()
    return pose_model


class SelectedGroupDataset(HPE_Dataset):
    """只读取一个指定组，不经过训练集/验证集划分。"""

    def __init__(self, root_path: Path, date: str, group: str, T: int):
        # 不调用 HPE_Dataset.__init__，避免读取和划分整个数据集。
        self.root_path = root_path
        self.date = date
        self.T = T
        self.base_source = "radar_high_pc"
        self.sensor_config = {
            "radar_high_bin": False,
            "radar_high_pc": True,
            "gt": True,
        }
        self.suffix_map = {
            "radar_high_bin": ".bin",
            "radar_high_pc": ".npy",
            "gt": ".pkl",
        }
        self.cached_sensor_names = {"radar_high_pc", "gt"}
        self.calib_cache = {}
        self.pointcloud_cache = {}
        self.gt_cache = {}
        self.action_cache = {}

        group_path = root_path / date / "data_collection" / group
        sensor_paths = {
            "radar_high_pc": group_path / "dpct高位机" / "PC",
            "gt": group_path / "camera results" / "smoothed 3D",
        }
        aligned = self._align_multi_sensor_files(
            sources=sensor_paths,
            base_source=self.base_source,
        )

        frame_count = len(aligned[self.base_source])
        if frame_count < T:
            raise ValueError(f"对齐后只有 {frame_count} 帧，小于 T={T}")

        # 滑窗后的每个 item 是长度为 T 的序列。
        self.data_path_list = {
            sensor: [
                paths[start : start + T]
                for start in range(frame_count - T + 1)
            ]
            for sensor, paths in aligned.items()
        }

    def __getitem__(self, idx):
        radar = self._get_sensor_data_from_path(
            "radar_high_pc",
            self.data_path_list["radar_high_pc"][idx],
        )
        gt = self._get_sensor_data_from_path(
            "gt",
            self.data_path_list["gt"][idx],
        )
        calib = self._load_calib_T(self.date)
        gt_for_high = self._transform_gt_sequence(
            gt, calib["gt_to_high"]["R"], calib["gt_to_high"]["t"]
        )
        gt_for_low = self._transform_gt_sequence(
            gt, calib["gt_to_low"]["R"], calib["gt_to_low"]["t"]
        )

        # 此组没有 action label；占位值不参与检测和姿态评估。
        action = [
            np.zeros((len(frame_gt), 4), dtype=np.float32)
            for frame_gt in gt_for_high
        ]
        return {
            "radar_high_pc": radar,
            "gt": gt,
            "action": action,
            "gt_for_high": gt_for_high,
            "gt_for_low": gt_for_low,
            "high_to_low_R": [calib["high_to_low"]["R"].copy() for _ in range(self.T)],
            "high_to_low_t": [calib["high_to_low"]["t"].copy() for _ in range(self.T)],
        }


def build_dataloader():
    dataset = SelectedGroupDataset(ROOT_PATH, DATE, GROUP, T)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_points=300,
            max_people=6,
        ),
    )


def build_pose_input(points, mask, bbox, detection_mask):
    """为每个有效 bbox 构造一条姿态模型输入序列。"""
    B, T, N, C = points.shape
    K = bbox.shape[2]
    center = (bbox[..., :3] + bbox[..., 3:]) / 2
    points = points[:, :, None].expand(-1, -1, K, -1, -1)
    mask = mask[:, :, None].expand(-1, -1, K, -1)
    xyz = points[..., :3]
    inside = (
        (xyz >= bbox[..., None, :3])
        & (xyz <= bbox[..., None, 3:])
    ).all(dim=-1)
    pose_mask = mask & inside & detection_mask[..., None]

    pose_points = points.clone()
    pose_points[..., :3] -= center[..., None, :]
    pose_points *= pose_mask[..., None]
    pose_points = pose_points.permute(0, 2, 1, 3, 4).reshape(B * K, T, N, C)
    pose_mask = pose_mask.permute(0, 2, 1, 3).reshape(B * K, T, N)
    return {"input": pose_points, "mask": pose_mask}, center


def build_oracle_detection(gt_bbox, gt_valid):
    """用 GT bbox/mask 构造与检测模型输出字段兼容的 oracle 结果。"""
    objectness_logits = torch.where(
        gt_valid,
        torch.full_like(gt_valid, 20, dtype=gt_bbox.dtype),
        torch.full_like(gt_valid, -20, dtype=gt_bbox.dtype),
    )
    return {
        "bbox": gt_bbox,
        "objectness_logits": objectness_logits,
        "mask": gt_valid,
    }


def place_poses_at_gt_root(pose_local, gt_pose):
    """将每个同槽位预测变为 root-relative 后放置到 GT 髋部 root。"""
    local_root = (pose_local[..., 11, :] + pose_local[..., 12, :]) / 2
    gt_root = (gt_pose[..., 11, :] + gt_pose[..., 12, :]) / 2
    return pose_local - local_root[..., None, :] + gt_root[..., None, :]


def transform_xyz(xyz, R, t):
    return xyz @ R.transpose(-1, -2) + t


def transform_bbox(bbox, R, t):
    """变换 bbox 的 8 个角点，再生成目标坐标系下的轴对齐框。"""
    bbox_min, bbox_max = bbox[..., :3], bbox[..., 3:]
    corners = torch.stack(
        [
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_max[..., 2]), -1),
        ],
        dim=-2,
    )
    corners = transform_xyz(corners, R, t)
    return torch.cat((corners.amin(dim=-2), corners.amax(dim=-2)), dim=-1)


def get_partial_hungarian_matches(
    pred_bbox,
    gt_bbox,
    pred_mask,
    gt_mask,
    bbox_l1_weight=5.0,
    bbox_iou_weight=2.0,
):
    """只匹配有效 query；允许 query 数少于 GT 人数。"""
    pc_min = pred_bbox.new_tensor(POINT_CLOUD_RANGE[:3])
    extent = (
        pred_bbox.new_tensor(POINT_CLOUD_RANGE[3:]) - pc_min
    ).clamp_min(1e-6)
    matches = []

    for batch_idx in range(pred_bbox.shape[0]):
        for time_idx in range(pred_bbox.shape[1]):
            pred_idx = pred_mask[batch_idx, time_idx].nonzero().flatten()
            gt_idx = gt_mask[batch_idx, time_idx].nonzero().flatten()
            if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=pred_bbox.device,
                )
                matches.append((empty, empty))
                continue

            pred = pred_bbox[batch_idx, time_idx, pred_idx]
            gt = gt_bbox[batch_idx, time_idx, gt_idx]
            pred_norm = torch.cat(
                ((pred[:, :3] - pc_min) / extent, (pred[:, 3:] - pc_min) / extent),
                dim=-1,
            )
            gt_norm = torch.cat(
                ((gt[:, :3] - pc_min) / extent, (gt[:, 3:] - pc_min) / extent),
                dim=-1,
            )
            l1_cost = (
                pred_norm[:, None] - gt_norm[None]
            ).abs().sum(dim=-1)
            iou_cost = 1 - pairwise_axis_aligned_iou_3d(pred, gt)
            cost = bbox_l1_weight * l1_cost + bbox_iou_weight * iou_cost
            pred_local, gt_local = linear_sum_assignment(cost.cpu().numpy())
            matches.append(
                (
                    pred_idx[
                        torch.as_tensor(
                            pred_local,
                            device=pred_idx.device,
                        )
                    ],
                    gt_idx[
                        torch.as_tensor(
                            gt_local,
                            device=gt_idx.device,
                        )
                    ],
                )
            )

    return matches


def reanchor_matched_poses_to_gt_root(pose, gt_pose, matches):
    """平滑或骨长约束后，将匹配姿态的 root 精确放回 GT root。"""
    anchored = pose.clone()
    pose_root = (pose[..., 11, :] + pose[..., 12, :]) / 2
    gt_root = (gt_pose[..., 11, :] + gt_pose[..., 12, :]) / 2
    time_count = pose.shape[1]

    for flat_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() == 0:
            continue
        batch_idx, time_idx = divmod(flat_idx, time_count)
        translation = (
            gt_root[batch_idx, time_idx, gt_idx]
            - pose_root[batch_idx, time_idx, pred_idx]
        )
        anchored[batch_idx, time_idx, pred_idx] += translation[:, None, :]
    return anchored


def analyze_results():
    loaded = torch.load(RESULT_PATH, map_location="cpu")
    if loaded.get("detection_source") != "gt_bbox":
        raise RuntimeError(
            "当前结果文件不是 GT bbox 推理结果；请先将 STAGE 设为 "
            "'inference' 重新生成 RESULT_PATH，再执行 analysis。"
        )

    # 滑窗步长为 1，同一物理帧会重复出现。按时间顺序只保留首次出现。
    unique_indices = []
    unique_paths = []
    seen = set()
    for window_idx, window_paths in enumerate(loaded["radar_paths"]):
        for time_idx, path in enumerate(window_paths):
            path = str(path)
            if path not in seen:
                seen.add(path)
                unique_indices.append(window_idx * T + time_idx)
                unique_paths.append(path)

    indices = torch.tensor(unique_indices, dtype=torch.long)

    def unique_tensor(value):
        return value.flatten(0, 1).index_select(0, indices).unsqueeze(0)

    # analysis 从平滑前姿态重新执行后处理，保证动作识别和视频使用的
    # 一定是平滑及骨长约束后的姿态。
    pose_pre_raw = loaded.get("pose_pre_raw", loaded["pose_pre"])
    pose_pre_processed = postprocess_pose_sequence(
        pose_pre_raw,
        loaded["detection_mask"].bool(),
        loaded["radar_paths"],
    )
    oracle_matches = get_partial_hungarian_matches(
        loaded["bbox_pre"],
        loaded["bbox_gt"],
        loaded["detection_mask"].bool(),
        loaded["gt_valid"].bool(),
    )
    pose_pre_processed = reanchor_matched_poses_to_gt_root(
        pose_pre_processed,
        loaded["pose_gt"],
        oracle_matches,
    )

    bbox_pre = unique_tensor(loaded["bbox_pre"])
    logits = unique_tensor(loaded["objectness_logits"])
    detection_mask = unique_tensor(loaded["detection_mask"]).bool()
    pose_pre = unique_tensor(pose_pre_processed)
    pose_gt = unique_tensor(loaded["pose_gt"])
    bbox_gt = unique_tensor(loaded["bbox_gt"])
    gt_mask = unique_tensor(loaded["gt_valid"]).bool()
    pc = unique_tensor(loaded["pc"])
    pc_valid = unique_tensor(loaded["pc_valid"])
    high_to_low_R = unique_tensor(loaded["high_to_low_R"])
    high_to_low_t = unique_tensor(loaded["high_to_low_t"])
    pose_gt_gravity = torch.empty_like(pose_gt)
    pose_pre_gravity = torch.empty_like(pose_pre)
    for frame_idx in range(pose_gt.shape[1]):
        R = high_to_low_R[0, frame_idx]
        t = high_to_low_t[0, frame_idx]
        pose_gt_gravity[:, frame_idx] = transform_xyz(
            pose_gt[:, frame_idx], R, t
        )
        pose_pre_gravity[:, frame_idx] = transform_xyz(
            pose_pre[:, frame_idx], R, t
        )

    pose_gt_for_action = pose_gt_gravity.numpy().copy()
    pose_pre_for_action = pose_pre_gravity.numpy().copy()
    pose_gt_for_action[~gt_mask.numpy()] = np.nan
    pose_pre_for_action[~detection_mask.numpy()] = np.nan
    action_gt_result = classify_actions(pose_gt_for_action)
    action_pre_result = classify_actions(pose_pre_for_action)
    action_gt = torch.from_numpy(action_gt_result.labels)
    action_pre = torch.from_numpy(action_pre_result.labels)

    matches = get_partial_hungarian_matches(
        bbox_pre,
        bbox_gt,
        detection_mask,
        gt_mask,
    )

    objectness = get_objectness(logits, gt_mask, matches).mean()
    bbox_l1 = get_bbox_l1(
        bbox_pre, bbox_gt, matches, POINT_CLOUD_RANGE
    )
    bbox_iou_loss = get_bbox_iou(bbox_pre, bbox_gt, matches)
    gt_num = int(gt_mask.sum())
    matched_gt_mask = torch.zeros_like(gt_mask)
    for frame_idx, (_, gt_idx) in enumerate(matches):
        matched_gt_mask[0, frame_idx, gt_idx] = True
    matched_num = int(matched_gt_mask.sum())

    tp = 0
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() == 0:
            continue
        iou = paired_axis_aligned_iou_3d(
            bbox_pre[0, frame_idx, pred_idx],
            bbox_gt[0, frame_idx, gt_idx],
        )
        tp += int((iou >= IOU_THRESHOLD).sum())

    detected_num = int(detection_mask.sum())
    fn = gt_num - tp
    fp = detected_num - matched_num
    tn = (
        detection_mask.numel()
        - detected_num
        - (gt_num - matched_num)
    )
    tp = torch.tensor(tp)
    fp = torch.tensor(fp)
    fn = torch.tensor(fn)
    tn = torch.tensor(tn)
    precision = get_precision(tp, fp, fn, tn)
    recall = get_recall(tp, fp, fn, tn)
    accuracy = get_acc(tp, fp, fn, tn)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"\n去重前帧数: {loaded['pose_gt'].shape[0] * loaded['pose_gt'].shape[1]}")
    print(f"去重后帧数: {len(unique_paths)}")
    print(f"GT 人数: {gt_num}")
    print(f"有效 query 数: {int(detection_mask.sum())}")
    print(f"匹配人数: {matched_num}")
    print(f"未匹配 GT 数: {gt_num - matched_num}")
    print(f"未匹配 query 数: {int(detection_mask.sum()) - matched_num}")
    print("\nBBox metrics")
    print(f"参与框误差计算人数: {matched_num}/{gt_num}")
    print(f"objectness: {objectness.item():.6f}")
    if matched_num:
        print(f"bbox_l1: {bbox_l1[matched_gt_mask].mean().item():.6f}")
        print(
            "bbox_iou: "
            f"{(1 - bbox_iou_loss[matched_gt_mask]).mean().item():.6f}"
        )
    print(f"TP: {int(tp)}, TN: {int(tn)}, FP: {int(fp)}, FN: {int(fn)}")
    print(f"precision: {precision.item():.6f}")
    print(f"recall: {recall.item():.6f}")
    print(f"accuracy: {accuracy.item():.6f}")
    print(f"f1: {f1.item():.6f}")

    # 姿态误差在 confidence 筛选后的匈牙利匹配人体上计算。
    pred_people = []
    gt_people = []
    flat_pose_pre = pose_pre.flatten(0, 1)
    flat_pose_gt = pose_gt.flatten(0, 1)
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() == 0:
            continue
        pred_people.append(flat_pose_pre[frame_idx, pred_idx])
        gt_people.append(flat_pose_gt[frame_idx, gt_idx])

    valid_pose_num = sum(value.shape[0] for value in pred_people)
    print("\nPose metrics")
    print(f"参与姿态评估人数: {valid_pose_num}/{gt_num}")
    if valid_pose_num:
        pred_people = torch.cat(pred_people)
        gt_people = torch.cat(gt_people)
        print(f"mpjpe: {get_mpjpe(pred_people, gt_people).mean().item():.6f}")
        pred_root = (pred_people[:, 11] + pred_people[:, 12]) / 2
        gt_root = (gt_people[:, 11] + gt_people[:, 12]) / 2
        root_relative_mpjpe = get_mpjpe(
            pred_people - pred_root[:, None],
            gt_people - gt_root[:, None],
        ).mean()
        print(f"root_relative_mpjpe: {root_relative_mpjpe.item():.6f}")
        print(f"pampjpe: {get_pampjpe(pred_people, gt_people).mean().item():.6f}")
        print(
            "bone_length: "
            f"{get_bone_length(pred_people, gt_people, type='coco').mean().item():.6f}"
        )

    matched_action_pre = []
    matched_action_gt = []
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        matched_action_pre.append(action_pre[0, frame_idx, pred_idx])
        matched_action_gt.append(action_gt[0, frame_idx, gt_idx])

    print("\nAction metrics")
    if matched_num:
        matched_action_pre = torch.cat(matched_action_pre).argmax(dim=-1)
        matched_action_gt = torch.cat(matched_action_gt).argmax(dim=-1)
        action_accuracy = (
            matched_action_pre == matched_action_gt
        ).float().mean()
        confusion = torch.zeros(4, 4, dtype=torch.long)
        for gt_label, pred_label in zip(
            matched_action_gt,
            matched_action_pre,
        ):
            confusion[gt_label, pred_label] += 1
        print(f"accuracy: {action_accuracy.item():.6f}")
        print(f"labels: {LABEL_NAMES}")
        print("confusion_matrix (row=GT, col=Pred):")
        print(confusion)

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    video_writer = None
    video_size = None
    with TemporaryDirectory(dir="/tmp") as temp_dir:
        try:
            for frame_idx in tqdm(
                range(len(unique_paths)),
                desc="Rendering video",
            ):
                pred_idx, gt_idx = matches[frame_idx]
                aligned_pose = torch.zeros_like(
                    pose_gt_gravity[:, frame_idx : frame_idx + 1]
                )
                aligned_pose[0, 0, gt_idx] = pose_pre_gravity[
                    0, frame_idx, pred_idx
                ]
                aligned_pose_mask = torch.zeros_like(
                    gt_mask[:, frame_idx : frame_idx + 1]
                )
                aligned_pose_mask[0, 0, gt_idx] = True
                aligned_action = torch.zeros_like(
                    action_gt[:, frame_idx : frame_idx + 1]
                )
                aligned_action[..., 3] = 1
                aligned_action[0, 0, gt_idx] = action_pre[
                    0, frame_idx, pred_idx
                ]

                R = high_to_low_R[0, frame_idx]
                t = high_to_low_t[0, frame_idx]
                plot_pc = pc[:, frame_idx : frame_idx + 1].clone()
                plot_pc[..., :3] = transform_xyz(plot_pc[..., :3], R, t)
                plot_pose_pre = aligned_pose
                plot_pose_gt = pose_gt_gravity[
                    :, frame_idx : frame_idx + 1
                ]
                plot_bbox_pre = transform_bbox(
                    bbox_pre[:, frame_idx : frame_idx + 1], R, t
                )
                plot_bbox_gt = transform_bbox(
                    bbox_gt[:, frame_idx : frame_idx + 1], R, t
                )

                frame_path = Path(temp_dir) / f"{frame_idx:06d}.png"
                plt_fig(
                    frame_path,
                    pre={
                        "pose": plot_pose_pre,
                        "mask": aligned_pose_mask,
                        "bbox": plot_bbox_pre,
                        "objectness_logits": logits[
                            :, frame_idx : frame_idx + 1
                        ],
                        "action_logits": aligned_action,
                    },
                    gt={
                        "padded": plot_pose_gt,
                        "bbox": plot_bbox_gt,
                        "mask": gt_mask[:, frame_idx : frame_idx + 1],
                        "action": action_gt[:, frame_idx : frame_idx + 1],
                        "action_label": LABEL_NAMES,
                    },
                    model_input={
                        "input": plot_pc,
                        "mask": pc_valid[:, frame_idx : frame_idx + 1],
                    },
                    matches=[(pred_idx, gt_idx)],
                    horizontal=True,
                    dpi=100,
                )

                image = cv2.imread(str(frame_path))
                if video_writer is None:
                    video_size = (image.shape[1], image.shape[0])
                    video_writer = cv2.VideoWriter(
                        str(VIDEO_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        VIDEO_FPS,
                        video_size,
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError("无法创建 MP4 视频")
                elif (image.shape[1], image.shape[0]) != video_size:
                    image = cv2.resize(image, video_size)
                video_writer.write(image)
        finally:
            if video_writer is not None:
                video_writer.release()

    print(f"\n可视化视频已保存到 {VIDEO_PATH}")


def run_inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pose_model = load_pose_model(device)
    dataloader = build_dataloader()
    print(f"模型已加载到 {device}")
    print(f"共 {len(dataloader.dataset)} 个滑窗，{len(dataloader)} 个 batch")
    results = {
        key: []
        for key in (
            "pc",
            "pc_valid",
            "bbox_pre",
            "objectness_logits",
            "detection_mask",
            "pose_pre",
            "pose_pre_local",
            "pose_input",
            "pose_input_mask",
            "pose_gt",
            "bbox_gt",
            "gt_valid",
            "high_to_low_R",
            "high_to_low_t",
        )
    }

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            points = batch["radar_high_pc"]["padded"].to(device)
            mask = batch["radar_high_pc"]["mask"].to(device)
            gt_pose = batch["gt_for_high"]["padded"].to(device)
            gt_bbox = batch["gt_for_high"]["bbox"].to(device)
            gt_valid = batch["gt_for_high"]["mask"].to(device)

            # Oracle detection：GT bbox 直接作为检测结果，不运行目标检测模型。
            detection = build_oracle_detection(gt_bbox, gt_valid)
            detection_mask = detection["mask"]
            pose_input, _ = build_pose_input(
                points,
                mask,
                detection["bbox"],
                detection_mask,
            )
            pose_output = pose_model(pose_input)
            B, T, K = detection_mask.shape
            pose_local = pose_output["pose"][:, :, 0].reshape(
                B, K, T, 17, 3
            ).permute(0, 2, 1, 3, 4)
            pose = place_poses_at_gt_root(pose_local, gt_pose)
            pose *= detection_mask[..., None, None]
            pose_points = pose_input["input"].reshape(
                B, K, T, points.shape[2], points.shape[3]
            ).permute(0, 2, 1, 3, 4)
            pose_mask = pose_input["mask"].reshape(
                B, K, T, points.shape[2]
            ).permute(0, 2, 1, 3)

            batch_results = {
                "pc": points,
                "pc_valid": mask,
                "bbox_pre": detection["bbox"],
                "objectness_logits": detection["objectness_logits"],
                "detection_mask": detection_mask,
                "pose_pre": pose,
                "pose_pre_local": pose_local,
                "pose_input": pose_points,
                "pose_input_mask": pose_mask,
                "pose_gt": gt_pose,
                "bbox_gt": gt_bbox,
                "gt_valid": gt_valid,
                "high_to_low_R": batch["high_to_low_R"],
                "high_to_low_t": batch["high_to_low_t"],
            }
            for key, value in batch_results.items():
                results[key].append(value.detach().cpu())

    results = {
        key: torch.cat(value, dim=0)
        for key, value in results.items()
    }
    results["input_key"] = "radar_high_pc"
    results["target_key"] = "gt_for_high"
    results["detection_source"] = "gt_bbox"
    results["radar_paths"] = dataloader.dataset.data_path_list["radar_high_pc"]
    results["pose_pre_raw"] = results["pose_pre"].clone()
    results["pose_pre"] = postprocess_pose_sequence(
        results["pose_pre_raw"],
        results["detection_mask"],
        results["radar_paths"],
    )
    root = (
        results["pose_pre"][..., 11, :]
        + results["pose_pre"][..., 12, :]
    ) / 2
    gt_root = (
        results["pose_gt"][..., 11, :]
        + results["pose_gt"][..., 12, :]
    ) / 2
    results["pose_pre"] += (
        gt_root - root
    )[..., None, :] * results["detection_mask"][..., None, None]
    root = gt_root
    results["pose_pre_local"] = (
        results["pose_pre"] - root[..., None, :]
    ) * results["detection_mask"][..., None, None]
    print(
        "GT bbox、GT root、姿态时序平滑与骨长约束已完成"
        "（平滑前结果保存在 pose_pre_raw）"
    )
    torch.save(results, RESULT_PATH)
    print(f"结果已保存到 {RESULT_PATH}")


if __name__ == "__main__":
    if STAGE == "inference":
        run_inference()
    elif STAGE == "analysis":
        analyze_results()
    else:
        raise ValueError(f"不支持的 STAGE: {STAGE}")
