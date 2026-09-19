"""GT 框裁剪点云 → P4Transformer → 历史姿态平滑 → WebAgg。

运行：python deploy/Inference_selected_group.py（VISUALIZE=False 时自动评估整组）
VISUALIZE=True 或 --visualize：空格/右箭头下一帧，左箭头缓存回看；组末输出整组指标。
仅保留 GT 框裁剪（当前帧与历史静点积累）处理结果。
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib.util
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.port"] = 8988
matplotlib.rcParams["webagg.open_in_browser"] = True
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

from run.utils.set_device import set_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_PATH = Path("/mnt/huawei")
DATE = os.environ.get("INFERENCE_DATE", "20260912")
GROUP = os.environ.get("INFERENCE_GROUP", "group_029")
VISUALIZE = True   # False 自动跑完整组并输出 MPJPE； True WebAgg 逐帧交互。
MODEL_POSE_PATH = PROJECT_ROOT / "experiments/P4Transformer/20260909_191752"
T = 8
POSE_OUTPUT_POSITION = 7  # 第 4 个位置：3 帧历史 + 当前帧 + 4 帧未来。
POSE_LOOKAHEAD = T - POSE_OUTPUT_POSITION - 1
MAX_POINTS = 300
TRAIN_MAX_POINTS = 200
GT_BBOX_MARGIN = 0.30
XYZ_LIMITS = ((0.0, 6.0), (-3.0, 3.0), (-2.0, 2.0))
ERROR_VIS_THRESHOLD_MM = 200.0
ERROR_VIS_MAX_FRAMES = 10
ERROR_VIS_ROOT = PROJECT_ROOT / "deploy/error_visualizations"
METRIC_VIS_ROOT = PROJECT_ROOT / "deploy/metric_visualizations"
METRIC_REFERENCE_MM = 150.0

# 低位机重力坐标系下的指标验收区：前方 0.4 m 盲区 + 1.6 m 半轴。
ACCEPTANCE_CENTER_XY = np.array([2.0, 0.0], dtype=np.float32)
ACCEPTANCE_RADII_XY = np.array([1.6, 2.4], dtype=np.float32)
ACCEPTANCE_COLOR = "#FFF2CC"
ACTION_LABELS = ("stand", "sit_squat", "lie", "other")
ACTION_COLORS = ("#4daf4a", "#377eb8", "#e41a1c", "#984ea3")


# 每条 GT 轨迹的历史静点补充：1=动点，2..7=静点。
STATIC_ACCUMULATION = True
DYNAMIC_STATIC_RATIO_THRESHOLD = 0.20
STATIC_HISTORY_FRAMES = 10
DYNAMIC_POINT_TAG = 1.0

# 原有姿态后处理参数；位移阈值单位为米/帧，并非米/秒。
SMOOTH_ALPHA = 0.35
MAX_INTERP_GAP = 8
MEDFILT_KERNEL = 7
VELOCITY_THRESHOLD = 0.35
BONE_LENGTH_WEIGHT = 0.65
BONE_ITERS = 2
# 时间滤波改善关节相对形状，但绝对坐标 EMA 会在人体移动时产生位置滞后。
# 最终将骨架髋中心锚回当前帧原始预测；短缺失帧使用插值后的髋中心。
PRESERVE_RAW_ROOT_POSITION = True

# 只在“几乎没有动点 + 点云覆盖已坍缩”时保持上一个可靠姿态。
# 这些阈值只用于后处理，不改变网络输入；可通过环境变量做 A/B 实验。
# group_100 实验未改善 MPJPE，因此默认关闭，仅保留为可复现的实验开关。
QUALITY_HOLD = os.environ.get("POSE_QUALITY_HOLD", "0") != "0"
QUALITY_MAX_DYNAMIC_POINTS = int(os.environ.get("POSE_QUALITY_MAX_DYNAMIC", "1"))
QUALITY_MIN_POINTS = int(os.environ.get("POSE_QUALITY_MIN_POINTS", "8"))
QUALITY_MIN_SPAN_M = float(os.environ.get("POSE_QUALITY_MIN_SPAN_M", "0.35"))
QUALITY_MIN_VOXELS = int(os.environ.get("POSE_QUALITY_MIN_VOXELS", "4"))
QUALITY_VOXEL_SIZE_M = 0.20

# 有明显位移时抑制单帧 180° 骨架翻转；保留当前髋中心，只保持相对姿态。
DIRECTION_HOLD = os.environ.get("POSE_DIRECTION_HOLD", "0") != "0"
DIRECTION_MIN_ROOT_MOTION_M = float(os.environ.get("POSE_DIRECTION_MIN_MOTION_M", "0.04"))
DIRECTION_FLIP_DEGREES = float(os.environ.get("POSE_DIRECTION_FLIP_DEGREES", "120"))
DIRECTION_MAX_HOLD_FRAMES = int(os.environ.get("POSE_DIRECTION_MAX_HOLD", "2"))

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


@dataclass
class PoseTrack:
    track_id: int
    source: object = None
    history: deque = field(default_factory=lambda: deque(maxlen=T))
    static_history: deque = field(
        default_factory=lambda: deque(maxlen=STATIC_HISTORY_FRAMES)
    )


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
    raw_root = np.asarray(track_sequence, dtype=np.float64)[:, [11, 12]].mean(axis=1)
    raw_root = suppress_velocity_outliers(
        raw_root[:, None, :], VELOCITY_THRESHOLD
    )[:, 0]
    root_target = np.full_like(raw_root, np.nan)
    for dim in range(3):
        root_target[:, dim] = interpolate_1d(raw_root[:, dim], MAX_INTERP_GAP)
    sequence = suppress_velocity_outliers(
        track_sequence, VELOCITY_THRESHOLD
    )
    valid_mask = np.all(np.isfinite(sequence), axis=2)
    for joint_idx in range(sequence.shape[1]):
        if valid_mask[:, joint_idx].sum() < 1:
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
        if PRESERVE_RAW_ROOT_POSITION and np.all(np.isfinite(sequence[frame_idx])) \
                and np.all(np.isfinite(root_target[frame_idx])):
            filtered_root = sequence[frame_idx, [11, 12]].mean(axis=0)
            sequence[frame_idx] += root_target[frame_idx] - filtered_root
    return sequence


def crop_and_pad(points, bbox=None, point_mask=None, max_points=MAX_POINTS):
    """按给定 mask 或框取点；保留绝对坐标并填充到固定点数。"""
    inside = (np.asarray(point_mask, dtype=bool) if point_mask is not None else
              ((points[:, :3] >= bbox[:3]) & (points[:, :3] <= bbox[3:])).all(1))
    if inside.shape != (len(points),):
        raise ValueError("选点 mask 与原始点云长度不一致")
    selected = points[inside]
    if not len(selected):
        return None, inside
    if len(selected) > max_points:
        # 确定性均匀抽样，避免同一帧重复查看时随机变化。
        selected = selected[np.linspace(0, len(selected) - 1, max_points, dtype=int)]
    padded = np.zeros((max_points, points.shape[1]), dtype=np.float32)
    mask = np.zeros(max_points, dtype=bool)
    padded[:len(selected)] = selected
    mask[:len(selected)] = True
    return (padded, mask), inside


def mask_sampled_points(points, point_mask):
    """训练格式：[200, C] 槽位不重排，未选中点清零并关闭 mask。"""
    point_mask = np.asarray(point_mask, dtype=bool)
    if len(points) > TRAIN_MAX_POINTS or point_mask.shape != (len(points),):
        raise ValueError("预采样点云或 mask 尺寸错误")
    padded = np.zeros((TRAIN_MAX_POINTS, points.shape[1]), dtype=np.float32)
    mask = np.zeros(TRAIN_MAX_POINTS, dtype=bool)
    padded[:len(points)] = points
    mask[:len(points)] = point_mask
    padded[~mask] = 0.0
    return padded, mask


def accumulate_track_static_points(track, points, point_mask, target_center):
    """动静比过低时，用同一 GT 轨迹的历史静点补充当前关联点。"""
    current = points[np.asarray(point_mask, dtype=bool)]
    dynamic = (
        np.isclose(current[:, -1], DYNAMIC_POINT_TAG)
        if len(current)
        else np.empty(0, dtype=bool)
    )
    dynamic_count = int(dynamic.sum())
    static_count = len(current) - dynamic_count
    ratio = dynamic_count / max(static_count, 1)
    added = []
    if (
        STATIC_ACCUMULATION
        and ratio < DYNAMIC_STATIC_RATIO_THRESHOLD
    ):
        for historical_static, historical_center in track.static_history:
            aligned = historical_static.copy()
            aligned[:, :3] += target_center - historical_center
            added.append(aligned)

    # 当前静点在下一帧才成为“历史”，避免本帧被重复加入。
    current_static = current[~dynamic].copy()
    if len(current_static):
        track.static_history.append(
            (current_static, np.asarray(target_center).copy())
        )
    if not added:
        return current, ratio, 0
    historical = np.concatenate(added, axis=0)
    return np.concatenate((current, historical), axis=0), ratio, len(historical)


def point_cloud_quality(points):
    """提取与姿态可观测性直接相关的小型统计。"""
    if __package__:
        from .pose_smoothing import point_cloud_quality as _point_cloud_quality
    else:
        from pose_smoothing import point_cloud_quality as _point_cloud_quality
    return _point_cloud_quality(
        points, voxel_size=QUALITY_VOXEL_SIZE_M, min_points=QUALITY_MIN_POINTS,
        min_span=QUALITY_MIN_SPAN_M, min_voxels=QUALITY_MIN_VOXELS,
        max_dynamic=QUALITY_MAX_DYNAMIC_POINTS, dynamic_tag=DYNAMIC_POINT_TAG,
    )


def suppress_direction_flips(sequence):
    """运动中朝向突然翻转时，平移上一相对骨架到当前髋中心。"""
    if __package__:
        from .pose_smoothing import pose_facing_vector
    else:
        from pose_smoothing import pose_facing_vector
    out = np.asarray(sequence, dtype=np.float64).copy()
    previous = None
    held_run = held_count = 0
    flip_cosine = np.cos(np.deg2rad(DIRECTION_FLIP_DEGREES))
    for index, pose in enumerate(out):
        if not np.isfinite(pose).all():
            continue
        if previous is not None:
            root = pose[[11, 12]].mean(axis=0)
            previous_root = previous[[11, 12]].mean(axis=0)
            motion = np.linalg.norm(root - previous_root)
            facing = pose_facing_vector(pose)
            previous_facing = pose_facing_vector(previous)
            flipped = (facing is not None and previous_facing is not None
                       and np.dot(facing, previous_facing) < flip_cosine)
            if (motion >= DIRECTION_MIN_ROOT_MOTION_M and flipped
                    and held_run < DIRECTION_MAX_HOLD_FRAMES):
                out[index] = previous + (root - previous_root)
                held_run += 1
                held_count += 1
                previous = out[index]
                continue
        held_run = 0
        previous = out[index]
    return out, held_count


def build_pose_input(tracks):
    """每条轨迹最近 T 帧；新轨迹左端复制首帧，内部漏检帧用空 mask。"""
    sequences, masks = [], []
    for track in tracks:
        history = list(track.history)
        first = next(item for item in history if item is not None)
        history = [first] * (T - len(history)) + history
        sequence = np.stack([item[0] if item is not None else np.zeros_like(first[0])
                             for item in history])
        mask = np.stack([item[1] if item is not None else np.zeros_like(first[1])
                         for item in history])
        # 四条姿态支路的最终输入边界：任意特征非有限的点均视为 padding。
        finite_points = np.isfinite(sequence).all(axis=-1)
        mask &= finite_points
        sequence[~mask] = 0.0
        sequences.append(sequence)
        masks.append(mask)
    return {"input": torch.from_numpy(np.stack(sequences)),
            "mask": torch.from_numpy(np.stack(masks))}


def load_models(device):
    from run.utils.checkpoint import load_model_checkpoint
    from run.utils.load_config import load_config

    experiment_config = MODEL_POSE_PATH / "config"
    model_source = experiment_config / "P4Transformer.py"
    spec = importlib.util.spec_from_file_location("trained_p4transformer", model_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载训练时保存的模型源码：{model_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose_model = module.P4Transformer(
        **load_config(experiment_config / "model_config.yaml")
    ).to(device)
    load_model_checkpoint(MODEL_POSE_PATH / "checkpoint/best.pth", pose_model, device)
    return pose_model.eval()


def load_extrinsics(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_point_cloud(path):
    """读取有效点云；异常或空帧返回统一空数组及跳过原因。"""
    try:
        points = np.load(path).astype(np.float32)
    except Exception as error:
        return np.empty((0, 6), dtype=np.float32), f"无法读取：{error}"
    if points.ndim != 2 or points.shape[1] < 5:
        return np.empty((0, 6), dtype=np.float32), f"点云形状无效：{points.shape}"
    points = points[np.isfinite(points).all(axis=1)]
    return (points, None) if len(points) else (points, "没有有效点")


def closest_timestamp_file(path, folder):
    def timestamp(item):
        try:
            seconds, nanoseconds = item.stem.split("_")
            return int(seconds) * 10**9 + int(nanoseconds)
        except ValueError:
            return None
    target = timestamp(Path(path))
    candidates = [(abs(value - target), item) for item in Path(folder).iterdir()
                  if item.is_file() and (value := timestamp(item)) is not None]
    return min(candidates)[1] if candidates else None


def decimal_frame_ticks(frame_count, target_ticks=9):
    """生成个位数均为 0 的帧刻度。"""
    last = frame_count // 10 * 10
    if last == 0:
        return np.array([0])
    step = max(10, int(np.ceil(last / target_ticks / 10)) * 10)
    return np.unique(np.append(np.arange(0, last + 1, step), last))


def draw_human_pose(ax, poses, color, label):
    for person_index, pose in enumerate(poses):
        ax.scatter(*pose.T, c=color, s=25, label=label if person_index == 0 else None)
        for parent, child in DIRECTED_BONES:
            ax.plot(*pose[[parent, child]].T, c=color, linewidth=2)


def frame_mpjpe(poses, gt):
    """按髋中心距离匈牙利匹配，计算原始雷达坐标下 17 关节 MPJPE（mm）。

    仅评估所有关节均有限的完整人体；不做 root 对齐或 Procrustes 对齐。
    人数不等时匹配 min(P, G) 对，同时返回数量，避免漏检被误读为零误差。
    """
    pred = poses[np.isfinite(poses).all(axis=(1, 2))]
    target = gt[np.isfinite(gt).all(axis=(1, 2))]
    if not len(pred) or not len(target):
        return None, 0, len(pred), len(target)
    pred_root = pred[:, [11, 12]].mean(axis=1)
    gt_root = target[:, [11, 12]].mean(axis=1)
    cost = np.linalg.norm(pred_root[:, None] - gt_root[None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    error_mm = np.linalg.norm(pred[rows] - target[cols], axis=-1).mean() * 1000
    return float(error_mm), len(rows), len(pred), len(target)


def frame_pa_mpjpe(poses, gt):
    """髋中心匹配后逐人做刚性旋转、平移和尺度对齐，计算 PA-MPJPE（mm）。"""
    pred = poses[np.isfinite(poses).all(axis=(1, 2))].astype(np.float64)
    target = gt[np.isfinite(gt).all(axis=(1, 2))].astype(np.float64)
    if not len(pred) or not len(target):
        return None, 0, len(pred), len(target)
    pred_root = pred[:, [11, 12]].mean(axis=1)
    gt_root = target[:, [11, 12]].mean(axis=1)
    rows, cols = linear_sum_assignment(
        np.linalg.norm(pred_root[:, None] - gt_root[None], axis=-1)
    )
    errors = []
    for source, reference in zip(pred[rows], target[cols]):
        source_centered = source - source.mean(axis=0, keepdims=True)
        reference_centered = reference - reference.mean(axis=0, keepdims=True)
        source_norm = np.linalg.norm(source_centered)
        reference_norm = np.linalg.norm(reference_centered)
        if source_norm < 1e-12 or reference_norm < 1e-12:
            continue
        source_unit = source_centered / source_norm
        reference_unit = reference_centered / reference_norm
        u, singular_values, vt = np.linalg.svd(source_unit.T @ reference_unit)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ correction @ vt
        scale = reference_norm * np.sum(singular_values * np.diag(correction)) / source_norm
        aligned = scale * source_centered @ rotation + reference.mean(axis=0, keepdims=True)
        errors.append(np.linalg.norm(aligned - reference, axis=-1).mean() * 1000)
    return (float(np.mean(errors)) if errors else None), len(errors), len(pred), len(target)


def frame_centered_mpjpe(poses, gt):
    """髋中心匹配并分别减去各自髋中心，只忽略全局平移误差。"""
    pred = poses[np.isfinite(poses).all(axis=(1, 2))]
    target = gt[np.isfinite(gt).all(axis=(1, 2))]
    if not len(pred) or not len(target):
        return None, 0, len(pred), len(target)
    pred_root = pred[:, [11, 12]].mean(axis=1)
    gt_root = target[:, [11, 12]].mean(axis=1)
    rows, cols = linear_sum_assignment(
        np.linalg.norm(pred_root[:, None] - gt_root[None], axis=-1)
    )
    pred_centered = pred[rows] - pred_root[rows, None]
    gt_centered = target[cols] - gt_root[cols, None]
    error_mm = np.linalg.norm(pred_centered - gt_centered, axis=-1).mean() * 1000
    return float(error_mm), len(rows), len(pred), len(target)


def smooth_pose_records(records, prefix="", quality_hold=None, direction_hold=None):
    """从全部原始预测重新平滑；历史显示随新观测更新，绝不反复平滑已滤波结果。

    轨迹 ID 单调递增，不会复用。仅在首次和最后一次观测之间插补，
    不在人体出现前或消失后生成骨架；缺失超过 MAX_INTERP_GAP 保持缺失。
    """
    quality_hold = QUALITY_HOLD if quality_hold is None else quality_hold
    direction_hold = DIRECTION_HOLD if direction_hold is None else direction_hold
    histories = {}
    for frame_index, record in enumerate(records):
        for pose, detection_index in zip(record[f"{prefix}poses_raw"], record[f"{prefix}pose_indices"]):
            track_id = int(record[f"{prefix}track_ids"][detection_index])
            histories.setdefault(track_id, []).append((frame_index, pose))
        record[f"{prefix}poses"] = []
        record[f"{prefix}pose_track_ids"] = []
        # 旧帧骨架会被更新，MPJPE 留待展示时按新姿态重新计算。
        for metric_key in ("mpjpe_mm", "matched_people", "raw_mpjpe_mm", "raw_matched_people",
                           "pa_mpjpe_mm", "raw_pa_mpjpe_mm", "centered_mpjpe_mm",
                           "raw_centered_mpjpe_mm"):
            record.pop(f"{prefix}{metric_key}", None)
    # ponytail: 每次新帧重算全部历史，长序列总耗时为二次增长；需要时改为后台批处理。
    quality_held_count = direction_held_count = 0
    for track_id, observations in histories.items():
        start, end = observations[0][0], observations[-1][0]
        sequence = np.full((end - start + 1, 17, 3), np.nan, dtype=np.float64)
        previous_reliable = None
        for frame_index, pose in observations:
            quality = records[frame_index].get(f"{prefix}quality", {}).get(track_id, {})
            should_hold = quality_hold and quality.get("hold", False) and previous_reliable is not None
            sequence[frame_index - start] = previous_reliable if should_hold else pose
            if should_hold:
                quality_held_count += 1
            else:
                previous_reliable = pose
        sequence[~np.isfinite(sequence)] = np.nan
        if direction_hold:
            sequence, count = suppress_direction_flips(sequence)
            direction_held_count += count
        processed = process_single_track(sequence)
        for offset, pose in enumerate(processed):
            if not np.isfinite(pose).all():
                continue
            record = records[start + offset]
            record[f"{prefix}poses"].append(pose)
            record[f"{prefix}pose_track_ids"].append(track_id)
    for record in records:
        record[f"{prefix}poses"] = np.asarray(record[f"{prefix}poses"], dtype=np.float32).reshape(-1, 17, 3)
        record[f"{prefix}pose_track_ids"] = np.asarray(record[f"{prefix}pose_track_ids"], dtype=int)
    return {"quality": quality_held_count, "direction": direction_held_count}


def aggregate_mpjpe(frame_stats):
    matched = sum(stats[1] for stats in frame_stats)
    error_sum = sum(error * count for error, count, _, _ in frame_stats if error is not None)
    return {"mpjpe_mm": error_sum / matched if matched else None,
            "matched_people": matched,
            "pred_people": sum(stats[2] for stats in frame_stats),
            "gt_people": sum(stats[3] for stats in frame_stats),
            "matched_frames": sum(stats[1] > 0 for stats in frame_stats)}


def transform_points(points, rotation, translation):
    """按现有数据流程执行 p_low = R_high_to_low @ p_high + t。"""
    points = np.asarray(points)
    return points @ rotation.T + translation


def transform_cloud(points, rotation, translation):
    transformed = points.copy()
    transformed[..., :3] = transform_points(points[..., :3], rotation, translation)
    return transformed


def transform_boxes(boxes, rotation, translation):
    """旋转 8 个角点后重新生成低位机坐标系下的轴对齐框。"""
    boxes = np.asarray(boxes)
    if not len(boxes):
        return boxes.copy()
    lo, hi = boxes[:, :3], boxes[:, 3:]
    corners = np.stack([
        np.where(np.asarray(bits, dtype=bool), hi, lo)
        for bits in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    ], axis=1)
    corners = transform_points(corners, rotation, translation)
    return np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)


def acceptance_mask(poses, rotation, translation):
    """返回髋中心位于低位机 xOy 验收椭圆内的人体 mask。"""
    if not len(poses):
        return np.zeros(0, dtype=bool)
    roots_low = transform_points(poses[:, [11, 12]].mean(axis=1), rotation, translation)
    return np.square((roots_low[:, :2] - ACCEPTANCE_CENTER_XY) /
                     ACCEPTANCE_RADII_XY).sum(axis=1) <= 1.0


def acceptance_gt(gt, rotation, translation):
    """仅保留低位机 xOy 平面中髋中心落入验收椭圆的 GT。"""
    if not len(gt):
        return gt
    return gt[acceptance_mask(gt, rotation, translation)]


def append_group_metrics_markdown(metrics, path):
    """写入结果并重建表格，避免旧表头或空行破坏 Markdown 渲染。"""
    metric_names = (
        ("MPJPE", "mpjpe_mm"),
        ("centered-MPJPE", "centered_mpjpe_mm"),
        ("PA-MPJPE", "pa_mpjpe_mm"),
    )
    headers = ["date", "group", "MODEL_POSE_PATH"]
    values = [str(metrics["date"]), str(metrics["group"]), str(MODEL_POSE_PATH)]
    for metric_label, metric_key in metric_names:
        for section in ("current_frame", "temporal_accumulation"):
            for stage in ("raw", "smooth"):
                key = "smoothed" if stage == "smooth" else stage
                headers.append(f"{section}-{stage} {metric_label} [inside/all]")
                pair = []
                for region in ("inside", "all"):
                    metric = metrics["regions"][region][section][key][metric_key]
                    pair.append(f"{metric:.3f}" if metric is not None else "N/A")
                values.append("/".join(pair))

    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if (line.startswith("|") and len(cells) == len(headers)
                    and cells[0] not in {"date", "group", "---"}):
                # 兼容旧的 group/date 列顺序。
                if cells[0].startswith("group_"):
                    cells[0], cells[1] = cells[1], cells[0]
                for index in range(3, len(cells)):
                    if "/" not in cells[index]:
                        cells[index] += "/N/A"
                rows.append(cells)
    rows = [row for row in rows if row[:2] != values[:2]]
    rows.append(values)
    lines = [
        "> 指标单位：mm；每项格式为“椭圆内/全部”，均按整组匹配人次加权。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 3 + ["---:"] * (len(headers) - 3)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SelectedGroupVisualizer:
    def __init__(self, device, show_gt=True, visualize=VISUALIZE):
        self.visualize = visualize
        self.device = torch.device(device)
        if self.device.type == "cuda":
            device_id = self.device.index
            self.device = set_device(
                torch.cuda.current_device() if device_id is None else device_id
            )
        group_path = ROOT_PATH / DATE / "data_collection" / GROUP
        self.paths = sorted((group_path / "dpct高位机/PC").glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"没有点云文件：{group_path / 'dpct高位机/PC'}")
        self.pose_model = load_models(self.device)
        self.gt_folder = group_path / "camera results/smoothed 3D"
        calib_path = ROOT_PATH / DATE / "calib"
        extrinsic_path = calib_path / "extrinsic_img_to_radar_high.npz"
        self.extrinsics = None
        if show_gt and self.gt_folder.is_dir() and extrinsic_path.exists():
            self.extrinsics = load_extrinsics(extrinsic_path)
        elif show_gt:
            print("未找到相机 GT 或外参，继续推理，MPJPE 将为 N/A。")
        low_extrinsic_path = calib_path / "extrinsic_img_to_radar_low.npz"
        if not extrinsic_path.exists() or not low_extrinsic_path.exists():
            raise FileNotFoundError("指标验收与重力坐标系绘图需要高、低位机外参文件")
        high = load_extrinsics(extrinsic_path)
        low = load_extrinsics(low_extrinsic_path)
        rotation = np.asarray(low["R_est"]) @ np.asarray(high["R_est"]).T
        translation = (np.asarray(low["t_est"]).reshape(3)
                       - rotation @ np.asarray(high["t_est"]).reshape(3))
        self.high_to_low = (rotation.astype(np.float32), translation.astype(np.float32))
        self.gt_crop_tracks = {}
        self.gt_accum_tracks = {}
        self.records = []
        self.current_index = 0
        self.group_metrics = None
        self.raw_training_ready = False
        self.fig = None
        self.frame_slider = None
        self.defer_visual_smoothing = False
        if not visualize:
            return
        self.fig = plt.figure(figsize=(14, 10), dpi=80)
        self.axes = np.asarray([self.fig.add_subplot(2, 2, idx + 1, projection="3d")
                                for idx in range(4)]).reshape(2, 2)
        self.fig.subplots_adjust(bottom=.12, hspace=.12, wspace=.05)
        slider_ax = self.fig.add_axes([.15, .035, .70, .025])
        self.frame_slider = Slider(slider_ax, "Frame", 1, len(self.paths),
                                   valinit=1, valstep=1, valfmt="%d")
        self.frame_slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    @torch.inference_mode()
    def infer_frame(self, index):
        if index != len(self.records):
            raise ValueError("新帧必须按时间顺序推理，回看请使用缓存。")
        path = self.paths[index]
        points, skip_reason = load_point_cloud(path)
        if skip_reason is not None:
            print(f"SKIPPED FRAME {DATE}/{GROUP} {path.name}: {skip_reason}")
        if len(points) > TRAIN_MAX_POINTS:
            sample_indices = np.random.default_rng(index).choice(
                len(points), TRAIN_MAX_POINTS, replace=False
            )
        else:
            sample_indices = np.arange(len(points))
        sampled = points[sample_indices]
        record = {"path": str(path)}
        self.records.append(record)
        gt, has_gt = self.load_gt(record)
        for history in self.gt_crop_tracks.values():
            history.append(None)
        gt_pose_tracks, gt_pose_indices, gt_accum_pose_tracks, gt_accum_pose_indices, gt_accumulated_clouds = [], [], [], [], []
        gt_crop_quality, gt_accum_quality = {}, {}
        gt_boxes = []
        gt_selected_mask = np.zeros(len(points), dtype=bool)
        gt_track_ids = np.arange(len(gt), dtype=int)
        for person_index, person_gt in enumerate(gt):
            history = self.gt_crop_tracks.setdefault(person_index, deque(maxlen=T))
            if len(history) == 0:
                history.extend([None] * min(len(self.records), T))
            bbox = np.concatenate((person_gt.min(axis=0) - GT_BBOX_MARGIN,
                                   person_gt.max(axis=0) + GT_BBOX_MARGIN))
            gt_boxes.append(bbox)
            gt_inside_original = ((points[:, :3] >= bbox[:3]) & (points[:, :3] <= bbox[3:])).all(1)
            gt_selected_mask |= gt_inside_original
            inside_sampled = ((sampled[:, :3] >= bbox[:3]) & (sampled[:, :3] <= bbox[3:])).all(1)
            gt_crop_quality[person_index] = point_cloud_quality(sampled[inside_sampled])
            if inside_sampled.any():
                history[-1] = mask_sampled_points(sampled, inside_sampled)
                gt_pose_tracks.append(PoseTrack(person_index, None, history=history))
                gt_pose_indices.append(person_index)
            accum_track = self.gt_accum_tracks.setdefault(person_index, PoseTrack(person_index, None))
            accum_track.history.append(None)
            if skip_reason is not None:
                continue
            accumulated, _, _ = accumulate_track_static_points(
                accum_track, sampled, inside_sampled, (bbox[:3] + bbox[3:]) / 2
            )
            accum_crop, _ = crop_and_pad(
                accumulated, point_mask=np.ones(len(accumulated), dtype=bool),
                max_points=TRAIN_MAX_POINTS
            )
            if accum_crop is not None:
                gt_accum_quality[person_index] = point_cloud_quality(accumulated)
                accum_track.history[-1] = accum_crop
                gt_accum_pose_tracks.append(accum_track)
                gt_accum_pose_indices.append(person_index)
                gt_accumulated_clouds.append(accumulated[np.isfinite(accumulated).all(axis=1)])
        gt_crop_poses = np.empty((0, 17, 3), dtype=np.float32)
        gt_crop_output_indices = np.empty(0, dtype=int)
        output_position = -1 if self.visualize else POSE_OUTPUT_POSITION
        window_ready = self.visualize or index >= T - 1
        if window_ready and gt_pose_tracks:
            gt_pose_input = {key: value.to(self.device) for key, value in build_pose_input(gt_pose_tracks).items()}
            valid = gt_pose_input["mask"][:, output_position].any(dim=1).cpu().numpy()
            gt_crop_poses = self.pose_model(gt_pose_input)["pose"][:, output_position, 0].cpu().numpy()[valid]
            gt_crop_output_indices = np.asarray(gt_pose_indices, dtype=int)[valid]
        gt_accum_poses = np.empty((0, 17, 3), dtype=np.float32)
        gt_accum_output_indices = np.empty(0, dtype=int)
        if window_ready and gt_accum_pose_tracks:
            gt_accum_input = {key: value.to(self.device) for key, value in build_pose_input(gt_accum_pose_tracks).items()}
            valid = gt_accum_input["mask"][:, output_position].any(dim=1).cpu().numpy()
            gt_accum_poses = self.pose_model(gt_accum_input)["pose"][:, output_position, 0].cpu().numpy()[valid]
            gt_accum_output_indices = np.asarray(gt_accum_pose_indices, dtype=int)[valid]
        record.update(gt_crop_track_ids=gt_track_ids,
                      gt_crop_pose_indices=np.empty(0, dtype=int),
                      gt_crop_poses_raw=np.empty((0, 17, 3), dtype=np.float32),
                      gt_crop_quality=gt_crop_quality,
                      gt_boxes=np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 6),
                      gt_selected_mask=gt_selected_mask,
                      gt_accum_track_ids=gt_track_ids,
                      gt_accum_pose_indices=np.empty(0, dtype=int),
                      gt_accum_poses_raw=np.empty((0, 17, 3), dtype=np.float32),
                      gt_accum_quality=gt_accum_quality,
                      gt_accumulated_person_indices=np.asarray(gt_accum_pose_indices, dtype=int),
                      gt_accumulated_points_by_person=gt_accumulated_clouds,
                      gt_accumulated_points=np.concatenate(gt_accumulated_clouds) if gt_accumulated_clouds else points[:0])
        if window_ready:
            target = record if self.visualize else self.records[index - POSE_LOOKAHEAD]
            target["gt_crop_pose_indices"] = gt_crop_output_indices
            target["gt_crop_poses_raw"] = gt_crop_poses
            target["gt_accum_pose_indices"] = gt_accum_output_indices
            target["gt_accum_poses_raw"] = gt_accum_poses
        if self.visualize and not self.defer_visual_smoothing:
            smooth_pose_records(self.records, "gt_crop_")
            smooth_pose_records(self.records, "gt_accum_")
        return record, points, gt_selected_mask

    def load_gt(self, record):
        if "gt" not in record:
            record["gt"] = np.empty((0, 17, 3), dtype=np.float32)
            record["has_gt"] = False
            if self.extrinsics is not None:
                pose_file = closest_timestamp_file(Path(record["path"]), self.gt_folder)
                if pose_file is not None:
                    record["gt_pose_path"] = str(pose_file)
                    with pose_file.open("rb") as source:
                        gt = transform_points(np.asarray(pickle.load(source)),
                                              self.extrinsics["R_est"],
                                              np.asarray(self.extrinsics["t_est"]).reshape(3))
                    record["gt"] = np.asarray(gt).reshape(-1, 17, 3)
                    record["has_gt"] = True
        return record["gt"], record["has_gt"]

    @torch.inference_mode()
    def infer_gt_raw_as_training(self, collect_quality=True):
        """按实验的点云积累、裁剪和重采样流程重算完整滑窗结果。"""
        if self.raw_training_ready:
            return
        from data2datasets.dataset_for_all_task import HPE_Dataset, collate_fn
        from run.utils.load_config import load_config
        from run.utils.process_one_epoch import resample_cropped_pointcloud

        training_data = load_config(MODEL_POSE_PATH / "config/config.yaml")["data"]
        if training_data["T"] != T or training_data["max_points"] != TRAIN_MAX_POINTS:
            raise ValueError("推理 T/max_points 与训练实验配置不一致")

        for record in self.records:
            record["gt_crop_poses_raw"] = np.empty((0, 17, 3), dtype=np.float32)
            record["gt_crop_pose_indices"] = np.empty(0, dtype=int)
            record["gt_crop_input_points"] = {}
            record["gt_crop_quality"] = {}
            record["gt_training_window"] = False

        point_sequences = []
        gt_sequences = []
        for record in self.records:
            points, _ = load_point_cloud(record["path"])
            point_sequences.append(points)
            gt_sequences.append(self.load_gt(record)[0])
        point_sequences = HPE_Dataset._accumulate_pointcloud_sequence(
            point_sequences, 0, training_data.get("acc_frame", 0)
        )

        for start in range(len(self.records) - T + 1):
            window = self.records[start:start + T]
            # collate_fn 内部使用 torch.randperm 下采样；固定每个滑窗的种子，
            # 否则后处理 A/B 两次运行的 raw 输入也会不同。
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(start)
                samples = collate_fn(
                    [{"radar_high_pc": point_sequences[start:start + T],
                      "gt_for_high": gt_sequences[start:start + T]}],
                    max_points=training_data["max_points"],
                    max_people=training_data["max_people"],
                )
            points = samples["radar_high_pc"]["padded"]
            point_mask = samples["radar_high_pc"]["mask"]
            gt_data = samples["gt_for_high"]
            person_mask = gt_data["mask"].permute(0, 2, 1)
            bbox = gt_data["bbox"].permute(0, 2, 1, 3)
            xyz = points[:, None, :, :, :3]
            cropped_mask = (
                ((xyz >= bbox[..., :3].unsqueeze(3)) &
                 (xyz <= bbox[..., 3:].unsqueeze(3))).all(dim=-1)
                & point_mask[:, None]
                & person_mask.unsqueeze(-1)
            )
            cropped_points = points[:, None].expand(-1, person_mask.shape[1], -1, -1, -1)
            cropped_points = cropped_points.masked_fill(~cropped_mask.unsqueeze(-1), 0.0)
            valid_people = (person_mask.any(dim=2) & cropped_mask.any(dim=(2, 3)))[0]
            if not valid_people.any():
                continue
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(start)
                model_points, model_mask = resample_cropped_pointcloud(
                    cropped_points[0, valid_people], cropped_mask[0, valid_people]
                )
            prediction = self.pose_model({
                "input": model_points.to(self.device),
                "mask": model_mask.to(self.device),
            })["pose"][:, :, 0].cpu().numpy()
            people = torch.nonzero(valid_people, as_tuple=True)[0].cpu().numpy()
            present = person_mask[0, valid_people, POSE_OUTPUT_POSITION].cpu().numpy()
            gt_present = present.copy()
            # 当前输出帧框内有效输入点严格大于 20，才保存该人的预测。
            present &= (
                cropped_mask[0, valid_people, POSE_OUTPUT_POSITION].sum(dim=-1) > 20
            ).cpu().numpy()
            record = window[POSE_OUTPUT_POSITION]
            record["gt_crop_poses_raw"] = prediction[present, POSE_OUTPUT_POSITION]
            record["gt_crop_pose_indices"] = people[present]
            record["gt_crop_quality"] = {
                int(person): point_cloud_quality(
                    cropped_points[0, person, POSE_OUTPUT_POSITION][
                        cropped_mask[0, person, POSE_OUTPUT_POSITION]
                    ].cpu().numpy()
                )
                for person in (people[gt_present] if collect_quality else [])
            }
            record["gt_crop_input_points"] = {
                int(person): cropped_points[0, person, POSE_OUTPUT_POSITION][
                    cropped_mask[0, person, POSE_OUTPUT_POSITION]
                ].cpu().numpy()
                for person in people[gt_present]
            }
            record["gt_training_window"] = True
        self.raw_training_ready = True

    def save_raw_cache(self, path):
        """保存一次确定性推理的未后处理数据，供离线参数搜索。"""
        if len(self.records) != len(self.paths):
            raise ValueError("整组尚未推理完成，不能保存 raw cache")
        self.infer_gt_raw_as_training()
        frames = []
        for record in self.records:
            points, _ = load_point_cloud(record["path"])
            frames.append({
                "path": record["path"],
                "gt_pose_path": record.get("gt_pose_path"),
                "points": points,
                "gt": record["gt"],
                "has_gt": record["has_gt"],
                "gt_boxes": record["gt_boxes"],
                "current_track_ids": record["gt_crop_track_ids"],
                "current_pose_indices": record["gt_crop_pose_indices"],
                "current_poses_raw": record["gt_crop_poses_raw"],
                "current_input_points": record["gt_crop_input_points"],
                "current_quality": record["gt_crop_quality"],
                "current_training_window": record["gt_training_window"],
                "accum_track_ids": record["gt_accum_track_ids"],
                "accum_pose_indices": record["gt_accum_pose_indices"],
                "accum_poses_raw": record["gt_accum_poses_raw"],
                "accum_input_points": {
                    int(person): cloud for person, cloud in zip(
                        record["gt_accumulated_person_indices"],
                        record["gt_accumulated_points_by_person"],
                    )
                },
                "accum_quality": record["gt_accum_quality"],
            })
        cache = {
            "version": 1,
            "date": DATE,
            "group": GROUP,
            "model_pose_path": str(MODEL_POSE_PATH),
            "pose_output_position": POSE_OUTPUT_POSITION,
            "high_to_low": self.high_to_low,
            "frames": frames,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as destination:
            pickle.dump(cache, destination, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        print(f"raw cache 已保存：{path} ({path.stat().st_size / 1024**2:.1f} MiB)")
        return path

    def evaluate_group(self):
        """全组统一计算平滑前、平滑后指标，均按匹配人次加权。"""
        if len(self.records) != len(self.paths):
            raise ValueError("整组尚未推理完成，不能报告整组 MPJPE")
        if self.group_metrics is not None:
            return self.group_metrics
        self.infer_gt_raw_as_training()
        held_counts = {
            "current_frame": smooth_pose_records(self.records, "gt_crop_"),
            "temporal_accumulation": smooth_pose_records(self.records, "gt_accum_"),
        }

        def summarize(prefix, inside_only, training_windows=False):
            result = {}
            for stage, suffix in (("raw", "poses_raw"), ("smoothed", "poses")):
                triples = []
                for metric in (frame_mpjpe, frame_centered_mpjpe, frame_pa_mpjpe):
                    stats = []
                    for record in self.records:
                        poses = record[f"{prefix}{suffix}"]
                        gt = record["gt"]
                        if inside_only:
                            inside = acceptance_mask(gt, *self.high_to_low)
                            accepted_ids = np.flatnonzero(inside)
                            ids_key = (f"{prefix}pose_indices" if stage == "raw"
                                       else f"{prefix}pose_track_ids")
                            poses = poses[np.isin(record[ids_key], accepted_ids)]
                            gt = gt[inside]
                        if training_windows and not record["gt_training_window"]:
                            gt = gt[:0]
                        stats.append(metric(poses, gt))
                    triples.append(aggregate_mpjpe(stats)["mpjpe_mm"])
                result[stage] = {
                    "mpjpe_mm": triples[0],
                    "centered_mpjpe_mm": triples[1],
                    "pa_mpjpe_mm": triples[2],
                }
            return result

        regions = {
            "inside": {
                "current_frame": summarize("gt_crop_", True, training_windows=True),
                "temporal_accumulation": summarize("gt_accum_", True),
            },
            "all": {
                "current_frame": summarize("gt_crop_", False, training_windows=True),
                "temporal_accumulation": summarize("gt_accum_", False),
            },
        }
        self.group_metrics = {
            "date": DATE, "group": GROUP, "frames": len(self.records),
            "gt_frames": sum(record["has_gt"] for record in self.records),
            "quality_hold_enabled": QUALITY_HOLD,
            "direction_hold_enabled": DIRECTION_HOLD,
            "quality_held_observations": held_counts,
            "regions": regions,
        }
        print(f"点云质量门控：{'ON' if QUALITY_HOLD else 'OFF'}；保持观测数 "
              f"current={held_counts['current_frame']['quality']}, "
              f"accumulated={held_counts['temporal_accumulation']['quality']}")
        print(f"运动朝向翻转抑制：{'ON' if DIRECTION_HOLD else 'OFF'}；保持观测数 "
              f"current={held_counts['current_frame']['direction']}, "
              f"accumulated={held_counts['temporal_accumulation']['direction']}")
        fmt = lambda value: "N/A" if value is None else f"{value:.3f} mm"
        for region, region_label in (("inside", "椭圆内"), ("all", "全部")):
            for section, section_label in (("current_frame", "当前帧GT裁剪"),
                                           ("temporal_accumulation", "时域积累GT裁剪")):
                for stage, stage_label in (("raw", "平滑前"), ("smoothed", "平滑后")):
                    values = regions[region][section][stage]
                    print(f"{region_label} {section_label}{stage_label} MPJPE: {fmt(values['mpjpe_mm'])}；"
                          f"Centered MPJPE: {fmt(values['centered_mpjpe_mm'])}；"
                          f"PA-MPJPE: {fmt(values['pa_mpjpe_mm'])}")
        return self.group_metrics


    def inside_frame_mpjpe(self, record, prefix, stage):
        inside_ids = np.flatnonzero(acceptance_mask(record["gt"], *self.high_to_low))
        poses = record[f"{prefix}{'poses_raw' if stage == 'raw' else 'poses'}"]
        ids_suffix = "pose_indices" if stage == "raw" else "pose_track_ids"
        pose_ids = record[f"{prefix}{ids_suffix}"]
        return frame_mpjpe(poses[np.isin(pose_ids, inside_ids)], record["gt"][inside_ids])[0]

    def save_metric_timeseries(self):
        """按人员保存椭圆内当前帧 GT 裁剪的 raw/smoothed MPJPE。"""
        frame_indices = np.arange(1, len(self.records) + 1)
        frame_ticks = decimal_frame_ticks(len(self.records))
        output_dir = METRIC_VIS_ROOT / DATE / GROUP
        output_dir.mkdir(parents=True, exist_ok=True)
        max_people = max((len(record["gt"]) for record in self.records), default=0)
        stage_values = {}
        for stage in ("raw", "smoothed"):
            stage_values[stage] = {}
            suffix = "poses_raw" if stage == "raw" else "poses"
            ids_suffix = "pose_indices" if stage == "raw" else "pose_track_ids"
            for person_index in range(max_people):
                values = []
                for record in self.records:
                    inside = acceptance_mask(record["gt"], *self.high_to_low)
                    poses = record[f"gt_crop_{suffix}"]
                    pose_ids = record[f"gt_crop_{ids_suffix}"]
                    matches = np.flatnonzero(pose_ids == person_index)
                    if (person_index >= len(inside) or not inside[person_index]
                            or not len(matches)):
                        values.append(np.nan)
                        continue
                    value = frame_mpjpe(
                        poses[matches[:1]], record["gt"][person_index:person_index + 1]
                    )[0]
                    values.append(np.nan if value is None else value)
                stage_values[stage][person_index] = values
        all_series = [values for stage in stage_values.values() for values in stage.values()]
        finite_values = np.asarray(all_series)
        finite_values = finite_values[np.isfinite(finite_values)]
        shared_ymax = float(finite_values.max() * 1.05) if len(finite_values) else 1.0
        shared_ymax = max(shared_ymax, METRIC_REFERENCE_MM * 1.05)

        action_classes = np.full((len(self.records), max_people), -1, dtype=int)
        for frame_index, record in enumerate(self.records):
            pose_path = record.get("gt_pose_path")
            if pose_path is None:
                continue
            action_dir = Path(pose_path).parent.parent / "action label"
            candidates = [action_dir / f"{Path(pose_path).stem}{suffix}"
                          for suffix in (".pkl", ".npz")]
            action_path = next((path for path in candidates if path.exists()), None)
            if action_path is None:
                continue
            if action_path.suffix == ".npz":
                with np.load(action_path) as data:
                    action_data = data["labels"]
            else:
                with action_path.open("rb") as source:
                    loaded = pickle.load(source)
                action_data = loaded["labels"] if isinstance(loaded, dict) else loaded
            action_data = np.asarray(action_data)
            inside = acceptance_mask(record["gt"], *self.high_to_low)
            if action_data.ndim == 2 and action_data.shape[0] == len(inside):
                inside_indices = np.flatnonzero(inside)
                action_classes[frame_index, inside_indices] = np.argmax(
                    action_data[inside_indices], axis=1
                )

        output_paths = []
        person_colors = plt.get_cmap("tab10")
        for stage, title in (("raw", "Raw"), ("smoothed", "Smoothed")):
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            for person_index, values in stage_values[stage].items():
                if np.isfinite(values).any():
                    ax.plot(frame_indices, values, color=person_colors(person_index % 10),
                            linewidth=1, label=f"P{person_index} MPJPE")
            ax.axhline(METRIC_REFERENCE_MM, color="red", linewidth=1.5,
                       linestyle="--", label=f"{METRIC_REFERENCE_MM:.0f} mm threshold")
            action_legend_added = set()
            band_height, band_gap = .025, .008
            for person_index in range(max_people):
                band_top = .98 - person_index * (band_height + band_gap)
                band_bottom = band_top - band_height
                if not np.any(action_classes[:, person_index] >= 0):
                    continue
                ax.text(.005, (band_bottom + band_top) / 2, f"P{person_index}",
                        transform=ax.transAxes, fontsize=7, va="center", zorder=4,
                        bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": 1})
                for class_index, (label, color) in enumerate(zip(ACTION_LABELS, ACTION_COLORS)):
                    active = action_classes[:, person_index] == class_index
                    transitions = np.diff(np.pad(active.astype(np.int8), (1, 1)))
                    starts = np.flatnonzero(transitions == 1)
                    ends = np.flatnonzero(transitions == -1)
                    for start, end in zip(starts, ends):
                        legend_label = None
                        if class_index not in action_legend_added:
                            legend_label = f"action: {label}"
                            action_legend_added.add(class_index)
                        ax.axvspan(start + .5, end + .5, ymin=band_bottom, ymax=band_top,
                                   color=color, alpha=.65, label=legend_label, zorder=.5)
            ax.set(title=f"{DATE}/{GROUP} - {title} current GT crop (inside acceptance ellipse)",
                   xlabel="Frame idx", ylabel="Error [mm]", xlim=(0, len(self.records)),
                   ylim=(0, shared_ymax), xticks=frame_ticks)
            ax.grid(alpha=.25)
            ax.legend()
            output_path = output_dir / f"{stage}_metric_timeseries.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            output_paths.append(output_path)
        print(f"逐帧指标图已分别保存到 {output_dir}")
        return output_paths

    def save_point_count_timeseries(self):
        """保存验收椭圆内人体 GT 框中的动态/静态点云数量。"""
        dynamic_counts, static_counts = [], []
        for record in self.records:
            points, _ = load_point_cloud(record["path"])
            inside_gt = acceptance_mask(record["gt"], *self.high_to_low)
            selected = np.zeros(len(points), dtype=bool)
            for box in record["gt_boxes"][inside_gt]:
                selected |= ((points[:, :3] >= box[:3]) &
                             (points[:, :3] <= box[3:])).all(axis=1)
            dynamic = selected & np.isclose(points[:, -1], DYNAMIC_POINT_TAG)
            dynamic_counts.append(int(dynamic.sum()))
            static_counts.append(int((selected & ~dynamic).sum()))

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        frame_indices = np.arange(1, len(self.records) + 1)
        frame_ticks = decimal_frame_ticks(len(self.records))
        ax.plot(frame_indices, dynamic_counts, label="Dynamic points", linewidth=1)
        ax.plot(frame_indices, static_counts, label="Static points", linewidth=1)
        ax.set(title=f"{DATE}/{GROUP} - Point counts inside acceptance ellipse",
               xlabel="Frame idx", ylabel="Point count", xlim=(0, len(self.records)),
               xticks=frame_ticks)
        ax.grid(alpha=.25)
        ax.legend()
        output_dir = METRIC_VIS_ROOT / DATE / GROUP
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "point_count_timeseries.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"逐帧点云数量图已保存到 {output_path}")
        return output_path

    def save_high_error_frames(self):
        """保存椭圆内 current raw/smoothed MPJPE 均超阈值的最差 20 帧。"""
        output_dir = ERROR_VIS_ROOT / DATE / GROUP
        selected = []
        for index, record in enumerate(self.records):
            current_raw_error = self.inside_frame_mpjpe(record, "gt_crop_", "raw")
            current_smooth_error = self.inside_frame_mpjpe(record, "gt_crop_", "smoothed")
            if all(value is not None and value > ERROR_VIS_THRESHOLD_MM
                   for value in (current_raw_error, current_smooth_error)):
                selected.append((index, current_raw_error, current_smooth_error))
        selected.sort(key=lambda item: item[2], reverse=True)
        selected = selected[:ERROR_VIS_MAX_FRAMES]
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_png in output_dir.glob("*.png"):
            old_png.unlink()
        if not selected:
            print(f"{DATE}/{GROUP}: 没有椭圆内 current raw 和 smoothed MPJPE 均超过 "
                  f"{ERROR_VIS_THRESHOLD_MM:.0f} mm 的帧")
            return 0

        self.fig = plt.figure(figsize=(14, 10), dpi=100, constrained_layout=True)
        self.axes = np.asarray([self.fig.add_subplot(2, 2, idx + 1, projection="3d")
                                for idx in range(4)]).reshape(2, 2)
        for index, current_raw_error, current_smooth_error in selected:
            fmt = lambda value: "N-A" if value is None else f"{value:.1f}mm"
            note = (f"ellipse-only  current_raw={fmt(current_raw_error)}  "
                    f"current_smooth={fmt(current_smooth_error)}")
            self.show_frame(index, note=note)
            frame_name = Path(self.records[index]["path"]).stem
            filename = (f"{DATE}_{GROUP}_frame_{index + 1:06d}_{frame_name}_ellipse-only_"
                        f"current_raw_{fmt(current_raw_error)}_"
                        f"current_smooth_{fmt(current_smooth_error)}.png")
            self.fig.savefig(output_dir / filename, dpi=100)
        plt.close(self.fig)
        self.fig = None
        print(f"{DATE}/{GROUP}: 已保存 {len(selected)} 帧到 {output_dir}")
        return len(selected)

    def show_frame(self, index, note=None):
        if index == len(self.records):
            record, points, _ = self.infer_frame(index)
        else:
            record = self.records[index]
            points, _ = load_point_cloud(record["path"])
        gt, has_gt = self.load_gt(record)
        rotation, translation = self.high_to_low
        plot_points = transform_cloud(points, rotation, translation)
        inside_gt = acceptance_mask(gt, rotation, translation)
        inside_ids = np.flatnonzero(inside_gt)
        metric_gt = gt[inside_gt]
        plot_gt = transform_points(metric_gt, rotation, translation)
        visible_boxes = record["gt_boxes"][inside_gt]
        visible_point_mask = np.zeros(len(points), dtype=bool)
        for box in visible_boxes:
            visible_point_mask |= ((points[:, :3] >= box[:3]) &
                                   (points[:, :3] <= box[3:])).all(axis=1)

        angle = np.linspace(0, 2 * np.pi, 80)
        radius = np.linspace(0, 1, 16)
        acceptance_x = ACCEPTANCE_CENTER_XY[0] + ACCEPTANCE_RADII_XY[0] * np.outer(radius, np.cos(angle))
        acceptance_y = ACCEPTANCE_CENTER_XY[1] + ACCEPTANCE_RADII_XY[1] * np.outer(radius, np.sin(angle))
        acceptance_z = np.full_like(acceptance_x, XYZ_LIMITS[2][0])
        for ax in self.axes.flat:
            ax.clear()
            for dimension, limits in zip("xyz", XYZ_LIMITS):
                getattr(ax, f"set_{dimension}lim")(*limits)
                getattr(ax, f"set_{dimension}label")(f"{dimension.upper()} [m]")
            ax.plot_surface(acceptance_x, acceptance_y, acceptance_z,
                            color=ACCEPTANCE_COLOR, alpha=.35, shade=False)

        def draw_points(ax, cloud, selected=None):
            selected = np.ones(len(cloud), dtype=bool) if selected is None else selected
            dynamic = selected & np.isclose(cloud[:, -1], DYNAMIC_POINT_TAG)
            static = selected & ~dynamic
            outside = ~selected
            ax.scatter(*cloud[outside, :3].T, c="gray", s=2, alpha=.12)
            ax.scatter(*cloud[dynamic, :3].T, c="red", s=5)
            ax.scatter(*cloud[static, :3].T, c="blue", s=5)
            ax.legend(handles=[
                Line2D([], [], color="red", marker=".", linestyle="None", label="Selected dynamic (tag=1)"),
                Line2D([], [], color="blue", marker=".", linestyle="None", label="Selected static (tag=2..7)"),
                Line2D([], [], color="gray", marker=".", linestyle="None", label="Outside GT boxes"),
            ], fontsize=7.5, loc="upper left")

        def draw_boxes(ax):
            edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                     (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
            for box in transform_boxes(visible_boxes, rotation, translation):
                lo, hi = box[:3], box[3:]
                corners = np.asarray([[x, y, z] for z in (lo[2], hi[2])
                                      for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
                for start, end in edges:
                    ax.plot(*corners[[start, end]].T, color="green", linewidth=1.5)

        accumulated_clouds = [cloud for person_index, cloud in zip(
            record["gt_accumulated_person_indices"], record["gt_accumulated_points_by_person"]
        ) if person_index in inside_ids]
        accumulated = transform_cloud(
            np.concatenate(accumulated_clouds) if accumulated_clouds else points[:0],
            rotation, translation,
        )
        draw_points(self.axes[0, 0], plot_points, visible_point_mask)
        draw_boxes(self.axes[0, 0])
        self.axes[0, 0].set_title("Current points: GT crop")
        draw_points(self.axes[1, 0], accumulated)
        draw_boxes(self.axes[1, 0])
        self.axes[1, 0].set_title("Accumulated points: GT crop")

        def errors(poses):
            return (frame_mpjpe(poses, metric_gt)[0],
                    frame_centered_mpjpe(poses, metric_gt)[0],
                    frame_pa_mpjpe(poses, metric_gt)[0])

        def metric_label(name, poses):
            values = errors(poses)
            formatted = ["N/A" if value is None else f"{value:.1f}" for value in values]
            return f"{name}: MPJPE/Centered/PA={'/'.join(formatted)} mm"

        for ax, raw, raw_ids, smooth, smooth_ids, title in (
            (self.axes[0, 1], record["gt_crop_poses_raw"], record["gt_crop_pose_indices"],
             record["gt_crop_poses"], record["gt_crop_pose_track_ids"], "Current crop pose: GT"),
            (self.axes[1, 1], record["gt_accum_poses_raw"], record["gt_accum_pose_indices"],
             record["gt_accum_poses"], record["gt_accum_pose_track_ids"], "Accumulated pose: GT"),
        ):
            raw = raw[np.isin(raw_ids, inside_ids)]
            smooth = smooth[np.isin(smooth_ids, inside_ids)]
            if has_gt:
                draw_human_pose(ax, plot_gt, "green", "Camera GT")
            draw_human_pose(ax, transform_points(raw, rotation, translation), "orange", "Raw")
            draw_human_pose(ax, transform_points(smooth, rotation, translation), "magenta", "Smoothed")
            ax.set_title(title)
            ax.legend(handles=[
                Line2D([], [], color="green", marker="o", label="Camera GT"),
                Line2D([], [], color="orange", marker="o", label=metric_label("Raw", raw)),
                Line2D([], [], color="magenta", marker="o", label=metric_label("Smoothed", smooth)),
            ], fontsize=7.5, loc="upper left")
        title = f"date={DATE}  group={GROUP}  frame={index + 1}/{len(self.paths)}  {Path(record['path']).name}"
        if note:
            title += f"  |  {note}"
        self.fig.suptitle(title)
        self.fig.canvas.draw_idle()
        if len(self.records) == len(self.paths) and not self.visualize:
            self.evaluate_group()
        return

    def precompute_visualization(self):
        """WebAgg 启动前完成整组末位推理，最后统一执行一次平滑。"""
        self.defer_visual_smoothing = True
        try:
            for index in tqdm(range(len(self.records), len(self.paths)),
                              desc="Precomputing WebAgg frames"):
                self.infer_frame(index)
        finally:
            self.defer_visual_smoothing = False
        smooth_pose_records(self.records, "gt_crop_")
        smooth_pose_records(self.records, "gt_accum_")

    def on_slider(self, value):
        index = int(value) - 1
        if index == self.current_index:
            return
        for next_index in range(len(self.records), index + 1):
            self.infer_frame(next_index)
        self.show_frame(index)
        self.current_index = index

    def on_key(self, event):
        if event.key in (" ", "right"):
            index = min(self.current_index + 1, len(self.paths) - 1)
        elif event.key == "left":
            index = max(self.current_index - 1, 0)
        else:
            return
        if index != self.current_index:
            self.frame_slider.set_val(index + 1)


def self_test():
    """无需权重/GPU 的 GT-only 数据处理自检。"""
    points = np.array([[1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8],
                       [20, 20, 20, 0, 0, 0]], dtype=np.float32)
    crop, inside = crop_and_pad(points, np.array([0, 0, 0, 5, 5, 6]))
    np.testing.assert_array_equal(crop[0][crop[1]], points[inside])
    padded, mask = mask_sampled_points(points, np.array([False, True, False]))
    assert padded.shape == (TRAIN_MAX_POINTS, 6) and mask.sum() == 1
    track = PoseTrack(0, history=deque([crop], maxlen=T))
    pose_input = build_pose_input([track])
    assert pose_input["input"].shape == (1, T, MAX_POINTS, 6)
    assert torch.isfinite(pose_input["input"]).all()
    gt = np.zeros((1, 17, 3), dtype=np.float32)
    pred = gt.copy()
    pred[..., 1] = .1
    np.testing.assert_allclose(frame_mpjpe(pred, gt)[0], 100, atol=1e-4)
    np.testing.assert_allclose(frame_centered_mpjpe(pred, gt)[0], 0, atol=1e-4)
    sparse_static = np.array([[0, 0, 0, 0, 0, 2], [.05, 0, 0, 0, 0, 2]], dtype=np.float32)
    sparse_dynamic = sparse_static.copy()
    sparse_dynamic[:, -1] = DYNAMIC_POINT_TAG
    assert point_cloud_quality(sparse_static)["hold"]
    assert not point_cloud_quality(sparse_dynamic)["hold"]
    pose0, pose1 = np.zeros((17, 3), dtype=np.float32), np.ones((17, 3), dtype=np.float32)
    records = [
        {"poses_raw": pose0[None], "pose_indices": np.array([0]),
         "track_ids": np.array([7]), "quality": {7: {"hold": False}}},
        {"poses_raw": pose1[None], "pose_indices": np.array([0]),
         "track_ids": np.array([7]), "quality": {7: {"hold": True}}},
    ]
    counts = smooth_pose_records(records, quality_hold=True, direction_hold=False)
    assert counts == {"quality": 1, "direction": 0}
    np.testing.assert_allclose(records[1]["poses"], pose0[None])
    facing_pose = np.zeros((17, 3), dtype=np.float32)
    facing_pose[5], facing_pose[6] = [-.2, 0, 1], [.2, 0, 1]
    facing_pose[11], facing_pose[12] = [-.2, 0, 0], [.2, 0, 0]
    flipped_pose = facing_pose.copy()
    flipped_pose[:, :2] *= -1
    flipped_pose[:, 0] += .1
    stabilized, count = suppress_direction_flips(np.stack((facing_pose, flipped_pose)))
    assert count == 1
    np.testing.assert_allclose(stabilized[1] - stabilized[1, [11, 12]].mean(0),
                               facing_pose - facing_pose[[11, 12]].mean(0))
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-gt", action="store_true", help="不读取相机 GT")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=VISUALIZE,
                        help="覆盖 VISUALIZE 开关，启用/关闭 WebAgg")
    parser.add_argument("--smoke-frames", type=int, default=0, help="顺序推理指定帧数并退出，不启动 Web 服务")
    parser.add_argument(
        "--raw-cache", type=Path,
        default=Path(f"/home/pai/Huawei/temp/Inference_selected_group_{DATE}_{GROUP}_raw.pkl"),
        help="离线模式保存未后处理数据的路径",
    )
    parser.add_argument("--save-raw-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="离线模式是否保存 raw cache")
    parser.add_argument("--raw-only", action="store_true",
                        help="保存 raw cache 后直接退出，不执行任何姿态后处理")
    args = parser.parse_args()
    if args.raw_only and not args.save_raw_cache:
        parser.error("--raw-only 需要保留 --save-raw-cache")
    if args.self_test:
        self_test()
        return
    app = SelectedGroupVisualizer(args.device, show_gt=not args.no_gt,
                                  visualize=args.visualize or bool(args.smoke_frames))
    if args.smoke_frames:
        for index in range(min(args.smoke_frames, len(app.paths))):
            app.show_frame(index)
            record = app.records[-1]
            assert np.isfinite(record["gt_crop_poses"]).all()
            assert app.axes.shape == (2, 2)
            print(f"frame {index}: gt_poses={len(record['gt_crop_poses'])}, "
                  f"gt_accum_poses={len(record['gt_accum_poses'])}")
        app.fig.canvas.draw()
        # 回看不能推进历史状态或再次执行模型。
        record_count = len(app.records)
        app.show_frame(0)
        assert len(app.records) == record_count
        plt.close(app.fig)
        return
    if not args.visualize:
        for index in tqdm(range(len(app.paths)), desc="GT crop inference"):
            app.infer_frame(index)
        if args.save_raw_cache:
            app.save_raw_cache(args.raw_cache)
        if args.raw_only:
            return
        metrics = app.evaluate_group()
        append_group_metrics_markdown(metrics, Path(__file__).with_name("record.md"))
        app.save_high_error_frames()
        app.save_metric_timeseries()
        app.save_point_count_timeseries()
        return
    app.precompute_visualization()
    app.show_frame(0)
    print("WebAgg 使用端口 8988；空格/右箭头下一帧，左箭头回看。")
    plt.show()


if __name__ == "__main__":
    main()
