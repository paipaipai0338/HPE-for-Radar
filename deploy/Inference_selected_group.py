"""训练式 4 帧积累、GT 裁剪、200 点重采样 → P4Transformer → WebAgg。

运行：python deploy/Inference_selected_group.py（VISUALIZE=False 时自动评估整组）
VISUALIZE=True 或 --visualize：空格/右箭头下一帧，左箭头缓存回看；组末输出整组指标。
三图分别显示当前单帧 GT 框裁剪点云（积累和重采样前）、椭圆内姿态和全部人体姿态。
下方单图按窗口末帧显示 Raw、Guard+躺姿锁、Keeper+躺姿锁的 MPJPE。
实线为椭圆内人体，虚线为椭圆外人体。
INFERENCE_TRAIN_CONFIG 可显式指定与权重配套的训练配置。
优先读取 /mnt/ssd/Huawei 打包数据，缺失时回退 /mnt/huawei；
INFERENCE_PACKED_DATA_ROOT 可覆盖 SSD 根目录。
仅对连续 T 帧有效的人体窗口取末帧；缺帧后重新等待 T 帧。
使用 last.pth；点数不足持续保留可信骨架，点数充足但突变时离线三帧确认/回填。
骨架转到低位机重力坐标系后判决站立/行走、坐/蹲、躺卧、其他，分别标注 GT/Raw/Guarded 行为。
默认只输出终端指标；--save-results / --save-raw-cache 显式启用结果保存。
"""

import argparse
import importlib.util
import io
import json
import re
from functools import lru_cache
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
import torch
from tqdm import tqdm

from run.utils.set_device import set_device
from run.utils.load_config import load_config
from preprocess.actionprocess import LABEL_NAMES, classify_actions
from utils.skeleton import get_pose_layout
from run.utils.temporal_guard import (TEMPORAL_GUARD_PARAMS, guard_metric_protocol,
                                      guard_pose_records, pose_change, pose_change_within,
                                      match_poses, frame_mpjpe)
from run.utils.group_metric_format import format_group_metrics
from run.utils.direction_keeper import DirectionKeeper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_PATH = Path("/mnt/huawei")
PACKED_DATA_ROOT = Path(os.environ.get("INFERENCE_PACKED_DATA_ROOT", "/mnt/ssd/Huawei"))
DATE = os.environ.get("INFERENCE_DATE", "20261022")
GROUP = os.environ.get("INFERENCE_GROUP", "group_015")
VISUALIZE = True   # False 自动跑完整组并输出 MPJPE； True WebAgg 逐帧交互。
MODEL_POSE_PATH = PROJECT_ROOT / "experiments/P4Transformer/20261019_161657"
TRAIN_CONFIG_PATH = Path(os.environ.get("INFERENCE_TRAIN_CONFIG", str(MODEL_POSE_PATH / "config" / "config.yaml")))
TRAIN_CONFIG = load_config(TRAIN_CONFIG_PATH)
T = TRAIN_CONFIG["data"]["T"]
POSE_OUTPUT_POSITION = T - 1  # 完整窗口的末帧。
TRAIN_MAX_POINTS = TRAIN_CONFIG["data"].get("person_max_points", 200)
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
ACTION_LABELS = LABEL_NAMES
ACTION_COLORS = ("#4daf4a", "#377eb8", "#e41a1c", "#984ea3")


DYNAMIC_POINT_TAG = 1.0

DIRECTED_BONES = [
    (11, 12), (11, 5), (12, 6), (11, 13), (13, 15), (12, 14),
    (14, 16), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 0), (0, 1), (1, 3), (0, 2), (2, 4),
]


def load_models(device):
    from run.utils.checkpoint import load_model_checkpoint

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
    load_model_checkpoint(MODEL_POSE_PATH / "checkpoint/last.pth", pose_model, device)
    return pose_model.eval()


@lru_cache(maxsize=None)
def packed_directory(directory):
    """读取与 dataset 相同的 SSD 打包索引，保留原始路径作为帧标识。"""
    from data2datasets.dataset_for_all_task import HPE_Dataset

    directory = Path(directory)
    pack_name = {"PC": "frames.pcpack", "smoothed 3D": "frames.gtpack",
                 "action label": "frames.actionpack"}.get(directory.name)
    if pack_name is None or not directory.is_relative_to(ROOT_PATH):
        return None
    packed_dir = PACKED_DATA_ROOT / directory.relative_to(ROOT_PATH)
    pack_path = packed_dir / pack_name
    index_path = packed_dir / "frames_index.npz"
    if not (HPE_Dataset._packed_path_exists(pack_path)
            and HPE_Dataset._packed_path_exists(index_path)):
        return None
    with np.load(index_path, allow_pickle=False) as index:
        names = index["frame_names"].astype(str).reshape(-1)
        offsets = index["offsets"].astype(np.int64).reshape(-1)
        lengths = index["lengths"].astype(np.int64).reshape(-1)
    if not (len(names) == len(offsets) == len(lengths)):
        raise ValueError(f"打包索引字段长度不一致: {index_path}")
    if (len(set(names)) != len(names)
            or any(not name or Path(name).name != name for name in names)
            or np.any(lengths < 0)
            or (len(names) and (offsets[0] != 0
                or np.any(offsets[1:] != offsets[:-1] + lengths[:-1])
                or offsets[-1] + lengths[-1] != pack_path.stat().st_size))):
        raise ValueError(f"打包索引文件名或 offset/length 无效: {index_path}")
    packed = (np.memmap(pack_path, mode="r", dtype=np.uint8)
              if pack_path.stat().st_size else np.empty(0, dtype=np.uint8))
    return packed, dict(zip(names, zip(offsets, lengths)))


def data_files(directory, suffix):
    directory = Path(directory)
    packed = packed_directory(directory)
    if packed is not None:
        return sorted(directory / name for name in packed[1] if name.endswith(suffix))
    return sorted(directory.glob(f"*{suffix}"))


def source_file(path):
    from data2datasets.dataset_for_all_task import HPE_Dataset

    path = Path(path)
    if path.is_relative_to(ROOT_PATH):
        candidate = PACKED_DATA_ROOT / path.relative_to(ROOT_PATH)
        if HPE_Dataset._packed_path_exists(candidate):
            return candidate
    return path


def open_data_file(path):
    path = Path(path)
    packed = packed_directory(path.parent)
    if packed is not None and path.name in packed[1]:
        offset, length = packed[1][path.name]
        return io.BytesIO(packed[0][offset:offset + length].tobytes())
    return source_file(path).open("rb")


def load_extrinsics(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_point_cloud(path):
    """读取有效点云；异常或空帧返回统一空数组及跳过原因。"""
    try:
        with open_data_file(path) as source:
            points = np.load(source).astype(np.float32)
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
    candidates = [(abs(value - target), item) for item in data_files(folder, ".pkl")
                  if (value := timestamp(item)) is not None]
    return min(candidates)[1] if candidates else None


def decimal_frame_ticks(frame_count, target_ticks=9):
    """生成个位数均为 0 的帧刻度。"""
    last = frame_count // 10 * 10
    if last == 0:
        return np.array([0])
    step = max(10, int(np.ceil(last / target_ticks / 10)) * 10)
    return np.unique(np.append(np.arange(0, last + 1, step), last))


def draw_human_pose(ax, poses, color, label):
    is_raw = label == "Raw"
    for person_index, pose in enumerate(poses):
        ax.scatter(*pose.T, facecolors="none" if is_raw else color, edgecolors=color,
                   s=65 if is_raw else 25, label=label if person_index == 0 else None)
        for parent, child in DIRECTED_BONES:
            ax.plot(*pose[[parent, child]].T, c=color, linewidth=2,
                    linestyle="--" if is_raw else "-")



def frame_pa_mpjpe(poses, gt):
    """髋中心匹配后逐人做刚性旋转、平移和尺度对齐，计算 PA-MPJPE（mm）。"""
    pred, target, counts = match_poses(poses.astype(np.float64), gt.astype(np.float64))
    errors = []
    for source, reference in zip(pred, target):
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
    return (float(np.mean(errors)) if errors else None), len(errors), *counts


def frame_centered_mpjpe(poses, gt):
    """髋中心匹配并分别减去各自髋中心，只忽略全局平移误差。"""
    return frame_mpjpe(poses, gt, centered=True)


def deployment_pose_reversal(raw, keeper, dt):
    """部署方向判据：肩髋同时反转，或头脚轴反转；退化轴不作证据。"""
    threshold = max(100.0, 360.0 * dt)

    def reversed_axis(a, b, minimum):
        lengths = np.linalg.norm(a), np.linalg.norm(b)
        if min(lengths) < minimum:
            return False
        angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / np.prod(lengths), -1, 1)))
        return angle > threshold

    return (reversed_axis(raw[6] - raw[5], keeper[6] - keeper[5], .1)
            and reversed_axis(raw[12] - raw[11], keeper[12] - keeper[11], .1)
            or reversed_axis(raw[:5].mean(0) - raw[[15, 16]].mean(0),
                             keeper[:5].mean(0) - keeper[[15, 16]].mean(0), .4))


def deployment_pose_records(records, direction_factory=None):
    """因果 keeper 对照：点数门控、反转确认、持续保持，不回填历史。

    复用全量 raw 以便公平对照，但低点数 raw 不进入此分支的状态更新。
    direction_factory 注入附加保持判据；选组默认仅启用躺姿头脚锁。
    """
    states = {}
    params = TEMPORAL_GUARD_PARAMS
    for index, record in enumerate(records):
        if direction_factory is not None:
            record["gt_crop_deployment_direction_state"] = {}
        stamp = Path(record["path"]).stem.split("_")
        timestamp = int(stamp[0]) + int(stamp[1]) * 1e-9
        raw_by_person = dict(zip(record["gt_crop_pose_indices"], record["gt_crop_poses_raw"]))
        people = record.get("gt_crop_track_ids", record["gt_crop_pose_indices"])
        poses, output_ids, statuses = [], [], {}
        for person in people:
            person = int(person)
            state = states.get(person)
            if state is None or state["last"] != index - 1:
                # shortcut: 使用连续 GT 人员索引；接入跟踪数据时换为带 generation 的 ID。
                state = {"keeper": None, "candidate": None, "count": 0,
                         "time": timestamp - .12, "last": index - 1,
                         "direction": direction_factory() if direction_factory is not None else None}
                states[person] = state
            dt = timestamp - state["time"]
            if dt <= 0:
                raise ValueError("Non-increasing point-cloud timestamp")
            raw = raw_by_person.get(person)
            eligible = (record["dynamic_point_counts"].get(person, 0) >= params["min_dynamic"]
                        and record["total_point_counts"].get(person, 0) >= params["min_total"])
            status = "kept" if state["keeper"] is not None else "excluded"
            direction = state["direction"]
            permit = direction.check(raw, record["dynamic_point_counts"].get(person, 0),
                                     record["total_point_counts"].get(person, 0), dt) if direction else True
            if not eligible or raw is None or not np.isfinite(raw).all():
                state["candidate"], state["count"] = None, 0
            elif permit and (state["keeper"] is None or not deployment_pose_reversal(raw, state["keeper"], dt)):
                state["keeper"] = raw.copy()
                state["candidate"], state["count"] = None, 0
                status = "accepted"
            else:
                stable = (state["candidate"] is not None
                          and pose_change_within(raw, state["candidate"], dt, stable=True))
                state["count"] = state["count"] + 1 if stable else 1
                state["candidate"] = raw.copy()
                if state["count"] >= params["confirm_frames"] and permit:
                    state["keeper"] = raw.copy()
                    state["candidate"], state["count"] = None, 0
                    status = "confirmed"
            if direction:
                if status in ("accepted", "confirmed"):
                    direction.commit()
                record["gt_crop_deployment_direction_state"][person] = direction.snapshot()
            state["last"], state["time"] = index, timestamp
            statuses[person] = status
            if state["keeper"] is not None:
                output_ids.append(person)
                poses.append(state["keeper"].copy())
        record["gt_crop_deployment_poses"] = np.asarray(poses, dtype=np.float32).reshape(-1, 17, 3)
        record["gt_crop_deployment_pose_indices"] = np.asarray(output_ids, dtype=int)
        record["gt_crop_deployment_status"] = statuses



def aggregate_mpjpe(frame_stats):
    matched = sum(stats[1] for stats in frame_stats)
    error_sum = sum(error * count for error, count, _, _ in frame_stats if error is not None)
    return {"mpjpe_mm": error_sum / matched if matched else None,
            "matched_people": matched,
            "pred_people": sum(stats[2] for stats in frame_stats),
            "gt_people": sum(stats[3] for stats in frame_stats),
            "matched_frames": sum(stats[1] > 0 for stats in frame_stats)}


def last_frame_errors(records, rotation, translation):
    """按末帧分别统计椭圆内外 MPJPE；相同时间戳、人员槽位只保留首次样本。"""
    stages = {
        "raw": ("poses_raw", "pose_indices"),
        "guarded": ("poses", "pose_track_ids"),
        "held": ("poses", "pose_track_ids"),
        "deployment": ("deployment_poses", "deployment_pose_indices"),
    }
    stats = {region: {stage: {"error_mm": np.full(len(records), np.nan),
                            "count": np.zeros(len(records), dtype=int)}
                      for stage in stages} for region in ("inside", "outside")}
    seen = set()
    for index, record in enumerate(records):
        stamp = Path(record["path"]).stem
        inside = acceptance_mask(record["gt"], rotation, translation)
        fresh = [p for p in range(len(inside)) if (stamp, p) not in seen]
        seen.update((stamp, p) for p in fresh)
        for stage, (suffix, ids_suffix) in stages.items():
            people, rows = np.unique(record["gt_crop_" + ids_suffix], return_index=True)
            selected = np.isin(people, fresh)
            if stage == "held":
                selected &= np.array([record["gt_crop_pose_status"][int(p)] == "held"
                                      for p in people], dtype=bool)
            poses = record["gt_crop_" + suffix][rows[selected]]
            truth = record["gt"][people[selected]]
            for region, mask in (("inside", inside[people[selected]]),
                                 ("outside", ~inside[people[selected]])):
                value, count, *_ = frame_mpjpe(poses[mask], truth[mask])
                if count:
                    stats[region][stage]["error_mm"][index] = value
                    stats[region][stage]["count"][index] = count
    return stats


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


def group_metric_log_lines(metrics):
    rows = {}
    for stage in ("raw", "guarded"):
        all_metrics, inside = [metrics["regions"][region]["current_frame"][stage]
                               for region in ("all", "inside")]
        metres = lambda value: value / 1000 if value is not None else None
        rows[stage] = dict(mpjpe=metres(all_metrics["mpjpe_mm"]), mpjpe_inside=metres(inside["mpjpe_mm"]),
            centered_mpjpe=metres(all_metrics["centered_mpjpe_mm"]),
            centered_mpjpe_inside=metres(inside["centered_mpjpe_mm"]),
            count=all_metrics.get("matched_people"), count_inside=inside.get("matched_people"))
    return format_group_metrics(f"{metrics['date']}/{metrics['group']}", metrics.get("label", "unlabelled"),
                                rows["raw"], rows["guarded"], model=str(MODEL_POSE_PATH))


def append_group_metrics_markdown(metrics, path):
    """保存 Raw/Guard 两行日志；旧 Markdown 行转换单位、顺序后保留。"""
    path = Path(path)
    rows = []
    legacy_rows = False
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        for line in previous.splitlines():
            if "Val group " in line:
                rows.append(line)
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 15:
                cells = cells[:3] + cells[3:5] + cells[7:9] + cells[11:13]
            if not line.startswith("|") or len(cells) != 9:
                continue
            if cells[0].startswith("group_"):
                cells[0], cells[1] = cells[1], cells[0]
            if not cells[0].isdigit():
                continue
            legacy_rows = True
            stages = []
            for offset in (0, 1):
                values = {}
                for key, index in (("mpjpe", 3 + offset), ("centered_mpjpe", 5 + offset)):
                    pair = re.findall(r"N/A|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", cells[index])
                    inside, total = (pair + ["N/A", "N/A"])[:2]
                    values[key] = None if total == "N/A" else float(total) / 1000
                    values[key + "_inside"] = None if inside == "N/A" else float(inside) / 1000
                stages.append(values)
            rows.extend(format_group_metrics(f"{cells[0]}/{cells[1]}", "unlabelled", *stages, model=cells[2]))
        if legacy_rows:
            backup = path.with_name(path.stem + ".before_raw_guard_lines" + path.suffix)
            if not backup.exists():
                backup.write_text(previous, encoding="utf-8")
    key = f"Val group {metrics['date']}/{metrics['group']} |"
    rows = [row for row in rows if key not in row]
    rows.extend(group_metric_log_lines(metrics))
    lines = ["# Group metrics: all / inside ellipse; unit=m; unavailable results=N/A.",
             "# Guard: " + guard_metric_protocol()["description"],
             "# Raw/Guard样本可能不同；Guard拒绝的人从GT匹配集合移除，需结合共同子集和覆盖率比较。", "", *rows]
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
        self.paths = data_files(group_path / "dpct高位机/PC", ".npy")
        if not self.paths:
            raise FileNotFoundError(f"没有点云文件：{group_path / 'dpct高位机/PC'}")
        print(f"训练配置：{TRAIN_CONFIG_PATH}；T={T}，单人点数={TRAIN_MAX_POINTS}，acc_frame={TRAIN_CONFIG['data'].get('acc_frame', 0)}")
        self.pose_model = load_models(self.device)
        self.gt_folder = group_path / "camera results/smoothed 3D"
        calib_path = ROOT_PATH / DATE / "calib"
        extrinsic_path = source_file(calib_path / "extrinsic_img_to_radar_high.npz")
        self.extrinsics = None
        if show_gt and data_files(self.gt_folder, ".pkl") and extrinsic_path.exists():
            self.extrinsics = load_extrinsics(extrinsic_path)
        elif show_gt:
            print("未找到相机 GT 或外参，继续推理，MPJPE 将为 N/A。")
        low_extrinsic_path = source_file(calib_path / "extrinsic_img_to_radar_low.npz")
        if not extrinsic_path.exists() or not low_extrinsic_path.exists():
            raise FileNotFoundError("指标验收与重力坐标系绘图需要高、低位机外参文件")
        high = load_extrinsics(extrinsic_path)
        low = load_extrinsics(low_extrinsic_path)
        rotation = np.asarray(low["R_est"]) @ np.asarray(high["R_est"]).T
        translation = (np.asarray(low["t_est"]).reshape(3)
                       - rotation @ np.asarray(high["t_est"]).reshape(3))
        self.high_to_low = (rotation.astype(np.float32), translation.astype(np.float32))
        self.records = []
        self.current_index = 0
        self.group_metrics = None
        self.raw_training_ready = False
        self.fig = None
        self.frame_slider = None
        self.defer_visual_guarding = False
        self.last_frame_metrics = None
        self.error_axes = None
        if not visualize:
            return
        self.fig = plt.figure(figsize=(20, 10), dpi=80)
        grid = self.fig.add_gridspec(2, 3, height_ratios=[3, 2])
        self.axes = np.asarray([self.fig.add_subplot(grid[0, idx], projection="3d")
                                for idx in range(3)]).reshape(1, 3)
        self.error_axes = np.array([self.fig.add_subplot(grid[1, :])])
        self.fig.subplots_adjust(bottom=.11, top=.89, hspace=.30, wspace=.08)
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
        record = {"path": str(path)}
        self.records.append(record)
        self.visualization_mpjpe = None
        gt, has_gt = self.load_gt(record)
        gt_boxes = []
        dynamic_micro_ratios = {}
        dynamic_point_counts = {}
        total_point_counts = {}
        gt_selected_mask = np.zeros(len(points), dtype=bool)
        gt_track_ids = np.arange(len(gt), dtype=int)
        for person_index, person_gt in enumerate(gt):
            bbox = np.concatenate((person_gt.min(axis=0) - GT_BBOX_MARGIN,
                                   person_gt.max(axis=0) + GT_BBOX_MARGIN))
            gt_boxes.append(bbox)
            gt_inside_original = ((points[:, :3] >= bbox[:3]) & (points[:, :3] <= bbox[3:])).all(1)
            gt_selected_mask |= gt_inside_original
            tags = points[gt_inside_original, -1]
            dynamic_count = int(np.isclose(tags, DYNAMIC_POINT_TAG).sum())
            micro_count = int(np.isin(tags, np.arange(2, 8)).sum())
            dynamic_point_counts[person_index] = dynamic_count
            total_point_counts[person_index] = len(tags)
            dynamic_micro_ratios[person_index] = (
                dynamic_count / micro_count if micro_count else
                (np.inf if dynamic_count else 0.0)
            )
        record.update(dynamic_micro_ratios=dynamic_micro_ratios,
                      dynamic_point_counts=dynamic_point_counts,
                      total_point_counts=total_point_counts,
                      gt_crop_track_ids=gt_track_ids,
                      gt_crop_pose_indices=np.empty(0, dtype=int),
                      gt_crop_poses_raw=np.empty((0, 17, 3), dtype=np.float32),
                      gt_crop_quality={},
                      gt_boxes=np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 6),
                      gt_selected_mask=gt_selected_mask)
        if self.visualize and not self.defer_visual_guarding:
            self.postprocess_poses()
        return record, points, gt_selected_mask

    def postprocess_poses(self):
        """时序回填完成后统一判决，标签顺序分别对齐 raw/guarded 人员索引。"""
        direction_factory = lambda: DirectionKeeper(*self.high_to_low, direction_check=False)
        for record in self.records:
            for key in list(record):
                if key.startswith(("gt_crop_keeper_base_", "gt_crop_motion_guard_")) or key == "gt_crop_poses_legacy":
                    del record[key]
        guard_pose_records(self.records, "gt_crop_", direction_factory=direction_factory)
        deployment_pose_records(self.records, direction_factory)
        self.last_frame_metrics = None
        for record in self.records:
            for suffix in ("_raw", ""):
                poses_low = transform_points(record[f"gt_crop_poses{suffix}"], *self.high_to_low)
                result = classify_actions(poses_low)
                record[f"gt_crop_actions{suffix}"] = result.labels
                record[f"gt_crop_action_valid{suffix}"] = result.valid

    def load_gt(self, record):
        if "gt" not in record:
            record["gt"] = np.empty((0, 17, 3), dtype=np.float32)
            record["has_gt"] = False
            if self.extrinsics is not None:
                pose_file = closest_timestamp_file(Path(record["path"]), self.gt_folder)
                if pose_file is not None:
                    record["gt_pose_path"] = str(pose_file)
                    with open_data_file(pose_file) as source:
                        gt = transform_points(np.asarray(pickle.load(source)),
                                              self.extrinsics["R_est"],
                                              np.asarray(self.extrinsics["t_est"]).reshape(3))
                    record["gt"] = np.asarray(gt).reshape(-1, 17, 3)
                    record["has_gt"] = True
        return record["gt"], record["has_gt"]

    @torch.inference_mode()
    def infer_gt_raw_as_training(self):
        """按实验的点云积累、裁剪和重采样流程重算完整滑窗结果。"""
        if self.raw_training_ready:
            return
        from data2datasets.dataset_for_all_task import HPE_Dataset, collate_fn
        from run.utils.process_one_epoch import (
            adapt_coordinate_frame, prepare_person_pointcloud, validate_coordinate_frame,
            validate_center_xy_in_low,
        )
        training_data = dict(TRAIN_CONFIG["data"])
        training_data.setdefault("person_max_points", 200)
        training_task = dict(TRAIN_CONFIG["task"])
        training_task.setdefault("coordinate", "low" if training_task.get("center_xy_in_low", False)
                                 else training_task.get("coordinate_frame", "high"))
        validate_coordinate_frame(training_task, training_data)
        validate_center_xy_in_low(training_task, training_data)
        if training_task["input"] != "radar_high_pc" or training_task["output"] != "gt_for_high":
            raise ValueError("此脚本要求 radar_high_pc 输入和 gt_for_high 输出")

        for record in self.records:
            record["gt_crop_poses_raw"] = np.empty((0, 17, 3), dtype=np.float32)
            record["gt_crop_pose_indices"] = np.empty(0, dtype=int)
            record["gt_crop_input_points"] = {}
            record["gt_crop_quality"] = {}
            record["gt_training_window"] = False
        self.last_frame_metrics = None

        point_sequences = []
        gt_sequences = []
        for record in tqdm(self.records, desc="Loading training point clouds"):
            points, _ = load_point_cloud(record["path"])
            point_sequences.append(points)
            gt_sequences.append(self.load_gt(record)[0])
        point_sequences = HPE_Dataset._accumulate_pointcloud_sequence(
            point_sequences, 0, training_data.get("acc_frame", 0)
        )

        for start in tqdm(range(max(0, len(self.records) - T + 1)),
                          desc="Training-window pose inference"):
            window = self.records[start:start + T]
            samples = collate_fn(
                [{"radar_high_pc": point_sequences[start:start + T],
                  "gt_for_high": gt_sequences[start:start + T]}],
                max_people=training_data["max_people"],
            )
            rotation, translation = self.high_to_low
            samples["high_to_low_R"] = torch.from_numpy(rotation.copy())[None, None].expand(1, T, -1, -1)
            samples["high_to_low_t"] = torch.from_numpy(translation.copy())[None, None].expand(1, T, -1)
            samples = adapt_coordinate_frame(samples, training_task, training_data, 'val')
            model_input, gt_data, valid_instances, cropped_points, cropped_mask, pointcloud_center = prepare_person_pointcloud(
                samples, training_task, training_data, self.device, 'val')
            # 缺 GT 或裁剪不足 10 点均打断窗口，不能拆成短片段推理。
            complete = model_input["mask"].any(dim=-1).all(dim=-1)
            valid_people = valid_instances[0].clone()
            valid_people[valid_instances[0]] = complete
            if not valid_people.any():
                continue
            person_mask = gt_data["mask"].permute(0, 2, 1)
            raw_cropped_points = cropped_points.clone()
            if training_task.get("center_xy_in_low", False) or training_task.get("center_on_pointcloud", False):
                raw_cropped_points[..., :3] += pointcloud_center.permute(0, 2, 1, 3).unsqueeze(-2)
            if training_task["coordinate"] == "low":
                raw_cropped_points[..., :3] = (
                    raw_cropped_points[..., :3] - torch.as_tensor(translation, device=self.device)
                ) @ torch.as_tensor(rotation, device=self.device)
            model_input = {key: value[complete] for key, value in model_input.items()}
            prediction = self.pose_model(model_input)["pose"][:, :, 0]
            prediction += pointcloud_center[0][:, valid_people].permute(1, 0, 2)[:, :, None]
            prediction = prediction.cpu().numpy()
            if training_task["coordinate"] == "low":
                prediction = (prediction - translation) @ rotation
            people = torch.nonzero(valid_people, as_tuple=True)[0].cpu().numpy()
            present = person_mask[0, valid_people, POSE_OUTPUT_POSITION].cpu().numpy()
            gt_present = present.copy()
            record = window[POSE_OUTPUT_POSITION]
            record["gt_crop_poses_raw"] = prediction[present, POSE_OUTPUT_POSITION]
            record["gt_crop_pose_indices"] = people[present]
            record["gt_crop_quality"] = {}
            record["gt_crop_input_points"] = {
                int(person): raw_cropped_points[0, person, POSE_OUTPUT_POSITION][
                    cropped_mask[0, person, POSE_OUTPUT_POSITION]
                ].cpu().numpy()
                for person in people[gt_present]
            }
            record["gt_training_window"] = True
        self.raw_training_ready = True
        self.visualization_mpjpe = None

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
                "dynamic_micro_ratios": record["dynamic_micro_ratios"],
                "dynamic_point_counts": record["dynamic_point_counts"],
                "total_point_counts": record["total_point_counts"],
                "current_track_ids": record["gt_crop_track_ids"],
                "current_pose_indices": record["gt_crop_pose_indices"],
                "current_poses_raw": record["gt_crop_poses_raw"],
                "current_input_points": record["gt_crop_input_points"],
                "current_quality": record["gt_crop_quality"],
                "current_training_window": record["gt_training_window"],
            })
        cache = {
            "version": 2,
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

    def metric_inputs(self, record, stage, inside_only=False):
        """拒绝输出的人员同时从 GT 子集中移除，防止多人匹配串人。"""
        prefix = "gt_crop_"
        suffix, ids_suffix = {
            "raw": ("poses_raw", "pose_indices"),
            "raw_kept": ("poses_raw", "pose_indices"),
            "guarded": ("poses", "pose_track_ids"),
            "deployment": ("deployment_poses", "deployment_pose_indices"),
        }[stage]
        gt = record["gt"]
        ids = np.arange(len(gt))
        if inside_only:
            ids = ids[acceptance_mask(gt, *self.high_to_low)]
        if stage in ("raw_kept", "guarded"):
            ids = ids[np.isin(ids, record[prefix + "pose_track_ids"])]
        if stage == "deployment":
            ids = ids[np.isin(ids, record[prefix + "deployment_pose_indices"])]
        poses = record[prefix + suffix]
        poses = poses[np.isin(record[prefix + ids_suffix], ids)]
        if not record.get("gt_training_window", True):
            gt = gt[:0]
        else:
            gt = gt[ids]
        return poses, gt

    def evaluate_group(self):
        """整组统计 Raw、Guard+躺姿锁、Keeper+躺姿锁及输出覆盖率。"""
        if len(self.records) != len(self.paths):
            raise ValueError("整组尚未推理完成，不能报告整组 MPJPE")
        if self.group_metrics is not None:
            return self.group_metrics
        self.infer_gt_raw_as_training()
        self.postprocess_poses()
        self.visualization_mpjpe = None
        regions = {}
        stages = ("raw", "raw_kept", "guarded", "deployment")
        for region, inside_only in (("inside", True), ("all", False)):
            result = {}
            for stage in stages:
                values = {}
                for name, metric in (("mpjpe_mm", frame_mpjpe),
                                     ("centered_mpjpe_mm", frame_centered_mpjpe),
                                     ("pa_mpjpe_mm", frame_pa_mpjpe)):
                    stats = aggregate_mpjpe([
                        metric(*self.metric_inputs(record, stage, inside_only))
                        for record in self.records
                    ])
                    values[name] = stats["mpjpe_mm"]
                    if name == "mpjpe_mm":
                        values["matched_people"] = stats["matched_people"]
                result[stage] = values
            eligible = sum(len(self.metric_inputs(record, "raw", inside_only)[1]) for record in self.records)
            kept = result["guarded"]["matched_people"]
            result["coverage"] = {
                "eligible_people": eligible, "kept_people": kept,
                "excluded_people": eligible - kept,
                "retained_pct": 100 * kept / eligible if eligible else None,
            }
            keeper_count = result["deployment"]["matched_people"]
            result["keeper_coverage"] = {
                "eligible_people": eligible, "kept_people": keeper_count,
                "excluded_people": eligible - keeper_count,
                "retained_pct": 100 * keeper_count / eligible if eligible else None,
            }
            regions[region] = {"current_frame": result}
        self.group_metrics = {
            "date": DATE, "group": GROUP, "frames": len(self.records),
            "gt_frames": sum(record["has_gt"] for record in self.records),
            "guard": {**guard_metric_protocol(), "lying_head_lock": True, "direction_check": False},
            "keeper": {"lying_head_lock": True, "direction_check": False, "offline_backfill": False},
            "label": json.loads((PROJECT_ROOT / "data2datasets/group_labels.json").read_text())["groups"].get(
                f"{DATE}/{GROUP}", {}).get("label", "unlabelled"),
            "regions": regions,
        }
        print("Guard protocol: " + self.group_metrics["guard"]["description"])
        print("Guard / Keeper 均启用躺姿头脚锁；新增双方向及 80% 动点确认关闭。")
        print("Scoring: Raw/Guard/Keeper may have different output sets; "
              "compare matched counts and coverage. raw_kept restricts Raw to Guard output IDs.")
        fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
        print(f"{DATE}/{GROUP}: checkpoint=last.pth；离线三帧确认/回填")
        for line in group_metric_log_lines(self.group_metrics):
            print(line)
        for region, label in (("inside", "椭圆内"), ("all", "全部")):
            keeper_metrics = regions[region]["current_frame"]["deployment"]
            keeper_coverage = regions[region]["current_frame"]["keeper_coverage"]
            print(f"{label} Keeper+躺姿锁 MPJPE：{fmt(keeper_metrics['mpjpe_mm'])} mm；"
                  f"保留率：{fmt(keeper_coverage['retained_pct'])}%")
            c = regions[region]["current_frame"]["coverage"]
            print(f"{label}保留率：{fmt(c['retained_pct'])}% "
                  f"({c['kept_people']}/{c['eligible_people']})；不输出 {c['excluded_people']} 人帧")
        return self.group_metrics

    def save_metric_timeseries(self):
        """按人员保存椭圆内当前帧 GT 裁剪的 raw/guarded MPJPE。"""
        frame_indices = np.arange(1, len(self.records) + 1)
        frame_ticks = decimal_frame_ticks(len(self.records))
        output_dir = METRIC_VIS_ROOT / DATE / GROUP
        output_dir.mkdir(parents=True, exist_ok=True)
        max_people = max((len(record["gt"]) for record in self.records), default=0)
        stage_values = {}
        for stage in ("raw", "guarded"):
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
            packed = packed_directory(action_dir)
            action_path = next((path for path in candidates
                                if (packed is not None and path.name in packed[1])
                                or source_file(path).is_file()), None)
            if action_path is None:
                continue
            if action_path.suffix == ".npz":
                with open_data_file(action_path) as source, np.load(source) as data:
                    action_data = data["labels"]
            else:
                with open_data_file(action_path) as source:
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
        for stage, title in (("raw", "Raw"), ("guarded", "Guarded")):
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
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        self.plot_last_frame_errors([ax])
        output_path = output_dir / "deployment_comparison.png"
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
        """保存椭圆内 current raw/guarded MPJPE 均超阈值的最差若干帧。"""
        output_dir = ERROR_VIS_ROOT / DATE / GROUP
        selected = []
        for index, record in enumerate(self.records):
            current_raw_error = frame_mpjpe(*self.metric_inputs(record, "raw", True))[0]
            current_guarded_error = frame_mpjpe(*self.metric_inputs(record, "guarded", True))[0]
            if all(value is not None and value > ERROR_VIS_THRESHOLD_MM
                   for value in (current_raw_error, current_guarded_error)):
                selected.append((index, current_raw_error, current_guarded_error))
        selected.sort(key=lambda item: item[2], reverse=True)
        selected = selected[:ERROR_VIS_MAX_FRAMES]
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_png in output_dir.glob("*.png"):
            old_png.unlink()
        if not selected:
            print(f"{DATE}/{GROUP}: 没有椭圆内 current raw 和 guarded MPJPE 均超过 "
                  f"{ERROR_VIS_THRESHOLD_MM:.0f} mm 的帧")
            return 0

        self.fig = plt.figure(figsize=(20, 6), dpi=100, constrained_layout=True)
        self.axes = np.asarray([self.fig.add_subplot(1, 3, idx + 1, projection="3d")
                                for idx in range(3)]).reshape(1, 3)
        for index, current_raw_error, current_guarded_error in selected:
            fmt = lambda value: "N-A" if value is None else f"{value:.1f}mm"
            note = (f"ellipse-only  current_raw={fmt(current_raw_error)}  "
                    f"current_guarded={fmt(current_guarded_error)}")
            self.show_frame(index, note=note)
            frame_name = Path(self.records[index]["path"]).stem
            filename = (f"{DATE}_{GROUP}_frame_{index + 1:06d}_{frame_name}_ellipse-only_"
                        f"current_raw_{fmt(current_raw_error)}_"
                        f"current_guarded_{fmt(current_guarded_error)}.png")
            self.fig.savefig(output_dir / filename, dpi=100)
        plt.close(self.fig)
        self.fig = None
        print(f"{DATE}/{GROUP}: 已保存 {len(selected)} 帧到 {output_dir}")
        return len(selected)

    def get_visualization_mpjpe(self):
        """按整组匹配人次加权，缓存标题指标，回看时无需重复统计。"""
        if getattr(self, "visualization_mpjpe", None) is None:
            self.visualization_mpjpe = {}
            for inside_only in (True, False):
                averages = {}
                for name, metric in (("mpjpe", frame_mpjpe),
                                     ("centered", frame_centered_mpjpe)):
                    averages[name] = []
                    for stage in ("raw", "guarded"):
                        stats = [metric(*self.metric_inputs(record, stage, inside_only))
                                 for record in self.records]
                        averages[name].append(aggregate_mpjpe(stats)["mpjpe_mm"])
                self.visualization_mpjpe[inside_only] = averages
        return self.visualization_mpjpe

    def plot_last_frame_errors(self, axes=None):
        if self.last_frame_metrics is None:
            self.last_frame_metrics = last_frame_errors(self.records, *self.high_to_low)
        frames = np.arange(1, len(self.records) + 1)
        for ax in self.error_axes if axes is None else axes:
            ax.clear()
            for stage, label, color in (("raw", "Raw", "tab:blue"),
                                       ("guarded", "Guard + lie lock", "tab:orange"),
                                       ("deployment", "Keeper + lie lock", "tab:brown")):
                if stage not in self.last_frame_metrics["inside"]:
                    continue
                for region, style in (("inside", "-"), ("outside", "--")):
                    ax.plot(frames, self.last_frame_metrics[region][stage]["error_mm"],
                            linestyle=style, color=color, label=f"{label} ({region})", linewidth=1.1)
            ax.axhline(METRIC_REFERENCE_MM, color="gray", linestyle="--", linewidth=1)
            ax.set(title="Last-frame MPJPE: Raw / Guard+lie lock / Keeper+lie lock", ylabel="MPJPE (mm)",
                   xlabel="Frame idx (window last frame); solid: inside ellipse; dashed: outside ellipse",
                   xticks=decimal_frame_ticks(len(self.records)), xlim=(0, max(1, len(self.records))))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(ncol=2, fontsize=8)

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
        visible_boxes = record["gt_boxes"]
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

        def draw_points(ax, cloud):
            dynamic = np.isclose(cloud[:, -1], DYNAMIC_POINT_TAG)
            static = ~dynamic
            ax.scatter(*cloud[dynamic, :3].T, c="red", s=5)
            ax.scatter(*cloud[static, :3].T, c="blue", s=5)
            ax.legend(handles=[
                Line2D([], [], color="red", marker=".", linestyle="None", label=f"Selected dynamic (tag=1): {int(dynamic.sum())}"),
                Line2D([], [], color="blue", marker=".", linestyle="None", label=f"Selected static (tag=2..7): {int(static.sum())}"),
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

        draw_points(self.axes[0, 0], plot_points[visible_point_mask])
        draw_boxes(self.axes[0, 0])
        self.axes[0, 0].set_title(
            f"Current frame GT crop: before accumulation / {TRAIN_MAX_POINTS}-point resampling")
        def metric_label(name, poses, target, average):
            values = (frame_mpjpe(poses, target)[0],
                      frame_centered_mpjpe(poses, target)[0],
                      frame_pa_mpjpe(poses, target)[0])
            formatted = ["N/A" if value is None else f"{value:.1f}" for value in values]
            mean = "N/A" if average is None else f"{average:.1f}"
            return (f"{name}: frame MPJPE/Centered/PA={'/'.join(formatted)} mm\n"
                    f"Group mean MPJPE={mean} mm")

        for ax, inside_only, title in (
            (self.axes[0, 1], True, "Training pose: inside ellipse"),
            (self.axes[0, 2], False, "Training pose: all people"),
        ):
            raw = record["gt_crop_poses_raw"]
            raw_people = record["gt_crop_pose_indices"]
            raw_actions = record["gt_crop_actions_raw"]
            raw_action_valid = record["gt_crop_action_valid_raw"]
            guarded = record["gt_crop_poses"]
            people = record["gt_crop_pose_track_ids"]
            actions = record["gt_crop_actions"]
            action_valid = record["gt_crop_action_valid"]
            target = gt
            if inside_only:
                raw_selected = np.isin(raw_people, inside_ids)
                raw, raw_people = raw[raw_selected], raw_people[raw_selected]
                raw_actions, raw_action_valid = raw_actions[raw_selected], raw_action_valid[raw_selected]
                selected = np.isin(people, inside_ids)
                guarded, people = guarded[selected], people[selected]
                actions, action_valid = actions[selected], action_valid[selected]
                target = metric_gt
            target_poses = target if has_gt else target[:0]
            target_actions = classify_actions(transform_points(target_poses, rotation, translation))
            if has_gt:
                draw_human_pose(ax, transform_points(target, rotation, translation), "green", "Camera GT")
            draw_human_pose(ax, transform_points(guarded, rotation, translation), "magenta", "Guarded")
            # Raw 后画为空心关节和虚线，重合时仍能看见两条支路。
            draw_human_pose(ax, transform_points(raw, rotation, translation), "orange", "Raw")
            for source, poses, ids, labels, valid_mask, z_offset in (
                ("GT", target_poses, inside_ids if inside_only else np.arange(len(gt)),
                 target_actions.labels, target_actions.valid, .24),
                ("Guarded", guarded, people, actions, action_valid, .08),
                ("Raw", raw, raw_people, raw_actions, raw_action_valid, -.08),
            ):
                for pose, person, action, valid in zip(
                    transform_points(poses, rotation, translation), ids, labels, valid_mask
                ):
                    if valid:
                        class_index = int(action.argmax())
                        position = pose[[11, 12]].mean(axis=0) + [0, 0, z_offset]
                        ax.text(*position, f"P{person} {source}: {ACTION_LABELS[class_index]}",
                                color=ACTION_COLORS[class_index], fontsize=9)
            metrics = self.get_visualization_mpjpe()[inside_only]
            averages = metrics["mpjpe"]
            formatted = ["N/A" if value is None else f"{value:.1f}" for value in averages]
            centered = ["N/A" if value is None else f"{value:.1f}" for value in metrics["centered"]]
            ax.set_title(f"{title}\nGroup mean MPJPE: Raw={formatted[0]}, Guarded={formatted[1]} mm\n"
                         f"Group mean Centered MPJPE: Raw={centered[0]}, Guarded={centered[1]} mm")
            ax.legend(handles=[
                Line2D([], [], color="green", marker="o", label="Camera GT"),
                Line2D([], [], color="orange", marker="o", label=metric_label("Raw", raw, target, averages[0])),
                Line2D([], [], color="magenta", marker="o", label=metric_label("Temporal", *self.metric_inputs(record, "guarded", inside_only), averages[1])),
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
        """WebAgg 启动前完成整组末位推理，最后统一执行离线时序判决。"""
        self.defer_visual_guarding = True
        try:
            for index in tqdm(range(len(self.records), len(self.paths)),
                              desc="Loading WebAgg frames"):
                self.infer_frame(index)
        finally:
            self.defer_visual_guarding = False
        self.infer_gt_raw_as_training()
        print("推理完成，正在执行离线骨架时序判决……", flush=True)
        self.postprocess_poses()
        self.plot_last_frame_errors()
        print("时序判决完成，正在准备 WebAgg……", flush=True)

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
    """无需权重/GPU 的指标、时序处理、行为判决和绘图自检。"""
    standing = np.zeros((17, 3), dtype=np.float32)
    standing[:5, 2] = 1.7
    standing[[5, 6], 2] = 1.4
    standing[[7, 8], 2] = 1.15
    standing[[9, 10, 11, 12], 2] = .9
    standing[[13, 14], 2] = .45
    standing[1::2, 1] = -.1
    standing[2::2, 1] = .1
    lying = standing[:, [2, 1, 0]]
    action_app = SelectedGroupVisualizer.__new__(SelectedGroupVisualizer)
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    translation = np.array([2., -1., .5])
    action_app.high_to_low = (rotation, translation)
    low_poses = np.stack((standing, lying, np.full_like(standing, np.nan)))
    action_app.records = []
    for i, low in enumerate((low_poses, lying[None], low_poses[:0])):
        people = np.array(([4, 9, 12], [4], [])[i], dtype=int)
        action_app.records.append(dict(
            path=f"100_{i * 100000000:09d}.npy",
            gt_crop_poses_raw=(low - translation) @ rotation,
            gt_crop_pose_indices=people,
            dynamic_point_counts={int(p): 100 if i == 0 else 0 for p in people},
            total_point_counts={int(p): 100 for p in people},
        ))
    action_app.postprocess_poses()
    first, held, empty = action_app.records
    np.testing.assert_array_equal(first["gt_crop_actions_raw"].argmax(axis=-1), [0, 2, 3])
    np.testing.assert_array_equal(first["gt_crop_action_valid_raw"], [True, True, False])
    np.testing.assert_array_equal(first["gt_crop_pose_track_ids"], [4, 9])
    np.testing.assert_array_equal(first["gt_crop_actions"].argmax(axis=-1), [0, 2])
    assert held["gt_crop_actions_raw"][0].argmax() == 2
    assert held["gt_crop_actions"][0].argmax() == 0
    assert empty["gt_crop_actions"].shape == empty["gt_crop_actions_raw"].shape == (0, 4)
    assert empty["gt_crop_action_valid"].shape == (0,)
    gt = np.zeros((1, 17, 3), dtype=np.float32)
    pred = gt.copy()
    pred[..., 1] = .1
    np.testing.assert_allclose(frame_mpjpe(pred, gt)[0], 100, atol=1e-4)
    np.testing.assert_allclose(frame_centered_mpjpe(pred, gt)[0], 0, atol=1e-4)
    invalid = np.full_like(gt, np.nan)
    shaped = np.arange(51, dtype=np.float32).reshape(1, 17, 3) / 100
    for metric in (frame_mpjpe, frame_centered_mpjpe, frame_pa_mpjpe):
        assert metric(gt[:0], gt) == (None, 0, 0, 1)
        assert metric(gt, gt[:0]) == (None, 0, 1, 0)
        assert metric(invalid, gt) == (None, 0, 0, 1)
        stats = metric(np.concatenate((shaped, invalid)), np.concatenate((shaped, shaped + 10)))
        assert stats[1:] == (1, 1, 2)
    metric_app = SelectedGroupVisualizer.__new__(SelectedGroupVisualizer)
    metric_app.records = [{}]
    metric_app.metric_inputs = lambda record, stage, inside: (pred, gt)
    for averages in metric_app.get_visualization_mpjpe().values():
        np.testing.assert_allclose(averages["mpjpe"], [100, 100], atol=1e-4)
        np.testing.assert_allclose(averages["centered"], [0, 0], atol=1e-4)
    pose = np.zeros((17, 3), dtype=np.float32)
    axis_pose = np.zeros((17, 3), dtype=np.float32)
    axis_pose[0, 2] = 1.
    axis_pose[[15, 16], 2] = -1.
    # 低点数不能用稳定预测解锁；无 raw 也持续保持，恢复后才确认新姿态。
    params = TEMPORAL_GUARD_PARAMS
    enough_dynamic, enough_total = params["min_dynamic"], params["min_total"]
    guarded_records = []
    observations = [
        (axis_pose, 0, 0),  # 没有可信历史不能凭空保持。
        (axis_pose, enough_dynamic, enough_total),
        *[(axis_pose * .05, enough_dynamic - 1, enough_total)] * 5,
        (axis_pose * .05, enough_dynamic, enough_total - 1),
        (None, 0, 0),
        *[(axis_pose + 1., enough_dynamic, enough_total)] * params["confirm_frames"],
    ]
    for i, (raw, dynamic, total) in enumerate(observations):
        guarded_records.append(dict(
            path=f"100_{i * 100000000:09d}.npy",
            gt_crop_poses_raw=(raw[None].copy() if raw is not None
                               else np.empty((0, 17, 3), dtype=np.float32)),
            gt_crop_pose_indices=np.array([0] if raw is not None else [], dtype=int),
            gt_crop_track_ids=np.array([0]),
            dynamic_point_counts={0: dynamic}, total_point_counts={0: total},
        ))
    guard_pose_records(guarded_records, "gt_crop_")
    assert guarded_records[0]["gt_crop_pose_status"][0] == "excluded"
    for record in guarded_records[2:9]:
        assert record["gt_crop_pose_status"][0] == "held"
        np.testing.assert_array_equal(record["gt_crop_poses"][0], axis_pose)
    for record in guarded_records[9:]:
        assert record["gt_crop_pose_status"][0] == "confirmed"
        np.testing.assert_array_equal(record["gt_crop_poses"][0], axis_pose + 1.)
    # 轨迹消失后不能将旧骨架传给重新出现的同索引人体。
    guarded_records[8]["gt_crop_track_ids"] = np.empty(0, dtype=int)
    guarded_records[9]["dynamic_point_counts"][0] = 0
    guard_pose_records(guarded_records, "gt_crop_")
    assert not len(guarded_records[8]["gt_crop_poses"])
    assert not len(guarded_records[9]["gt_crop_poses"])
    # 重合时 Raw 保留橙色空心关节和虚线。
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    draw_human_pose(ax, pose[None], "magenta", "Guarded")
    draw_human_pose(ax, pose[None], "orange", "Raw")
    assert len(ax.collections) == 2 and len(ax.lines) == 2 * len(DIRECTED_BONES)
    assert ax.lines[-1].get_linestyle() == "--"
    assert len(ax.collections[-1].get_facecolors()) == 0
    plt.close(fig)
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "points.npy"
        sparse_dynamic = np.zeros((202, 6), dtype=np.float32)
        sparse_dynamic[:, 0] = 2
        sparse_dynamic[-1, 0] = 5
        sparse_dynamic[:, -1] = DYNAMIC_POINT_TAG
        static_point = np.array([[5, 0, 0, 0, 0, 2]], dtype=np.float32)
        outside_point = np.array([[2, 2, 0, 0, 0, DYNAMIC_POINT_TAG]], dtype=np.float32)
        np.save(path, np.concatenate((sparse_dynamic, static_point, outside_point)))
        app = SelectedGroupVisualizer.__new__(SelectedGroupVisualizer)
        app.visualize = True
        app.paths = [path]
        app.high_to_low = (np.eye(3), np.zeros(3))
        gt = np.zeros((2, 17, 3), dtype=np.float32)
        gt[..., :2] = ACCEPTANCE_CENTER_XY
        gt[1, :, 0] = 5  # 椭圆外人体的点云与框也必须显示。
        empty_poses = np.empty((0, 17, 3), dtype=np.float32)
        record = dict(path=path, gt=gt, has_gt=True,
                      gt_boxes=np.array([[0, -1, -1, 4, 1, 1], [4, -1, -1, 6, 1, 1]]),
                      gt_crop_poses_raw=empty_poses, gt_crop_poses=empty_poses,
                      gt_crop_pose_indices=np.empty(0, dtype=int),
                      gt_crop_pose_track_ids=np.empty(0, dtype=int),
                      gt_crop_actions_raw=np.empty((0, len(ACTION_LABELS)), dtype=np.float32),
                      gt_crop_action_valid_raw=np.empty(0, dtype=bool),
                      gt_crop_actions=np.empty((0, len(ACTION_LABELS)), dtype=np.float32),
                      gt_crop_action_valid=np.empty(0, dtype=bool))
        app.records = [record]
        app.get_visualization_mpjpe = lambda: {
            inside: {"mpjpe": [None, None], "centered": [None, None]}
            for inside in (True, False)
        }
        app.fig = plt.figure()
        app.axes = np.array([app.fig.add_subplot(1, 3, i + 1, projection="3d")
                             for i in range(3)]).reshape(1, 3)
        try:
            for clouds in (
                {},
                {0: sparse_dynamic[:0]},
                {0: np.tile(sparse_dynamic, (4, 1))},
            ):
                record["gt_crop_input_points"] = clouds
                app.show_frame(0)
                for pose_ax in app.axes[0, 1:]:
                    assert "Group mean Centered MPJPE: Raw=N/A, Guarded=N/A mm" in pose_ax.get_title()
                ax = app.axes[0, 0]
                assert ax.get_title().startswith("Current frame GT crop: before accumulation")
                labels = [text.get_text() for text in ax.get_legend().get_texts()]
                assert labels == ["Selected dynamic (tag=1): 202",
                                  "Selected static (tag=2..7): 1"]
                assert sum(len(c._offsets3d[0]) for c in ax.collections
                           if hasattr(c, "_offsets3d")) == 203
                assert len(ax.lines) == 24  # 两个人体框各 12 条边。
                np.testing.assert_allclose(np.array(ax.collections[1]._offsets3d).T, sparse_dynamic[:, :3])
                np.testing.assert_allclose(np.array(ax.collections[2]._offsets3d).T, static_point[:, :3])
            record["gt_crop_poses"] = np.stack((standing + [2, 0, 0], lying + [8, 0, 0]))
            record["gt_crop_pose_track_ids"] = np.array([0, 7])
            result = classify_actions(record["gt_crop_poses"])
            record["gt_crop_actions"], record["gt_crop_action_valid"] = result.labels, result.valid
            record["gt_crop_poses_raw"] = np.stack((standing + [8, 0, 0], lying + [2, 0, 0],
                                                     np.full_like(standing, np.nan)))
            record["gt_crop_pose_indices"] = np.array([7, 0, 9])
            result = classify_actions(record["gt_crop_poses_raw"])
            record["gt_crop_actions_raw"], record["gt_crop_action_valid_raw"] = result.labels, result.valid
            record["gt"] = np.stack((lying + [2, 0, 0], standing + [8, 0, 0],
                                     np.full_like(standing, np.nan)))
            record["gt_boxes"] = np.repeat(record["gt_boxes"][:1], 3, axis=0)
            app.show_frame(0)
            assert [text.get_text() for text in app.axes[0, 1].texts] == [
                "P0 GT: lie", "P0 Guarded: stand", "P0 Raw: lie"]
            assert [text.get_text() for text in app.axes[0, 2].texts] == [
                "P0 GT: lie", "P1 GT: stand",
                "P0 Guarded: stand", "P7 Guarded: lie", "P7 Raw: stand", "P0 Raw: lie"]
            record.update(gt=empty_poses, has_gt=False, gt_boxes=np.empty((0, 6)))
            app.show_frame(0)
            assert sum(len(c._offsets3d[0]) for c in app.axes[0, 0].collections
                       if hasattr(c, "_offsets3d")) == 0
            assert not app.axes[0, 1].texts
            assert [text.get_text() for text in app.axes[0, 2].texts] == [
                "P0 Guarded: stand", "P7 Guarded: lie", "P7 Raw: stand", "P0 Raw: lie"]
        finally:
            plt.close(app.fig)
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
    parser.add_argument("--save-raw-cache", action=argparse.BooleanOptionalAction, default=False,
                        help="离线模式是否保存 raw cache（默认不保存）")
    parser.add_argument("--raw-only", action="store_true",
                        help="保存 raw cache 后直接退出，不执行任何姿态后处理")
    parser.add_argument("--save-results", action="store_true",
                        help="保存指标及图片；默认仅在终端输出")
    args = parser.parse_args()
    if args.raw_only and not args.save_raw_cache:
        parser.error("--raw-only 需要保留 --save-raw-cache")
    if args.self_test:
        self_test()
        return
    app = SelectedGroupVisualizer(args.device, show_gt=not args.no_gt,
                                  visualize=args.visualize or bool(args.smoke_frames))
    if args.smoke_frames:
        frame_count = min(max(T, args.smoke_frames), len(app.paths))
        for index in range(frame_count):
            app.infer_frame(index)
        app.infer_gt_raw_as_training()
        app.postprocess_poses()
        app.plot_last_frame_errors()
        for index in range(frame_count):
            app.show_frame(index)
            record = app.records[index]
            assert np.isfinite(record["gt_crop_poses"]).all()
            assert app.axes.shape == (1, 3)
            actions = {int(person): ACTION_LABELS[int(action.argmax())]
                       for person, action, valid in zip(record["gt_crop_pose_track_ids"],
                                                       record["gt_crop_actions"],
                                                       record["gt_crop_action_valid"]) if valid}
            print(f"frame {index}: gt_poses={len(record['gt_crop_poses'])}, actions={actions}")
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
        if args.save_results:
            append_group_metrics_markdown(metrics, Path(__file__).with_name("record.md"))
            app.save_high_error_frames()
            app.save_metric_timeseries()
            app.save_point_count_timeseries()
        return
    app.precompute_visualization()
    app.show_frame(0)
    print("正在启动 WebAgg：http://127.0.0.1:8988/；空格/右箭头下一帧，左箭头回看。", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
