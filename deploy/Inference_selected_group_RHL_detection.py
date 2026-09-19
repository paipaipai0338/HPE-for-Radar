"""RHL 检测/跟踪框裁剪点云 -> P4Transformer 姿态评估。

保留 Inference_selected_group.py 的评估、平滑和绘图流程，仅将 GT 框选点
替换为 RHL/demo.py 的 DETR 检测与多目标跟踪结果。
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RHL_ROOT = PROJECT_ROOT / "RHL"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RHL_ROOT))

from src.models.milihupsen import DetrMiliHuPsen, DetrMiliHuPsenConfig
from src.models.modules.processor import PocPaddingProcessor
from src.utils.functions.customs import parseYAML
from src.utils.tracking.core import DetectionTarget, KalmanStats, TrackingAlgParam, TrackingInfo
from src.utils.tracking.functions import (
    cal_tatget_associate_gate,
    set_target,
    set_target_apriori_associate_points,
    set_target_associate_points,
    set_target_det_associate_points,
    set_target_size_associate_points,
    target_associate_greedy,
    target_predict,
    target_update,
    temp_target_update,
)
from WSC.API.recognizer import BehaviorRecognizer, PersonBox
from WSC.common.labels import LABEL_NAMES


# RHL 脚本独立入口；不会读取 GT-only 脚本的 INFERENCE_DATE/GROUP。
DATE = os.environ.get("RHL_INFERENCE_DATE", "20260912")
GROUP = os.environ.get("RHL_INFERENCE_GROUP", "group_099")
ROOT_PATH = Path("/mnt/huawei")
VISUALIZE = True
MODEL_POSE_PATH = PROJECT_ROOT / "experiments/P4Transformer/20260909_191752"
T = 8
POSE_OUTPUT_POSITION = 7
POSE_LOOKAHEAD = T - POSE_OUTPUT_POSITION - 1
MAX_POINTS = 300
TRAIN_MAX_POINTS = 200
GT_BBOX_MARGIN = 0.30
XYZ_LIMITS = ((0.0, 6.0), (-3.0, 3.0), (-2.0, 2.0))
ERROR_VIS_THRESHOLD_MM = 200.0
ERROR_VIS_MAX_FRAMES = 10
ERROR_VIS_ROOT = PROJECT_ROOT / "deploy/error_visualizations_RHL_detection"
METRIC_VIS_ROOT = PROJECT_ROOT / "deploy/metric_visualizations_RHL_detection"
METRIC_REFERENCE_MM = 150.0
ACCEPTANCE_CENTER_XY = np.array([2.0, 0.0], dtype=np.float32)
ACCEPTANCE_RADII_XY = np.array([1.6, 2.4], dtype=np.float32)
ACCEPTANCE_COLOR = "#FFF2CC"
ACTION_LABELS = ("stand", "sit_squat", "lie", "other")
ACTION_COLORS = ("#4daf4a", "#377eb8", "#e41a1c", "#984ea3")
STATIC_ACCUMULATION = True
DYNAMIC_STATIC_RATIO_THRESHOLD = 0.20
STATIC_HISTORY_FRAMES = 10
DYNAMIC_POINT_TAG = 1.0
SMOOTH_ALPHA = 0.35
MAX_INTERP_GAP = 8
MEDFILT_KERNEL = 7
VELOCITY_THRESHOLD = 0.35
BONE_LENGTH_WEIGHT = 0.65
BONE_ITERS = 2
PRESERVE_RAW_ROOT_POSITION = True
DIRECTED_BONES = [
    (11, 12), (11, 5), (12, 6), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 0), (0, 1), (1, 3),
    (0, 2), (2, 4),
]
DETECTION_CKPT = RHL_ROOT / "ckpt/ckpt_format_001"
VOXEL_CONFIG_PATH = RHL_ROOT / "src/configs/voxel.yaml"
DETECTION_THRESHOLD = 0.40
PERSON_XY_EXPANSION_RADIUS = 0.20
PERSON_Z_EXPANSION_RADIUS = 0.25
PERSON_OVERLAP_MARGIN = 0.10
BEHAVIOR_CHECKPOINT = (
    PROJECT_ROOT / "WSC/model_pointnet_8dim_t16_xy_expansion_0719_exp2_z/best.pt"
)
BEHAVIOR_MAX_POINTS = 128
BEHAVIOR_COLORS = ("#4daf4a", "#377eb8", "#e41a1c", "#984ea3")
POSE_MATCH_MAX_DISTANCE = 0.50  # 低位机髋中心距离，单位 m


# 复用算法源码，但 RHL 的参数只写入独立模块，不修改 GT-only 模块全局变量。
_base_spec = importlib.util.spec_from_file_location(
    "deploy.Inference_selected_group_RHL_base", PROJECT_ROOT / "deploy/Inference_selected_group.py"
)
if _base_spec is None or _base_spec.loader is None:
    raise ImportError("无法加载 GT-only 评估算法")
base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = base
_base_spec.loader.exec_module(base)
for _name in (
    "ROOT_PATH", "DATE", "GROUP", "VISUALIZE", "MODEL_POSE_PATH", "T",
    "POSE_OUTPUT_POSITION", "POSE_LOOKAHEAD", "MAX_POINTS", "TRAIN_MAX_POINTS",
    "GT_BBOX_MARGIN", "XYZ_LIMITS", "ERROR_VIS_THRESHOLD_MM", "ERROR_VIS_MAX_FRAMES",
    "ERROR_VIS_ROOT", "METRIC_VIS_ROOT", "METRIC_REFERENCE_MM",
    "ACCEPTANCE_CENTER_XY", "ACCEPTANCE_RADII_XY", "ACCEPTANCE_COLOR",
    "ACTION_LABELS", "ACTION_COLORS", "STATIC_ACCUMULATION",
    "DYNAMIC_STATIC_RATIO_THRESHOLD", "STATIC_HISTORY_FRAMES", "DYNAMIC_POINT_TAG",
    "SMOOTH_ALPHA", "MAX_INTERP_GAP", "MEDFILT_KERNEL", "VELOCITY_THRESHOLD",
    "BONE_LENGTH_WEIGHT", "BONE_ITERS", "PRESERVE_RAW_ROOT_POSITION",
    "DIRECTED_BONES",
):
    setattr(base, _name, globals()[_name])


def update_config(config, values):
    for key, value in values.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            update_config(current, value)
        elif isinstance(value, dict) and isinstance(current, dict):
            current.update(value)
        else:
            setattr(config, key, value)


def load_detection_model(device):
    config = DetrMiliHuPsenConfig()
    config_path = DETECTION_CKPT / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as source:
            update_config(config, json.load(source))
    model = DetrMiliHuPsen(config)
    model.load_pt(pt_path=DETECTION_CKPT / "DetrMiliHuPsen.pt")
    return model.to(device).eval()


def resolve_point_cloud_velocity(points, eps=1e-10):
    """与 RHL/demo.py 一致，将 [x,y,z,径向速度] 展开为六维状态。"""
    if not len(points):
        return np.empty((0, 6), dtype=np.float32)
    xyz = points[:, :3]
    radial_velocity = points[:, 3]
    distance = np.linalg.norm(xyz, axis=1)
    velocity = radial_velocity[:, None] * xyz / (distance[:, None] + eps)
    return np.column_stack((xyz, velocity)).astype(np.float32)


def detection_targets(result, voxel_config, fmt_points):
    boxes = result.get("boxes")
    if boxes is None or not len(boxes):
        return []
    boxes = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    region = voxel_config["region"]
    lower = np.array([region[axis][0] for axis in ("XLIM", "YLIM", "ZLIM")])
    extent = np.array([region[axis][1] - region[axis][0]
                       for axis in ("XLIM", "YLIM", "ZLIM")])
    targets = []
    for box in boxes:
        center = lower + box[:3] * extent
        size = box[3:6] * extent
        physical_box = np.concatenate((center - size / 2, center + size / 2)).astype(np.float32)
        mask = ((fmt_points[:, :3] >= physical_box[:3]) &
                (fmt_points[:, :3] <= physical_box[3:])).all(axis=1)
        associated = fmt_points[mask]
        center_state = (associated.mean(axis=0).astype(np.float32) if len(associated)
                        else np.zeros(6, dtype=np.float32))
        targets.append(DetectionTarget(
            bounding_box=physical_box,
            bbox_center=center.astype(np.float32),
            association_points=associated,
            center_state=center_state,
            point_mask=mask,
            point_indices=np.flatnonzero(mask),
        ))
    return targets


def box_corners(box):
    """将 [xmin,ymin,zmin,xmax,ymax,zmax] 展开为八角点。"""
    lower, upper = np.asarray(box)[:3], np.asarray(box)[3:]
    return np.asarray([
        [x, y, z]
        for z in (lower[2], upper[2])
        for y in (lower[1], upper[1])
        for x in (lower[0], upper[0])
    ], dtype=np.float32)


def expand_box_seed_indices(points, seed_indices, corners, xy_radius, z_radius):
    """按 Association.md 在低位机框限制内扩展种子点邻域。"""
    xyz = np.asarray(points)[:, :3]
    seeds = np.asarray(seed_indices, dtype=int)
    corners = np.asarray(corners, dtype=np.float32)
    if not len(seeds):
        return seeds
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return np.empty(0, dtype=int)
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    candidate_mask = np.all(
        (xyz[:, :2] >= lower[:2]) & (xyz[:, :2] <= upper[:2]), axis=1
    )
    candidate_mask &= ((xyz[:, 2] >= lower[2] - z_radius) &
                       (xyz[:, 2] <= upper[2] + z_radius))
    candidate_mask[seeds] = True
    candidates = np.flatnonzero(candidate_mask)
    offsets = xyz[candidates, None] - xyz[seeds]
    near_seed = (
        (np.linalg.norm(offsets[..., :2], axis=-1) <= xy_radius)
        & (np.abs(offsets[..., 2]) <= z_radius)
    ).any(axis=1)
    return candidates[near_seed]


def resolve_person_point_overlap(points, seeds, candidates, centers,
                                 ambiguity_margin=PERSON_OVERLAP_MARGIN):
    """共享点唯一归属；距离难以区分时丢弃该点。"""
    if len(candidates) < 2:
        return candidates
    claims = np.zeros((len(candidates), len(points)), dtype=bool)
    seed_claims = np.zeros_like(claims)
    for row, (seed, candidate) in enumerate(zip(seeds, candidates, strict=True)):
        claims[row, candidate] = True
        seed_claims[row, seed] = True
    shared = claims.sum(axis=0) > 1
    if not shared.any():
        return candidates
    seed_counts = seed_claims.sum(axis=0)
    references = np.asarray(centers, dtype=np.float32)[:, :2].copy()
    for row in range(len(seeds)):
        exclusive = seed_claims[row] & (seed_counts == 1)
        if exclusive.sum() >= 5:
            references[row] = np.median(points[exclusive, :2], axis=0)
    for point_index in np.flatnonzero(shared):
        owners = np.flatnonzero(claims[:, point_index])
        claims[:, point_index] = False
        exclusive_seed_owner = np.flatnonzero(seed_claims[:, point_index])
        if len(exclusive_seed_owner) == 1:
            claims[exclusive_seed_owner[0], point_index] = True
            continue
        distances = np.linalg.norm(
            references[owners] - points[point_index, :2], axis=1
        )
        order = np.argsort(distances)
        if len(order) == 1 or distances[order[1]] - distances[order[0]] > ambiguity_margin:
            claims[owners[order[0]], point_index] = True
    return [np.flatnonzero(row) for row in claims]


def associate_tracked_points(high_points, targets, rotation, translation):
    """高位机取种子，在低位机坐标扩点/消歧，返回原始点索引。"""
    if not targets:
        return []
    seeds = []
    low_corners = []
    for target in targets:
        box = np.asarray(target.bounding_box, dtype=np.float32)
        valid = box.shape == (6,) and np.isfinite(box).all()
        seeds.append(np.flatnonzero(
            ((high_points[:, :3] >= box[:3]) &
             (high_points[:, :3] <= box[3:])).all(axis=1)
        ) if valid else np.empty(0, dtype=int))
        low_corners.append(
            base.transform_points(box_corners(box), rotation, translation)
            if valid else np.full((8, 3), np.nan, dtype=np.float32)
        )
    low_points = base.transform_cloud(high_points, rotation, translation)
    candidates = [
        expand_box_seed_indices(
            low_points, seed, corners,
            PERSON_XY_EXPANSION_RADIUS, PERSON_Z_EXPANSION_RADIUS,
        )
        for seed, corners in zip(seeds, low_corners, strict=True)
    ]
    centers = np.asarray([corners.mean(axis=0) for corners in low_corners])
    return resolve_person_point_overlap(
        low_points, seeds, candidates, centers, PERSON_OVERLAP_MARGIN
    )


def append_uid_frame(history, timestamp, pose_slot, low_cloud, max_gap):
    """姿态和行为共用的 UID 帧连续性；空关联和超时同时断开两路窗口。"""
    reset = not len(low_cloud) or (bool(history) and timestamp - history[-1][0] > max_gap)
    if reset:
        history.clear()
    if len(low_cloud):
        history.append((timestamp, pose_slot, low_cloud))
    return reset


def uid_legend_label(uid, result, raw_error, smooth_error):
    behavior = result.label if result is not None and result.label_index is not None else "N/A"
    raw = "N/A" if raw_error is None else f"{raw_error:.0f}"
    smooth = "N/A" if smooth_error is None else f"{smooth_error:.0f}"
    return f"UID {uid} | {behavior} | MPJPE raw/smoothed: {raw} / {smooth} mm"


class RHLDetectionVisualizer(base.SelectedGroupVisualizer):
    def __init__(self, device, show_gt=True, visualize=VISUALIZE,
                 detection_threshold=DETECTION_THRESHOLD, date=DATE, group=GROUP):
        # 独立评估模块的日期/group 只跟随这个 RHL 实例。
        self.date, self.group = date, group
        base.DATE, base.GROUP = date, group
        super().__init__(device, show_gt=show_gt, visualize=visualize)
        self.voxel_config = parseYAML(file_path=VOXEL_CONFIG_PATH)
        self.detector_processor = PocPaddingProcessor(voxel_config=self.voxel_config)
        self.detector = load_detection_model(self.device)
        self.detection_threshold = detection_threshold
        self.tracking_info = TrackingInfo()
        self.kalman_stats = KalmanStats()
        self.tracking_params = TrackingAlgParam()
        self.detection_accum_tracks = {}
        calib = ROOT_PATH / self.date / "calib"
        self.behavior_recognizer = BehaviorRecognizer(
            checkpoint=BEHAVIOR_CHECKPOINT,
            low_extrinsic=calib / "extrinsic_img_to_radar_low.npz",
            high_extrinsic=calib / "extrinsic_img_to_radar_high.npz",
            device=self.device,
            max_points=BEHAVIOR_MAX_POINTS,
            box_expansion_xy=PERSON_XY_EXPANSION_RADIUS,
            box_expansion_z=PERSON_Z_EXPANSION_RADIUS,
            overlap_margin=PERSON_OVERLAP_MARGIN,
        )
        # 单一 UID 历史保存同一次关联的高位机姿态槽位和低位机行为点云。
        self.uid_frames = {}

    def update_tracking(self, detections, fmt_points):
        info, params = self.tracking_info, self.tracking_params
        target_predict(info, self.kalman_stats)
        cal_tatget_associate_gate(info, params)
        set_target_apriori_associate_points(info, fmt_points, params)
        target_associate_greedy(info, detections, params)
        set_target(info, detections, fmt_points, params)
        set_target_det_associate_points(info, detections, fmt_points)
        target_update(info, detections, params, self.kalman_stats)
        set_target_size_associate_points(info, fmt_points)
        set_target_associate_points(info)
        temp_target_update(info, params)

    @staticmethod
    def evaluation_ids(track_ids, boxes, gt):
        """只为指标/绘图把跟踪目标匹配到 GT 人员；不参与点云框选。"""
        mapped = np.arange(len(track_ids), dtype=int) + 1_000_000
        if not len(boxes) or not len(gt):
            return mapped
        box_centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        gt_centers = gt[:, [11, 12]].mean(axis=1)
        valid_boxes = np.isfinite(box_centers).all(axis=1)
        valid_gt = np.isfinite(gt_centers).all(axis=1)
        if not valid_boxes.any() or not valid_gt.any():
            return mapped
        rows, cols = linear_sum_assignment(
            np.linalg.norm(
                box_centers[valid_boxes, None] - gt_centers[None, valid_gt],
                axis=-1,
            )
        )
        mapped[np.flatnonzero(valid_boxes)[rows]] = np.flatnonzero(valid_gt)[cols]
        return mapped

    @staticmethod
    def align_boxes_for_display(boxes, evaluation_ids, gt_count):
        """适配原绘图的一人一框结构；完整检测框仍单独保存在 record 中。"""
        aligned = np.full((gt_count, 6), np.nan, dtype=np.float32)
        for box, person_id in zip(boxes, evaluation_ids):
            if 0 <= person_id < gt_count:
                aligned[person_id] = box
        return aligned

    @torch.inference_mode()
    def infer_frame(self, index):
        if index != len(self.records):
            raise ValueError("新帧必须按时间顺序推理，回看请使用缓存。")
        path = self.paths[index]
        points, skip_reason = base.load_point_cloud(path)
        if skip_reason is not None:
            print(f"SKIPPED FRAME {self.date}/{self.group} {path.name}: {skip_reason}")

        # 与姿态训练 collate 一致：先对整帧采样，再在固定 200 个槽位上框选清零。
        if len(points) > TRAIN_MAX_POINTS:
            sample_indices = np.random.default_rng(index).choice(
                len(points), TRAIN_MAX_POINTS, replace=False
            )
        else:
            sample_indices = np.arange(len(points))
        sampled = points[sample_indices]

        record = {"path": str(path)}
        self.records.append(record)
        gt, _ = self.load_gt(record)

        raw_detector_points = points[:, :4]
        fmt_points = resolve_point_cloud_velocity(raw_detector_points)
        inputs = self.detector_processor(raw_point_cloud=raw_detector_points).to(self.device)
        outputs = self.detector(inputs)
        result = self.detector_processor.post_process_object_detection(
            outputs=outputs, threshold=self.detection_threshold
        )[0]
        detections = detection_targets(result, self.voxel_config, fmt_points)
        self.update_tracking(detections, fmt_points)

        for track in self.detection_accum_tracks.values():
            track.history.append(None)

        active = list(self.tracking_info.tracking_targets)
        boxes = np.asarray([target.bounding_box for target in active], dtype=np.float32).reshape(-1, 6)
        uids = np.asarray([target.uid for target in active], dtype=int)
        assignments = associate_tracked_points(points, active, *self.high_to_low)
        seconds, nanoseconds = path.stem.split("_")
        timestamp = int(seconds) + int(nanoseconds) / 1e9
        low_points = base.transform_cloud(points[:, :6], *self.high_to_low)
        behavior_inputs, behavior_clouds, behavior_windows = [], {}, {}
        eval_ids = self.evaluation_ids(uids, boxes, gt)
        display_boxes = self.align_boxes_for_display(boxes, eval_ids, len(gt))
        selected_mask = np.zeros(len(points), dtype=bool)
        pose_tracks, pose_positions = [], []
        accum_tracks, accum_positions, accumulated_clouds = [], [], []

        for position, (target, assigned_indices) in enumerate(
            zip(active, assignments, strict=True)
        ):
            original_mask = np.zeros(len(points), dtype=bool)
            original_mask[assigned_indices] = True
            selected_mask |= original_mask
            sampled_mask = original_mask[sample_indices]
            uid = int(target.uid)
            history = self.uid_frames.setdefault(
                uid, deque(maxlen=self.behavior_recognizer.config.sequence_length)
            )
            low_cloud = low_points[assigned_indices]
            accum = self.detection_accum_tracks.setdefault(
                uid, base.PoseTrack(uid, target)
            )
            pose_slot = (base.mask_sampled_points(sampled, sampled_mask)
                         if sampled_mask.any() else None)
            reset = append_uid_frame(
                history, timestamp, pose_slot, low_cloud,
                self.behavior_recognizer.max_frame_gap_seconds,
            )
            if reset:
                accum.history.clear()
                accum.static_history.clear()
            if np.isfinite(target.bounding_box).all():
                behavior_inputs.append(PersonBox(uid, target.bounding_box))
                behavior_clouds[uid] = low_cloud
                behavior_windows[uid] = [entry[2] for entry in history]
            if sampled_mask.any():
                pose_history = deque(
                    (entry[1] for entry in list(history)[-T:]), maxlen=T
                )
                pose_tracks.append(base.PoseTrack(uid, target, history=pose_history))
                pose_positions.append(position)

            if len(accum.history) == 0:
                accum.history.extend([None] * min(len(self.records), T))
            cloud, _, _ = base.accumulate_track_static_points(
                accum, sampled, sampled_mask, (target.bounding_box[:3] + target.bounding_box[3:]) / 2
            )
            crop, _ = base.crop_and_pad(
                cloud, point_mask=np.ones(len(cloud), dtype=bool), max_points=TRAIN_MAX_POINTS
            )
            if crop is not None:
                accum.history[-1] = crop
                accum_tracks.append(accum)
                accum_positions.append(position)
                accumulated_clouds.append(cloud[np.isfinite(cloud).all(axis=1)])

        behavior_results = self.behavior_recognizer.update(
            points[:, :6], behavior_inputs, timestamp,
            associated_low_points=behavior_clouds,
            frame_windows=behavior_windows,
        )

        output_position = -1 if self.visualize else POSE_OUTPUT_POSITION
        window_ready = self.visualize or index >= T - 1

        def predict(tracks, positions):
            if not window_ready or not tracks:
                return np.empty((0, 17, 3), np.float32), np.empty(0, dtype=int)
            batch = {key: value.to(self.device) for key, value in base.build_pose_input(tracks).items()}
            valid = batch["mask"][:, output_position].any(dim=1).cpu().numpy()
            poses = self.pose_model(batch)["pose"][:, output_position, 0].cpu().numpy()[valid]
            return poses, np.asarray(positions, dtype=int)[valid]

        poses, pose_positions = predict(pose_tracks, pose_positions)
        accum_poses, accum_positions = predict(accum_tracks, accum_positions)
        record.update(
            # 姿态历史和平滑只认 RHL tracker uid；GT 对齐绝不反馈进轨迹。
            gt_crop_track_ids=uids,
            gt_crop_pose_indices=pose_positions,
            gt_crop_poses_raw=poses,
            gt_boxes=display_boxes,
            gt_selected_mask=selected_mask,
            gt_accum_track_ids=uids,
            gt_accum_pose_indices=accum_positions,
            gt_accum_poses_raw=accum_poses,
            gt_accumulated_person_indices=eval_ids[np.asarray(accum_positions, dtype=int)],
            gt_accumulated_points_by_person=accumulated_clouds,
            gt_accumulated_points=(np.concatenate(accumulated_clouds)
                                   if accumulated_clouds else points[:0]),
            detection_track_uids=uids,
            detection_eval_ids=eval_ids,
            detection_boxes=boxes,
            behavior_results={result.track_id: result for result in behavior_results},
        )
        if self.visualize and not self.defer_visual_smoothing:
            base.smooth_pose_records(self.records, "gt_crop_")
            base.smooth_pose_records(self.records, "gt_accum_")
        return record, points, selected_mask

    def infer_gt_raw_as_training(self):
        """检测裁剪已按训练的采样/填充顺序构造，不再用 GT 框重算。"""
        for record in self.records:
            record["gt_training_window"] = True

    @staticmethod
    def pose_eval_ids(record, prefix, stage):
        """将预测的稳定 uid 映射到当前帧 GT id，仅供评估/显示。"""
        if stage == "raw":
            positions = record[f"{prefix}pose_indices"]
            uids = record[f"{prefix}track_ids"][positions]
        else:
            uids = record[f"{prefix}pose_track_ids"]
        mapping = dict(zip(record["detection_track_uids"], record["detection_eval_ids"]))
        return np.asarray([mapping.get(int(uid), -1) for uid in uids], dtype=int)

    def uid_mpjpe(self, record, prefix, stage):
        """椭圆内按低位机髋中心一对一匹配；超过配置距离不计分。"""
        poses = record[f"{prefix}{'poses_raw' if stage == 'raw' else 'poses'}"]
        if stage == "raw":
            uids = record[f"{prefix}track_ids"][record[f"{prefix}pose_indices"]]
        else:
            uids = record[f"{prefix}pose_track_ids"]
        gt = record["gt"]
        pred_valid = (np.isfinite(poses).all(axis=(1, 2))
                      & base.acceptance_mask(poses, *self.high_to_low))
        gt_valid = (np.isfinite(gt).all(axis=(1, 2))
                    & base.acceptance_mask(gt, *self.high_to_low))
        poses, uids, gt = poses[pred_valid], uids[pred_valid], gt[gt_valid]
        if not len(poses) or not len(gt):
            return {}
        pred_roots = base.transform_points(poses[:, [11, 12]].mean(axis=1),
                                           *self.high_to_low)
        gt_roots = base.transform_points(gt[:, [11, 12]].mean(axis=1),
                                         *self.high_to_low)
        distances = np.linalg.norm(pred_roots[:, None] - gt_roots[None], axis=-1)
        # 虚拟 GT 列允许预测不匹配，避免远处目标占用真实 GT。
        costs = np.concatenate((np.where(distances <= POSE_MATCH_MAX_DISTANCE,
                                         distances, 1e6),
                                np.full((len(poses), len(poses)),
                                        POSE_MATCH_MAX_DISTANCE + 1e-6)), axis=1)
        rows, cols = linear_sum_assignment(costs)
        return {
            int(uids[row]): float(np.linalg.norm(poses[row] - gt[col], axis=-1).mean() * 1000)
            for row, col in zip(rows, cols)
            if col < len(gt) and distances[row, col] <= POSE_MATCH_MAX_DISTANCE
        }

    def evaluate_group(self):
        """使用稳定 uid 平滑；只在逐帧计分边界映射到 GT id。"""
        if len(self.records) != len(self.paths):
            raise ValueError("整组尚未推理完成，不能报告整组 MPJPE")
        if self.group_metrics is not None:
            return self.group_metrics
        self.infer_gt_raw_as_training()
        base.smooth_pose_records(self.records, "gt_crop_")
        base.smooth_pose_records(self.records, "gt_accum_")

        def summarize(prefix, inside_only):
            result = {}
            for stage, suffix in (("raw", "poses_raw"), ("smoothed", "poses")):
                values = []
                for metric in (base.frame_mpjpe, base.frame_centered_mpjpe,
                               base.frame_pa_mpjpe):
                    stats = []
                    for record in self.records:
                        poses = record[f"{prefix}{suffix}"]
                        gt = record["gt"]
                        if inside_only:
                            inside = base.acceptance_mask(gt, *self.high_to_low)
                            eval_ids = self.pose_eval_ids(record, prefix, stage)
                            poses = poses[np.isin(eval_ids, np.flatnonzero(inside))]
                            gt = gt[inside]
                        stats.append(metric(poses, gt))
                    values.append(base.aggregate_mpjpe(stats)["mpjpe_mm"])
                result[stage] = {
                    "mpjpe_mm": values[0],
                    "centered_mpjpe_mm": values[1],
                    "pa_mpjpe_mm": values[2],
                }
            return result

        regions = {
            "inside": {
                "current_frame": summarize("gt_crop_", True),
                "temporal_accumulation": summarize("gt_accum_", True),
            },
            "all": {
                "current_frame": summarize("gt_crop_", False),
                "temporal_accumulation": summarize("gt_accum_", False),
            },
        }
        self.group_metrics = {
            "date": self.date, "group": self.group, "frames": len(self.records),
            "gt_frames": sum(record["has_gt"] for record in self.records),
            "regions": regions,
        }
        return self.group_metrics

    def inside_frame_mpjpe(self, record, prefix, stage):
        inside = base.acceptance_mask(record["gt"], *self.high_to_low)
        poses = record[f"{prefix}{'poses_raw' if stage == 'raw' else 'poses'}"]
        eval_ids = self.pose_eval_ids(record, prefix, stage)
        poses = poses[np.isin(eval_ids, np.flatnonzero(inside))]
        return base.frame_mpjpe(poses, record["gt"][inside])[0]

    def save_metric_timeseries(self):
        backups = []
        for record in self.records:
            saved = {}
            for prefix in ("gt_crop_", "gt_accum_"):
                for stage, key in (("raw", f"{prefix}pose_indices"),
                                   ("smoothed", f"{prefix}pose_track_ids")):
                    saved[key] = record[key]
                    record[key] = self.pose_eval_ids(record, prefix, stage)
            backups.append(saved)
        try:
            return super().save_metric_timeseries()
        finally:
            for record, saved in zip(self.records, backups, strict=True):
                record.update(saved)

    def save_behavior_results(self):
        """保存逐帧/逐 UID 行为结果，并绘制后处理标签时间线。"""
        output_dir = METRIC_VIS_ROOT / self.date / self.group
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        track_ids = set()
        for frame_index, record in enumerate(self.records, start=1):
            for uid, result in record.get("behavior_results", {}).items():
                if result.label_index is None:
                    continue
                track_ids.add(int(uid))
                rows.append({
                    "frame": frame_index,
                    "timestamp": Path(record["path"]).stem,
                    "track_uid": int(uid),
                    "label": result.label,
                    "label_index": result.label_index,
                })
        json_path = output_dir / "behavior_results_RHL_detection.json"
        with json_path.open("w", encoding="utf-8") as target:
            json.dump(rows, target, ensure_ascii=False, indent=2)

        fig, ax = base.plt.subplots(figsize=(12, 5), constrained_layout=True)
        for row_index, uid in enumerate(sorted(track_ids)):
            uid_rows = [row for row in rows if row["track_uid"] == uid]
            frames = [row["frame"] for row in uid_rows if row["label_index"] is not None]
            labels = [row["label_index"] + row_index * 5
                      for row in uid_rows if row["label_index"] is not None]
            if frames:
                ax.step(frames, labels, where="post", linewidth=1.5, label=f"UID {uid}")
        ticks, tick_labels = [], []
        for row_index, uid in enumerate(sorted(track_ids)):
            for label_index, label in enumerate(LABEL_NAMES):
                ticks.append(label_index + row_index * 5)
                tick_labels.append(f"UID {uid}: {label}")
        ax.set(title=f"{self.date}/{self.group} - WSC behavior timeline",
               xlabel="Frame idx", ylabel="RHL track / behavior",
               yticks=ticks, yticklabels=tick_labels)
        ax.grid(alpha=.25)
        if track_ids:
            ax.legend()
        plot_path = output_dir / "behavior_timeseries_RHL_detection.png"
        fig.savefig(plot_path, dpi=150)
        base.plt.close(fig)
        print(f"行为识别结果已保存到 {json_path} 和 {plot_path}")
        return json_path, plot_path

    def show_frame(self, index):
        if index == len(self.records):
            self.infer_frame(index)
        record = self.records[index]
        backups = {}
        # 原绘图按 GT id 筛选；临时提供展示 ID，平滑历史中的 uid 不变。
        for prefix in ("gt_crop_", "gt_accum_"):
            for stage, key in (("raw", f"{prefix}pose_indices"),
                               ("smoothed", f"{prefix}pose_track_ids")):
                backups[key] = record[key]
                record[key] = self.pose_eval_ids(record, prefix, stage)
        try:
            super().show_frame(index)
        finally:
            record.update(backups)
        if self.fig is not None:
            self.axes[0, 0].set_title("Current points: RHL detection/tracking crop")
            self.axes[1, 0].set_title("Accumulated points: RHL tracked crop")
            self.axes[0, 1].set_title("Current crop pose: RHL detection")
            self.axes[1, 1].set_title("Accumulated pose: RHL detection")
            # 父类姿态图例是全目标平均误差；RHL 只展示逐 UID 结果。
            for ax in (self.axes[0, 1], self.axes[1, 1]):
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            behavior_results = record.get("behavior_results", {})
            crop_raw_errors = self.uid_mpjpe(record, "gt_crop_", "raw")
            crop_smooth_errors = self.uid_mpjpe(record, "gt_crop_", "smoothed")
            accum_raw_errors = self.uid_mpjpe(record, "gt_accum_", "raw")
            accum_smooth_errors = self.uid_mpjpe(record, "gt_accum_", "smoothed")
            crop_handles, accum_handles = [], []
            for uid, box in zip(record["detection_track_uids"],
                                record["detection_boxes"], strict=True):
                result = behavior_results.get(int(uid))
                if not np.isfinite(box).all():
                    continue
                position = base.transform_points(
                    np.asarray([[(box[0] + box[3]) / 2,
                                 (box[1] + box[4]) / 2, box[5]]]),
                    *self.high_to_low,
                )[0]
                color = ("gray" if result is None or result.label_index is None
                         else BEHAVIOR_COLORS[result.label_index])
                for handles, raw_errors, smooth_errors in (
                    (crop_handles, crop_raw_errors, crop_smooth_errors),
                    (accum_handles, accum_raw_errors, accum_smooth_errors),
                ):
                    handles.append(base.Line2D(
                        [], [], color=color, marker="o", linestyle="None",
                        label=uid_legend_label(int(uid), result,
                                               raw_errors.get(int(uid)),
                                               smooth_errors.get(int(uid))),
                    ))
                for ax in self.axes.flat:
                    ax.text(*position, f"UID {uid}", color=color, fontsize=7,
                            bbox={"facecolor": "white", "alpha": .7,
                                  "edgecolor": color, "pad": 2})
            for ax, handles in ((self.axes[0, 1], crop_handles),
                                (self.axes[1, 1], accum_handles)):
                if handles:
                    ax.legend(handles=handles, fontsize=7.5, loc="upper right")
            self.fig.canvas.draw_idle()


def self_test():
    from deploy import Inference_selected_group as original

    assert base is not original
    assert (base.DATE, base.GROUP, base.MODEL_POSE_PATH) == (DATE, GROUP, MODEL_POSE_PATH)
    assert (original.DATE, original.GROUP) == (
        os.environ.get("INFERENCE_DATE", "20260912"),
        os.environ.get("INFERENCE_GROUP", "group_029"),
    )
    points = np.array([[1, 0, 0, 2], [0, 2, 0, 3]], dtype=np.float32)
    velocity = resolve_point_cloud_velocity(points)
    np.testing.assert_allclose(velocity[0, 3:], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(velocity[1, 3:], [0, 3, 0], atol=1e-6)
    mask = np.array([True, False])
    padded, valid = base.mask_sampled_points(points, mask)
    assert padded.shape == (TRAIN_MAX_POINTS, 4) and valid.sum() == 1
    boxes = np.array([[0, 0, 0, 2, 2, 2], [np.nan] * 6], dtype=np.float32)
    gt = np.zeros((2, 17, 3), dtype=np.float32)
    gt[1] = np.nan
    ids = RHLDetectionVisualizer.evaluation_ids(np.array([7, 8]), boxes, gt)
    np.testing.assert_array_equal(ids, [0, 1_000_001])
    app = object.__new__(RHLDetectionVisualizer)
    app.high_to_low = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    pose = np.zeros((2, 17, 3), dtype=np.float32)
    pose[0, :, 0] = 2.0
    pose[1, :, 0] = 2.5
    truth = pose.copy()
    truth[0, :, 0] += .05
    record = {"gt": truth, "gt_crop_poses_raw": pose,
              "gt_crop_track_ids": np.array([7, 8]),
              "gt_crop_pose_indices": np.array([0, 1])}
    errors = app.uid_mpjpe(record, "gt_crop_", "raw")
    assert set(errors) == {7, 8} and abs(errors[7] - 50) < 1e-3
    truth[1, :, 0] = 2.5 + POSE_MATCH_MAX_DISTANCE + .2
    assert set(app.uid_mpjpe(record, "gt_crop_", "raw")) == {7}
    cloud = np.array([
        [0.00, 0, 0, 0], [0.15, 0, 0, 0], [0.30, 0, 0, 0],
        [0.45, 0, 0, 0], [0.60, 0, 0, 0],
    ], dtype=np.float32)
    corners = box_corners(np.array([0, -.1, -.1, .5, .1, .1]))
    expanded = expand_box_seed_indices(cloud, np.array([1]), corners, .2, .25)
    np.testing.assert_array_equal(expanded, [0, 1, 2])
    assigned = resolve_person_point_overlap(
        cloud, [np.array([0]), np.array([4])],
        [np.array([0, 1, 2]), np.array([2, 3, 4])],
        np.array([[0, 0, 0], [.6, 0, 0]]), .1,
    )
    assert 2 not in assigned[0] and 2 not in assigned[1]
    class Target:
        bounding_box = np.array([0, -.1, -.1, .5, .1, .1], dtype=np.float32)
    transformed = associate_tracked_points(
        cloud, [Target()], np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_array_equal(transformed[0], [0, 1, 2, 3])
    uid_history = deque(maxlen=16)
    behavior_cloud = np.ones((2, 6), dtype=np.float32)
    for frame in range(18):
        assert not append_uid_frame(uid_history, frame * .1, frame, behavior_cloud, .5)
    assert len(uid_history) == 16
    assert [item[1] for item in list(uid_history)[-T:]] == list(range(10, 18))
    assert append_uid_frame(uid_history, 2.5, None, behavior_cloud, .5)
    assert len(uid_history) == 1 and uid_history[0][1] is None
    assert append_uid_frame(uid_history, 2.6, None, behavior_cloud[:0], .5)
    assert not uid_history
    assert uid_legend_label(7, None, 50, None) == (
        "UID 7 | N/A | MPJPE raw/smoothed: 50 / N/A mm"
    )
    print("RHL detection self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DATE, help="RHL 评估日期，默认读取 RHL_INFERENCE_DATE")
    parser.add_argument("--group", default=GROUP, help="RHL 评估组，默认读取 RHL_INFERENCE_GROUP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD)
    parser.add_argument("--no-gt", action="store_true", help="不读取相机 GT")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction,
                        default=VISUALIZE)
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    app = RHLDetectionVisualizer(
        args.device, show_gt=not args.no_gt,
        visualize=args.visualize or bool(args.smoke_frames),
        detection_threshold=args.threshold,
        date=args.date, group=args.group,
    )
    if args.smoke_frames:
        for index in range(min(args.smoke_frames, len(app.paths))):
            app.show_frame(index)
        summary = {
            uid: result.label
            for uid, result in app.records[-1].get("behavior_results", {}).items()
            if result.label_index is not None
        }
        print(f"last-frame behavior={summary}")
        app.fig.canvas.draw()
        torch.cuda.empty_cache()
        return
    if not args.visualize:
        for index in tqdm(range(len(app.paths)), desc="RHL detection crop inference"):
            app.infer_frame(index)
        metrics = app.evaluate_group()
        base.append_group_metrics_markdown(
            metrics, Path(__file__).with_name("record_RHL_detection.md")
        )
        app.save_high_error_frames()
        app.save_metric_timeseries()
        app.save_point_count_timeseries()
        app.save_behavior_results()
        return
    app.precompute_visualization()
    app.show_frame(0)
    print("WebAgg 使用端口 8988；空格/右箭头下一帧，左箭头回看。")
    base.plt.show()


if __name__ == "__main__":
    main()
