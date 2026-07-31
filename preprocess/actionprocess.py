"""
单帧行为判决接口

调用示例:

    from src.action_recognition import LABEL_NAMES, classify_actions

    result = classify_actions(poses)
    labels = result.labels  # N x 4，one-hot标签，类别顺序见 LABEL_NAMES
    valid = result.valid    # N，False 表示该人体包含 NaN 或 Inf，True表示有效

注：
    输入最后两维必须为 17 x 3，前面可以是任意批量维度。
    例如（N表示人数）：
        N x 17 x 3 输出 N x 4，
        B x T x N x 17 x 3 输出 B x T x N x 4。
    输入必须使用 COCO-17 关键点顺序，
    且指定的竖直轴正方向朝上（校正到低位机坐标系）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


LABEL_NAMES = ("stand", "sit_squat", "lie", "other")
STAND, SIT_SQUAT, LIE, OTHER = range(len(LABEL_NAMES))


@dataclass(frozen=True, slots=True)
class StandRules:
    height_ratio_min: float = 1.05                  # 人体竖直包围范围/身体尺度的下限
    torso_angle_max: float = 30.0                   # 躯干偏离竖直方向的最大角度（度）
    shoulder_to_hip_min: float = 0.18               # 肩相对髋的归一化竖直高度下限
    support_hip_to_ankle_min: float = 0.40          # 支撑腿髋到踝的归一化竖直高度下限
    support_knee_to_ankle_min: float = 0.15         # 支撑腿膝到踝的归一化竖直高度下限
    support_knee_angle_min: float = 130.0           # 支撑腿膝关节最小夹角（度）
    step_height_ratio_min: float = 0.95             # 行走或抬腿时人体竖直比例下限
    step_hip_to_ankle_min: float = 0.35             # 行走或抬腿时髋到踝竖直比例下限
    step_hip_to_ankle_max: float = 0.48             # 行走或抬腿时髋到踝竖直比例上限
    step_knee_angle_difference_min: float = 15.0    # 行走时双膝夹角差下限（度）


@dataclass(frozen=True, slots=True)
class SitSquatRules:
    height_ratio_min: float = 0.40                  # 坐蹲人体竖直比例下限，排除近水平姿态
    shoulder_to_hip_min: float = 0.06               # 肩必须高于髋的归一化竖直距离
    hip_to_ankle_min: float = 0.12                  # 髋必须高于踝的归一化竖直距离
    torso_angle_max: float = 70.0                   # 坐蹲躯干偏离竖直方向的最大角度（度）
    seated_height_ratio_max: float = 1.08           # 标准坐姿人体竖直比例上限
    seated_hip_to_ankle_max: float = 0.52           # 标准坐姿髋到踝竖直比例上限
    seated_hip_to_knee_max: float = 0.15            # 标准坐姿髋到膝竖直比例上限
    seated_min_knee_to_ankle_min: float = 0.06      # 两腿中较小的膝踝竖直比例下限
    seated_max_knee_angle_max: float = 145.0        # 两个膝关节中较大夹角的上限（度）
    squat_height_ratio_max: float = 0.85            # 标准蹲姿人体竖直比例上限
    squat_hip_to_ankle_max: float = 0.47            # 标准蹲姿髋到踝竖直比例上限
    squat_body_angle_max: float = 55.0              # 踝到鼻方向偏离竖直的最大角度（度）
    squat_mean_knee_angle_max: float = 105.0        # 双膝平均夹角上限（度）
    squat_knee_to_ankle_min: float = 0.12           # 蹲姿平均膝到踝竖直比例下限


@dataclass(frozen=True, slots=True)
class LieRules:
    horizontal_ratio_min: float = 0.55              # 人体水平包围范围/身体尺度的下限
    torso_angle_min: float = 68.0                   # 躯干偏离竖直方向的最小角度（度）
    hip_to_ankle_max: float = 0.14                  # 髋到踝的归一化竖直距离上限


@dataclass(frozen=True, slots=True)
class ActionRules:
    vertical_axis: int = 2                                              # 竖直坐标轴：0=x、1=y、2=z
    min_body_scale: float = 1.0e-6                                      # 身体尺度有效下限，避免除零
    stand: StandRules = field(default_factory=StandRules)               # 站立与行走规则
    sit_squat: SitSquatRules = field(default_factory=SitSquatRules)     # 坐姿与蹲姿规则
    lie: LieRules = field(default_factory=LieRules)                     # 仰躺、侧躺与俯卧规则


@dataclass(frozen=True, slots=True)
class ActionResult:
    """与输入人体逐一对齐的 one-hot 行为标签和有效性掩码。"""

    labels: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True, slots=True)
class _PoseFeatures:
    height_ratio: float
    horizontal_ratio: float
    torso_angle: float
    body_angle: float
    hip_to_ankle: float
    hip_to_knee: float
    knee_to_ankle: float
    min_knee_to_ankle: float
    shoulder_to_hip: float
    left_knee_angle: float
    right_knee_angle: float
    mean_knee_angle: float
    support_hip_to_ankle: float
    support_knee_to_ankle: float
    support_knee_angle: float


DEFAULT_ACTION_RULES = ActionRules()


def _midpoint(pose: np.ndarray, left: int, right: int) -> np.ndarray:
    return (pose[left] + pose[right]) * 0.5


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1.0e-6:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _angle_from_vertical(vector: np.ndarray, vertical_axis: int) -> float:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-6:
        return 90.0
    cosine = abs(vector[vertical_axis]) / norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _max_pairwise_distance(points: np.ndarray) -> float:
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return float(np.linalg.norm(differences, axis=-1).max())


def _extract_features(
    pose: np.ndarray,
    rules: ActionRules,
) -> _PoseFeatures | None:
    vertical_axis = rules.vertical_axis
    shoulder = _midpoint(pose, 5, 6)
    hip = _midpoint(pose, 11, 12)
    knee = _midpoint(pose, 13, 14)
    ankle = _midpoint(pose, 15, 16)

    # 所有距离特征均用躯干与双腿长度归一化，避免依赖人体绝对身高。
    left_leg = np.linalg.norm(pose[11] - pose[13]) + np.linalg.norm(pose[13] - pose[15])
    right_leg = np.linalg.norm(pose[12] - pose[14]) + np.linalg.norm(pose[14] - pose[16])
    body_scale = np.linalg.norm(shoulder - hip) + 0.5 * (left_leg + right_leg)
    if body_scale < rules.min_body_scale:
        return None

    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
    height_extent = float(np.ptp(pose[:, vertical_axis]))
    horizontal_extent = _max_pairwise_distance(pose[:, horizontal_axes])
    hip_to_ankle = float((hip - ankle)[vertical_axis] / body_scale)
    hip_to_knee = float((hip - knee)[vertical_axis] / body_scale)
    knee_to_ankle = float((knee - ankle)[vertical_axis] / body_scale)
    min_knee_to_ankle = float(
        min(
            (pose[13] - pose[15])[vertical_axis],
            (pose[14] - pose[16])[vertical_axis],
        )
        / body_scale
    )
    shoulder_to_hip = float((shoulder - hip)[vertical_axis] / body_scale)
    left_knee_angle = _angle_degrees(pose[11], pose[13], pose[15])
    right_knee_angle = _angle_degrees(pose[12], pose[14], pose[16])
    # 较低的脚踝对应当前支撑腿，抬腿或越障时不使用摆动腿约束站立。
    support_ids = (
        (11, 13, 15)
        if pose[15, vertical_axis] <= pose[16, vertical_axis]
        else (12, 14, 16)
    )
    support_hip, support_knee, support_ankle = (pose[index] for index in support_ids)

    return _PoseFeatures(
        height_ratio=height_extent / body_scale,
        horizontal_ratio=horizontal_extent / body_scale,
        torso_angle=_angle_from_vertical(shoulder - hip, vertical_axis),
        body_angle=_angle_from_vertical(pose[0] - ankle, vertical_axis),
        hip_to_ankle=hip_to_ankle,
        hip_to_knee=hip_to_knee,
        knee_to_ankle=knee_to_ankle,
        min_knee_to_ankle=min_knee_to_ankle,
        shoulder_to_hip=shoulder_to_hip,
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        mean_knee_angle=0.5 * (left_knee_angle + right_knee_angle),
        support_hip_to_ankle=float(
            (support_hip - support_ankle)[vertical_axis] / body_scale
        ),
        support_knee_to_ankle=float(
            (support_knee - support_ankle)[vertical_axis] / body_scale
        ),
        support_knee_angle=_angle_degrees(
            support_hip,
            support_knee,
            support_ankle,
        ),
    )


def _classify_pose(pose: np.ndarray, rules: ActionRules) -> int:
    features = _extract_features(pose, rules)
    if features is None:
        return OTHER

    stand = rules.stand
    standard_stand = (
        features.height_ratio >= stand.height_ratio_min
        and features.torso_angle <= stand.torso_angle_max
        and features.shoulder_to_hip >= stand.shoulder_to_hip_min
        and features.support_hip_to_ankle >= stand.support_hip_to_ankle_min
        and features.support_knee_to_ankle >= stand.support_knee_to_ankle_min
        and features.support_knee_angle >= stand.support_knee_angle_min
    )
    walking_step = (
        features.height_ratio >= stand.step_height_ratio_min
        and features.torso_angle <= stand.torso_angle_max
        and features.shoulder_to_hip >= stand.shoulder_to_hip_min
        and stand.step_hip_to_ankle_min
        <= features.hip_to_ankle
        <= stand.step_hip_to_ankle_max
        and abs(features.left_knee_angle - features.right_knee_angle)
        >= stand.step_knee_angle_difference_min
    )

    sit = rules.sit_squat
    sit_base = (
        features.height_ratio >= sit.height_ratio_min
        and features.shoulder_to_hip >= sit.shoulder_to_hip_min
        and features.hip_to_ankle >= sit.hip_to_ankle_min
        and features.torso_angle <= sit.torso_angle_max
    )
    standard_sit = (
        sit_base
        and features.height_ratio <= sit.seated_height_ratio_max
        and features.hip_to_ankle <= sit.seated_hip_to_ankle_max
        and features.hip_to_knee <= sit.seated_hip_to_knee_max
        and features.min_knee_to_ankle >= sit.seated_min_knee_to_ankle_min
        and max(features.left_knee_angle, features.right_knee_angle)
        <= sit.seated_max_knee_angle_max
    )
    standard_squat = (
        sit_base
        and features.height_ratio <= sit.squat_height_ratio_max
        and features.hip_to_ankle <= sit.squat_hip_to_ankle_max
        and features.body_angle <= sit.squat_body_angle_max
        and features.mean_knee_angle <= sit.squat_mean_knee_angle_max
        and features.knee_to_ankle >= sit.squat_knee_to_ankle_min
    )

    lie = rules.lie
    standard_lie = (
        features.horizontal_ratio >= lie.horizontal_ratio_min
        and features.torso_angle >= lie.torso_angle_min
        and features.hip_to_ankle <= lie.hip_to_ankle_max
    )

    # 躺姿优先，避免曲腿躺姿因膝关节弯曲被坐蹲条件抢占。
    if standard_lie:
        return LIE
    if standard_stand or walking_step:
        return STAND
    if standard_sit or standard_squat:
        return SIT_SQUAT
    return OTHER


def classify_actions(
    poses: np.ndarray,
    rules: ActionRules = DEFAULT_ACTION_RULES,
) -> ActionResult:
    """判决任意前导维度的 COCO-17 三维姿态并保持输入维度对齐。"""
    pose_array = np.asarray(poses, dtype=np.float64)
    if pose_array.ndim < 2 or pose_array.shape[-2:] != (17, 3):
        raise ValueError(f"poses must end with shape 17 x 3, got {pose_array.shape}")
    if rules.vertical_axis not in (0, 1, 2):
        raise ValueError("rules.vertical_axis must be 0, 1, or 2")

    # 展平 B/T/N 等前导维度，逐人体判决后再恢复原始批量形状。
    leading_shape = pose_array.shape[:-2]
    flat_poses = pose_array.reshape(-1, 17, 3)
    flat_valid = np.isfinite(flat_poses).all(axis=(1, 2))
    flat_labels = np.zeros((len(flat_poses), len(LABEL_NAMES)), dtype=np.float32)
    flat_labels[:, OTHER] = 1.0
    for pose_index in np.flatnonzero(flat_valid):
        label_index = _classify_pose(flat_poses[pose_index], rules)
        flat_labels[pose_index] = 0.0
        flat_labels[pose_index, label_index] = 1.0

    return ActionResult(
        labels=flat_labels.reshape(*leading_shape, len(LABEL_NAMES)),
        valid=flat_valid.reshape(leading_shape),
    )


__all__ = [
    "ActionResult",
    "ActionRules",
    "DEFAULT_ACTION_RULES",
    "LABEL_NAMES",
    "LieRules",
    "SitSquatRules",
    "StandRules",
    "classify_actions",
]
