from pathlib import Path
import time
import torch


from metrics.pose import get_bone_length, get_mpjpe, get_pampjpe
from metrics.detection import get_bbox_iou, get_bbox_l1, get_objectness, get_hungarian_match, get_tp_fp_fn_tn, get_precision, get_recall, get_acc
from run.utils.load_config import load_config
from run.utils.plot_fig import plt_fig
from utils.COCO import COCO_NAMES


def transform_points_high_to_low(points, rotation, translation):
    """将 [B,T,...,3] 点坐标从 high 雷达系变换到 low 雷达系。"""
    if points is None:
        return None
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError(
            'points 应为 [B,T,...,3]，'
            f'实际 shape={tuple(points.shape)}'
        )
    if rotation.shape != (*points.shape[:2], 3, 3):
        raise ValueError(
            'high_to_low_R 应为 [B,T,3,3] 且与 points 对齐，'
            f'points={tuple(points.shape)}, R={tuple(rotation.shape)}'
        )
    if translation.shape != (*points.shape[:2], 3):
        raise ValueError(
            'high_to_low_t 应为 [B,T,3] 且与 points 对齐，'
            f'points={tuple(points.shape)}, t={tuple(translation.shape)}'
        )

    transformed = torch.einsum(
        'bt...c,btdc->bt...d',
        points,
        rotation.to(dtype=points.dtype, device=points.device),
    )
    translation = translation.to(
        dtype=points.dtype,
        device=points.device,
    )
    translation_shape = (
        *translation.shape[:2],
        *((1,) * (points.ndim - 3)),
        3,
    )
    return transformed + translation.reshape(translation_shape)


def transform_pointcloud_high_to_low(pointcloud, rotation, translation):
    """仅变换点云最后一维中的 xyz，保留其余点特征。"""
    if pointcloud is None:
        return None
    if pointcloud.ndim < 4 or pointcloud.shape[-1] < 3:
        raise ValueError(
            'pointcloud 应为 [B,T,...,N,C] 且 C>=3，'
            f'实际 shape={tuple(pointcloud.shape)}'
        )
    transformed = pointcloud.clone()
    transformed[..., :3] = transform_points_high_to_low(
        pointcloud[..., :3],
        rotation,
        translation,
    )
    return transformed


def transform_bbox_high_to_low(bbox, rotation, translation):
    """变换 bbox 的 8 个角点并生成 low 雷达系下的轴对齐框。"""
    if bbox is None:
        return None
    if bbox.ndim < 3 or bbox.shape[-1] != 6:
        raise ValueError(
            'bbox 应为 [B,T,...,6]，'
            f'实际 shape={tuple(bbox.shape)}'
        )

    bbox_min = bbox[..., :3]
    bbox_max = bbox[..., 3:]
    corners = torch.stack(
        [
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_min[..., 2]), dim=-1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_min[..., 2]), dim=-1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_min[..., 2]), dim=-1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_min[..., 2]), dim=-1),
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_max[..., 2]), dim=-1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_max[..., 2]), dim=-1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_max[..., 2]), dim=-1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_max[..., 2]), dim=-1),
        ],
        dim=-2,
    )
    transformed_corners = transform_points_high_to_low(
        corners,
        rotation,
        translation,
    )
    return torch.cat(
        (
            transformed_corners.amin(dim=-2),
            transformed_corners.amax(dim=-2),
        ),
        dim=-1,
    )


result_path = Path('/home/pai/Huawei/run/result.pkl')
config_path = Path('/home/pai/Huawei/run/config.yaml')
fig_path = Path('/home/pai/Huawei/temp')
catastrophic_mpjpe_threshold = 0.4
max_catastrophic_figures = 100

loaded = torch.load(result_path, map_location='cpu')
cfg = load_config(config_path)
xyz_limits = cfg['data']['xyz_limits']
matching_cfg = cfg['task']['matching_for_hungarian']
score_thresh = cfg['task']['val']['score_thresh']
iou_thresh = cfg['task']['val']['iou_thresh']

pc = loaded['pc']
pc_valid = loaded['pc_valid']
pose_pre = loaded['pose_pre']
bbox_pre = loaded['bbox_pre']
objectness_logits = loaded['objectness_logits']
action_logits = loaded.get('action_logits')
pose_gt = loaded['pose_gt']
bbox_gt = loaded['bbox_gt']
action_gt = loaded.get('action_gt')
action_label = loaded.get('action_label')
gt_valid = loaded['gt_valid']
high_to_low_R = loaded['high_to_low_R']
high_to_low_t = loaded['high_to_low_t']
input_key = loaded.get('input_key')
target_key = loaded.get('target_key')

print('\nShape')
print('input_key:', input_key)
print('target_key:', target_key)
print('pc:', pc.shape)
print('pc_valid:', pc_valid.shape)
print('pose_pre:', None if pose_pre is None else pose_pre.shape)
print('bbox_pre:', None if bbox_pre is None else bbox_pre.shape)
print('objectness_logits:', None if objectness_logits is None else objectness_logits.shape)
print('action_logits:', None if action_logits is None else action_logits.shape)
print('pose_gt:', pose_gt.shape)
print('bbox_gt:', bbox_gt.shape)
print('action_gt:', None if action_gt is None else action_gt.shape)
print('action_label:', action_label)
print('gt_valid:', gt_valid.shape)
print('high_to_low_R:', None if high_to_low_R is None else high_to_low_R.shape,)
print('high_to_low_t:', None if high_to_low_t is None else high_to_low_t.shape,)

gt_mask = gt_valid.bool()
gt_num = int(gt_mask.sum().item())
matches = None

if action_gt is not None:
    if action_gt.shape[:-1] != gt_mask.shape or action_gt.shape[-1] != 4:
        raise ValueError(
            "action_gt 应为 [B,T,K,4] 且与 gt_valid 对齐，"
            f"action_gt={tuple(action_gt.shape)}, "
            f"gt_valid={tuple(gt_valid.shape)}"
        )

    valid_action = action_gt[gt_mask]
    action_indices = valid_action.argmax(dim=-1)
    labels = (
        action_label
        if action_label is not None
        else ['stand', 'sit_squat', 'lie', 'other']
    )

    print('\nAction GT distribution')
    for class_idx, class_name in enumerate(labels):
        class_count = int(
            (action_indices == class_idx).sum().item()
        )
        print(f'{class_name}: {class_count}')

if bbox_pre is not None and objectness_logits is not None:
    matches = get_hungarian_match(
        bbox_pre,
        bbox_gt,
        gt_mask,
        xyz_limits,
        bbox_l1_weight=matching_cfg['bbox_l1_weight'],
        bbox_iou_weight=matching_cfg['bbox_iou_weight'],
    )
    objectness = get_objectness(objectness_logits, gt_mask, matches)
    bbox_l1 = get_bbox_l1(bbox_pre, bbox_gt, matches, xyz_limits)
    bbox_iou_loss = get_bbox_iou(bbox_pre, bbox_gt, matches)

    objectness_mean = objectness.mean().item()
    if gt_num > 0:
        bbox_l1_mean = bbox_l1.masked_select(gt_mask).mean().item()
        bbox_iou_loss_mean = (
            bbox_iou_loss.masked_select(gt_mask).mean().item()
        )
        bbox_iou_mean = 1.0 - bbox_iou_loss_mean
    else:
        bbox_l1_mean = float('nan')
        bbox_iou_loss_mean = float('nan')
        bbox_iou_mean = float('nan')

    tp, fp, fn, tn = get_tp_fp_fn_tn(objectness_logits, gt_mask, bbox_pre, bbox_gt, matches, score_thresh, iou_thresh)
    precision = get_precision(tp, fp, fn, tn)
    recall = get_recall(tp, fp, fn, tn)
    acc = get_acc(tp, fp, fn, tn)

    print('\nBBox metrics')
    print(f'objectness: {objectness_mean:.6f}')
    print(f'bbox_l1: {bbox_l1_mean:.6f}')
    print(f'bbox_iou: {bbox_iou_mean:.6f}')
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'Accuracy: {acc:.6f}')

if pose_pre is not None:
    mpjpe = get_mpjpe(pose_pre, pose_gt, type='coco')
    pampjpe = get_pampjpe(pose_pre, pose_gt, type='coco')
    bone_length = get_bone_length(
        pose_pre,
        pose_gt,
        type='coco',
    )

    if gt_num > 0:
        mpjpe_mean = mpjpe.masked_select(gt_mask).mean().item()
        pampjpe_mean = pampjpe.masked_select(gt_mask).mean().item()
        bone_length_mean = (
            bone_length.masked_select(gt_mask).mean().item()
        )
    else:
        mpjpe_mean = float('nan')
        pampjpe_mean = float('nan')
        bone_length_mean = float('nan')

    print('\nPose metrics')
    print(f'mpjpe: {mpjpe_mean:.6f}')
    print(f'pampjpe: {pampjpe_mean:.6f}')
    print(f'bone_length: {bone_length_mean:.6f}')

    torso_joint_names = {
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_hip',
        'right_hip',
    }
    torso_joint_indices = [
        joint_idx
        for joint_idx, joint_name in enumerate(COCO_NAMES)
        if joint_name in torso_joint_names
    ]
    limb_joint_indices = [
        joint_idx
        for joint_idx, joint_name in enumerate(COCO_NAMES)
        if joint_name not in torso_joint_names
    ]
    per_joint_error = torch.norm(pose_pre - pose_gt, dim=-1)
    valid_per_joint_error = per_joint_error[gt_mask]
    if gt_num > 0:
        per_joint_mpjpe = valid_per_joint_error.mean(dim=0)
        torso_mpjpe = valid_per_joint_error[
            :, torso_joint_indices
        ].mean()
        limb_mpjpe = valid_per_joint_error[
            :, limb_joint_indices
        ].mean()
        print('\nMPJPE by body region')
        print(f'torso: {torso_mpjpe.item():.6f}')
        print(f'limbs: {limb_mpjpe.item():.6f}')
    else:
        print('\nMPJPE by joint: no valid GT person')
        print('MPJPE by body region: no valid GT person')

    # TODO: 后续可在这里继续细分灾难样本类型（躺卧、稀疏点云等）。
    # 当前先按单人单帧 MPJPE 选择灾难样本，并按误差从高到低保存，
    # 供下方可视化使用。
    catastrophic_mask = (
        gt_mask
        & (mpjpe >= catastrophic_mpjpe_threshold)
    )
    catastrophic_indices = catastrophic_mask.nonzero(as_tuple=False)
    if catastrophic_indices.numel() > 0:
        catastrophic_errors = mpjpe[catastrophic_mask]
        catastrophic_order = torch.argsort(
            catastrophic_errors,
            descending=True,
        )
        if max_catastrophic_figures is not None:
            catastrophic_order = catastrophic_order[
                :max_catastrophic_figures
            ]
        catastrophic_indices = catastrophic_indices[
            catastrophic_order
        ]
        catastrophic_errors = catastrophic_errors[
            catastrophic_order
        ]
    else:
        catastrophic_errors = torch.empty(dtype=mpjpe.dtype)

    print('\nCatastrophic samples')
    print(
        f'threshold: {catastrophic_mpjpe_threshold:.3f} m | '
        f'total: {int(catastrophic_mask.sum())} | '
        f'to visualize: {len(catastrophic_indices)}'
    )
else:
    catastrophic_indices = torch.empty((0, 3), dtype=torch.long)
    catastrophic_errors = torch.empty(dtype=torch.float32)

# 指标仍在模型原本的 high 雷达坐标系中计算；仅将传给 plt_fig 的
# 几何数据转换到 low 雷达坐标系。
if high_to_low_R is None or high_to_low_t is None:
    raise ValueError(
        '绘图坐标转换需要 result.pkl 中同时包含 '
        'high_to_low_R 和 high_to_low_t'
    )

num_frames = pose_gt.shape[1]
catastrophic_fig_path = fig_path / 'catastrophic'
catastrophic_fig_path.mkdir(parents=True, exist_ok=True)
for figure_idx, (catastrophic_index, catastrophic_error) in enumerate(
    zip(catastrophic_indices, catastrophic_errors)
):
    sample_idx, time_idx, person_idx = catastrophic_index.tolist()
    sample_slice = slice(sample_idx, sample_idx + 1)
    time_slice = slice(time_idx, time_idx + 1)
    rotation = high_to_low_R[sample_slice, time_slice]
    translation = high_to_low_t[sample_slice, time_slice]
    plot_pc = transform_pointcloud_high_to_low(
        pc[sample_slice, time_slice], rotation, translation,
    )
    plot_pose_pre = transform_points_high_to_low(
        pose_pre[sample_slice, time_slice], rotation, translation,
    )
    plot_pose_gt = transform_points_high_to_low(
        pose_gt[sample_slice, time_slice], rotation, translation,
    )
    plot_bbox_pre = transform_bbox_high_to_low(
        None if bbox_pre is None else bbox_pre[sample_slice, time_slice],
        rotation,
        translation,
    )
    plot_bbox_gt = transform_bbox_high_to_low(
        bbox_gt[sample_slice, time_slice], rotation, translation,
    )

    pre = {
        'pose': plot_pose_pre,
        'bbox': plot_bbox_pre,
        'objectness_logits': (
            None
            if objectness_logits is None
            else objectness_logits[sample_slice, time_slice]
        ),
        'action_logits': (
            None
            if action_logits is None
            else action_logits[sample_slice, time_slice]
        ),
    }
    gt = {
        'padded': plot_pose_gt,
        'bbox': plot_bbox_gt,
        'mask': gt_valid[sample_slice, time_slice],
    }
    if action_gt is not None:
        gt['action'] = action_gt[sample_slice, time_slice]
    if action_label is not None:
        gt['action_label'] = action_label
    model_input = {
        'input': plot_pc,
    }
    if pc_valid is not None:
        model_input['mask'] = pc_valid[sample_slice, time_slice]
    sample_matches = (
        None
        if matches is None
        else matches[
            sample_idx * num_frames + time_idx:
            sample_idx * num_frames + time_idx + 1
        ]
    )
    error_mm = round(float(catastrophic_error) * 1000)
    plt_fig(
        catastrophic_fig_path / (
            f'rank_{figure_idx:03d}_sample_{sample_idx}_'
            f'frame_{time_idx}_person_{person_idx}_'
            f'mpjpe_{error_mm}mm.png'
        ),
        pre,
        gt,
        model_input,
        matches=sample_matches,
    )
    print(
        f'\rFigure: {figure_idx + 1}/{len(catastrophic_indices)} | '
        f'{catastrophic_fig_path}',
        end='',
        flush=True,
    )
