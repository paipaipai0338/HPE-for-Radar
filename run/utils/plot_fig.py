from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from metrics.detection import paired_axis_aligned_iou_3d
from utils.COCO import COCO_SKELETON


BOX_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)

DEFAULT_ACTION_LABELS = (
    'stand',
    'sit_squat',
    'lie',
    'other',
)


def _to_numpy(value):
    if value is None:
        return None
    return value.detach().cpu().numpy()


def _draw_action_legend(
    ax,
    actions,
    mask,
    prefix,
    action_labels,
    color,
):
    """在图例中显示没有对应 pose 可绘制的行为类别。"""
    if actions is None:
        return
    if actions.ndim != 2 or actions.shape[-1] != len(action_labels):
        raise ValueError(
            f'{prefix} action must be [P,{len(action_labels)}], '
            f'got {actions.shape}'
        )
    for person_idx, action in enumerate(actions):
        if mask is not None and not mask[person_idx]:
            continue
        class_idx = int(np.argmax(action))
        class_name = action_labels[class_idx]
        ax.scatter(
            [],
            [],
            [],
            color=color,
            label=f'{prefix} person {person_idx} action {class_name}',
        )


def _draw_pose(
    ax,
    poses,
    mask,
    color,
    prefix,
    mpjpe=None,
    actions=None,
    action_labels=DEFAULT_ACTION_LABELS,
):
    if poses is None:
        return
    if actions is not None and poses.shape[0] != actions.shape[0]:
        raise ValueError(
            f'{prefix} pose/action person dimensions differ: '
            f'pose={poses.shape[0]}, action={actions.shape[0]}'
        )

    for person_idx, joints in enumerate(poses):
        if mask is not None and not mask[person_idx]:
            continue

        label = f'{prefix} person {person_idx}'
        if actions is not None:
            class_idx = int(np.argmax(actions[person_idx]))
            label += f' action {action_labels[class_idx]}'
        if mpjpe is not None:
            label += f' MPJPE={mpjpe[person_idx]:.3f} m'
        ax.scatter(
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
            s=5,
            color=color,
            label=label,
        )
        for joint_a, joint_b in COCO_SKELETON:
            ax.plot(
                [joints[joint_a, 0], joints[joint_b, 0]],
                [joints[joint_a, 1], joints[joint_b, 1]],
                [joints[joint_a, 2], joints[joint_b, 2]],
                color=color,
                linewidth=1.5,
            )


def _draw_split_point_cloud(
    ax,
    points,
    mask,
    bboxes,
    inside_label,
    outside_label,
):
    if points is None or mask is None:
        return

    valid_points = points[mask.astype(bool)]
    point_xyz = valid_points[:, :3]
    point_xyz = point_xyz[np.isfinite(point_xyz).all(axis=-1)]
    if point_xyz.shape[0] == 0:
        return

    inside = np.zeros(point_xyz.shape[0], dtype=bool)
    if bboxes is not None:
        for bbox in bboxes:
            inside |= (
                (point_xyz >= bbox[:3]).all(axis=-1)
                & (point_xyz <= bbox[3:]).all(axis=-1)
            )

    outside_xyz = point_xyz[~inside]
    if outside_xyz.shape[0] > 0:
        ax.scatter(
            outside_xyz[:, 0],
            outside_xyz[:, 1],
            outside_xyz[:, 2],
            s=5,
            color='gray',
            alpha=0.35,
            label=outside_label,
        )

    inside_xyz = point_xyz[inside]
    if inside_xyz.shape[0] > 0:
        ax.scatter(
            inside_xyz[:, 0],
            inside_xyz[:, 1],
            inside_xyz[:, 2],
            s=5,
            color='green',
            alpha=0.8,
            label=inside_label,
        )


def _draw_bbox(
    ax,
    bboxes,
    mask,
    color,
    prefix,
    objectness_score=None,
    ious=None,
):
    if bboxes is None:
        return

    for person_idx, bbox in enumerate(bboxes):
        if mask is not None and not mask[person_idx]:
            continue

        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        label = f'{prefix} bbox'
        if objectness_score is not None:
            label += f' confidence={objectness_score[person_idx]:.3f}'
        if ious is not None:
            label += f' IoU={ious[person_idx]:.3f}'
        corners = (
            (x_min, y_min, z_min),
            (x_max, y_min, z_min),
            (x_min, y_max, z_min),
            (x_max, y_max, z_min),
            (x_min, y_min, z_max),
            (x_max, y_min, z_max),
            (x_min, y_max, z_max),
            (x_max, y_max, z_max),
        )
        for edge_idx, (start, end) in enumerate(BOX_EDGES):
            start_point = corners[start]
            end_point = corners[end]
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                linewidth=1.5,
                label=(
                    label
                    if edge_idx == 0
                    else None
                ),
            )


def _set_axis(ax, title):
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(-3.0, 3.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique_legend = dict(zip(labels, handles))
        ax.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc='upper right',
        )


def plt_fig(
    fig_path,
    pre,
    gt,
    model_input,
    matches=None,
    horizontal=False,
    dpi=400,
):
    """绘制 pose 或 bbox；缺失的预测类型使用对应 GT 补充显示。"""
    pose_pre = _to_numpy(pre.get('pose'))
    pose_pre_mask = _to_numpy(pre.get('mask'))
    bbox_pre = _to_numpy(pre.get('bbox'))
    action_logits = _to_numpy(pre.get('action_logits'))
    objectness_logits = pre.get('objectness_logits')
    objectness_score = _to_numpy(
        None
        if objectness_logits is None
        else objectness_logits.sigmoid()
    )

    pose_gt = _to_numpy(gt['padded'])
    bbox_gt = _to_numpy(gt['bbox'])
    action_gt = _to_numpy(gt.get('action'))
    action_labels = tuple(
        gt.get('action_label') or DEFAULT_ACTION_LABELS
    )
    gt_mask = _to_numpy(gt['mask'])

    if action_gt is not None:
        if (
            action_gt.ndim != 4
            or action_gt.shape[:3] != pose_gt.shape[:3]
            or action_gt.shape[-1] != len(action_labels)
        ):
            raise ValueError(
                'gt action must be [B,T,K,C] and align with pose GT, '
                f'got action={action_gt.shape}, pose={pose_gt.shape}'
            )
    if action_logits is not None:
        if (
            action_logits.ndim != 4
            or action_logits.shape[:2] != pose_gt.shape[:2]
            or action_logits.shape[-1] != len(action_labels)
        ):
            raise ValueError(
                'pred action_logits must be [B,T,Q,C], '
                f'got {action_logits.shape}'
            )

    input_data = model_input.get('input')
    input_mask = model_input.get('mask')
    if input_mask is not None:
        if input_data is None:
            raise ValueError(
                "model_input['mask'] exists but model_input['input'] "
                "is missing"
            )
        if input_data.ndim != 4 or input_data.shape[-1] < 3:
            raise ValueError(
                "masked model_input['input'] must be [B, T, N, C] "
                f"with C >= 3, got {tuple(input_data.shape)}"
            )
        if input_mask.ndim != 3:
            raise ValueError(
                "model_input['mask'] must be [B, T, N], "
                f"got {tuple(input_mask.shape)}"
            )
        if input_data.shape[:3] != input_mask.shape:
            raise ValueError(
                "model_input input/mask B,T,N dimensions differ"
            )
        point_cloud = _to_numpy(input_data)
        point_mask = _to_numpy(input_mask)
    else:
        point_cloud = None
        point_mask = None

    if bbox_pre is not None and objectness_score is None:
        raise ValueError(
            "pre['bbox'] exists but pre['objectness_logits'] is missing"
        )
    if bbox_pre is None and objectness_score is not None:
        raise ValueError(
            "pre['objectness_logits'] exists but pre['bbox'] is missing"
        )
    if (
        bbox_pre is not None
        and bbox_pre.shape[:-1] != objectness_score.shape
    ):
        raise ValueError(
            "pre['bbox'] and pre['objectness_logits'] B,T,Q "
            "dimensions differ"
        )
    if bbox_pre is not None and matches is None:
        raise ValueError(
            "matches is required when pre['bbox'] exists"
        )

    available = (pose_gt, bbox_gt, pose_pre, bbox_pre)
    reference = next((value for value in available if value is not None), None)
    if reference is None:
        raise ValueError(
            'plot_fig requires at least one of pose or bbox in pre/gt'
        )

    batch_idx = 0
    num_frames = reference.shape[1]
    fig = plt.figure(
        figsize=(15, 5 * num_frames)
        if horizontal
        else (5 * num_frames, 16)
    )

    for time_idx in range(num_frames):
        current_gt_mask = gt_mask[batch_idx, time_idx].astype(bool)
        current_pose_gt = pose_gt[batch_idx, time_idx]
        current_bbox_gt = bbox_gt[batch_idx, time_idx]
        current_action_gt = (
            None
            if action_gt is None
            else action_gt[batch_idx, time_idx]
        )
        current_action_logits = (
            None
            if action_logits is None
            else action_logits[batch_idx, time_idx]
        )
        valid_bbox_gt = current_bbox_gt[current_gt_mask]
        current_points = (
            None
            if point_cloud is None
            else point_cloud[batch_idx, time_idx]
        )
        current_point_mask = (
            None
            if point_mask is None
            else point_mask[batch_idx, time_idx]
        )

        # 第一行：完整 GT，以及相对于 GT bbox 的点云内外分布。
        gt_ax = fig.add_subplot(
            num_frames if horizontal else 3,
            3 if horizontal else num_frames,
            time_idx * 3 + 1 if horizontal else time_idx + 1,
            projection='3d',
        )
        _draw_split_point_cloud(
            gt_ax,
            current_points,
            current_point_mask,
            valid_bbox_gt,
            inside_label='Points inside GT bbox',
            outside_label='Points outside GT bbox',
        )
        _draw_pose(
            gt_ax,
            current_pose_gt,
            current_gt_mask,
            color='red',
            prefix='GT',
            actions=current_action_gt,
            action_labels=action_labels,
        )
        _draw_bbox(
            gt_ax,
            current_bbox_gt,
            current_gt_mask,
            color='orange',
            prefix='GT',
        )
        _set_axis(
            gt_ax,
            f'GT | Batch:{batch_idx}, Time:{time_idx}',
        )

        # 第二行：匈牙利匹配的预测 bbox，以及相对于预测框的点云分布。
        bbox_ax = fig.add_subplot(
            num_frames if horizontal else 3,
            3 if horizontal else num_frames,
            (
                time_idx * 3 + 2
                if horizontal
                else time_idx + num_frames + 1
            ),
            projection='3d',
        )

        if bbox_pre is not None:
            if time_idx >= len(matches):
                raise ValueError(
                    f"Expected at least {num_frames} frame matches, "
                    f"got {len(matches)}"
                )
            pred_idx, gt_idx = matches[time_idx]
            matched_pred_bbox = pre['bbox'][
                batch_idx, time_idx, pred_idx
            ]
            matched_gt_bbox = gt['bbox'][
                batch_idx, time_idx, gt_idx
            ]
            matched_score = pre['objectness_logits'][
                batch_idx, time_idx, pred_idx
            ].sigmoid()
            matched_iou = paired_axis_aligned_iou_3d(
                matched_pred_bbox,
                matched_gt_bbox,
            )
            matched_pred_bbox_numpy = _to_numpy(matched_pred_bbox)
            _draw_split_point_cloud(
                bbox_ax,
                current_points,
                current_point_mask,
                matched_pred_bbox_numpy,
                inside_label='Points inside Pred bbox',
                outside_label='Points outside Pred bbox',
            )
            _draw_bbox(
                bbox_ax,
                matched_pred_bbox_numpy,
                mask=None,
                color='lightskyblue',
                prefix='Pred',
                objectness_score=_to_numpy(matched_score),
                ious=_to_numpy(matched_iou),
            )
        else:
            _draw_split_point_cloud(
                bbox_ax,
                current_points,
                current_point_mask,
                bboxes=None,
                inside_label='Points inside Pred bbox',
                outside_label='Points outside Pred bbox',
            )
        _set_axis(
            bbox_ax,
            f'BBox Prediction | Batch:{batch_idx}, Time:{time_idx}',
        )

        # 第三行：GT pose 与预测 pose 的直接对比。
        pose_ax = fig.add_subplot(
            num_frames if horizontal else 3,
            3 if horizontal else num_frames,
            (
                time_idx * 3 + 3
                if horizontal
                else time_idx + 2 * num_frames + 1
            ),
            projection='3d',
        )
        _draw_pose(
            pose_ax,
            current_pose_gt,
            current_gt_mask,
            color='red',
            prefix='GT',
            actions=current_action_gt,
            action_labels=action_labels,
        )
        if pose_pre is not None:
            current_pose_pre = pose_pre[batch_idx, time_idx]
            pred_action_mask = (
                pose_pre_mask[batch_idx, time_idx].astype(bool)
                if pose_pre_mask is not None
                else (
                    current_gt_mask
                    if current_pose_pre.shape[0] == current_gt_mask.shape[0]
                    else None
                )
            )
            pose_mpjpe = np.linalg.norm(
                current_pose_pre - current_pose_gt,
                axis=-1,
            ).mean(axis=-1)
            _draw_pose(
                pose_ax,
                current_pose_pre,
                pred_action_mask,
                color='blue',
                prefix='Pred',
                mpjpe=pose_mpjpe,
                actions=current_action_logits,
                action_labels=action_labels,
            )
        elif current_action_logits is not None:
            if bbox_pre is not None:
                pred_idx, gt_idx = matches[time_idx]
                _draw_action_legend(
                    pose_ax,
                    current_action_logits[pred_idx],
                    mask=None,
                    prefix='Pred',
                    action_labels=action_labels,
                    color='darkblue',
                )
            elif current_action_logits.shape[0] == current_pose_gt.shape[0]:
                _draw_action_legend(
                    pose_ax,
                    current_action_logits,
                    current_gt_mask,
                    prefix='Pred',
                    action_labels=action_labels,
                    color='darkblue',
                )
        _set_axis(
            pose_ax,
            f'Pose Comparison | Batch:{batch_idx}, Time:{time_idx}',
        )

    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
