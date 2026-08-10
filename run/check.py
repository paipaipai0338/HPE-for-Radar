from pathlib import Path
import time
import torch


from metrics.pose import get_bone_length, get_mpjpe, get_pampjpe
from run.utils.load_config import load_config
from run.utils.plot_fig import plt_fig


result_path = Path('/home/pai/Huawei/run/result.pkl')
config_path = Path('/home/pai/Huawei/run/config.yaml')
fig_path = Path('/home/pai/Huawei/temp')

loaded = torch.load(result_path, map_location='cpu')
cfg = load_config(config_path)
point_cloud_range = cfg['data']['point_cloud_range']
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
        point_cloud_range,
        bbox_l1_weight=matching_cfg['bbox_l1_weight'],
        bbox_iou_weight=matching_cfg['bbox_iou_weight'],
    )
    objectness = get_objectness(objectness_logits, gt_mask, matches)
    bbox_l1 = get_bbox_l1(bbox_pre, bbox_gt, matches, point_cloud_range)
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

num_frames = pose_gt.shape[1]
for sample_idx in range(pose_gt.shape[0]):
    pre = {
        'pose': (
            None
            if pose_pre is None
            else pose_pre[sample_idx:sample_idx + 1]
        ),
        'bbox': (
            None
            if bbox_pre is None
            else bbox_pre[sample_idx:sample_idx + 1]
        ),
        'objectness_logits': (
            None
            if objectness_logits is None
            else objectness_logits[sample_idx:sample_idx + 1]
        ),
        'action_logits': (
            None
            if action_logits is None
            else action_logits[sample_idx:sample_idx + 1]
        ),
    }
    gt = {
        'padded': pose_gt[sample_idx:sample_idx + 1],
        'bbox': bbox_gt[sample_idx:sample_idx + 1],
        'mask': gt_valid[sample_idx:sample_idx + 1],
    }
    if action_gt is not None:
        gt['action'] = action_gt[sample_idx:sample_idx + 1]
    if action_label is not None:
        gt['action_label'] = action_label
    model_input = {
        'input': pc[sample_idx:sample_idx + 1],
    }
    if pc_valid is not None:
        model_input['mask'] = pc_valid[sample_idx:sample_idx + 1]
    sample_matches = (
        None
        if matches is None
        else matches[
            sample_idx * num_frames:(sample_idx + 1) * num_frames
        ]
    )
    plt_fig(
        fig_path / f'temp_{sample_idx % 10}.png',
        pre,
        gt,
        model_input,
        matches=sample_matches,
    )
    print(
        f'\rFigure: {sample_idx + 1}/{pose_gt.shape[0]} | '
        f'{fig_path}',
        end='',
        flush=True,
    )
