import json
from pathlib import Path
import torch
from metrics.pose import (
    apply_pose_matches,
    get_bce,
    get_bone_length,
    get_center_mpjpe,
    get_mpjpe,
    get_pampjpe,
    get_pose_direction,
    get_pose_hungarian_match,
    get_relative_pose_mpjpe,
    get_time_smoothness,
)
from metrics.detection import get_hungarian_match, get_bbox_iou, get_bbox_l1, get_objectness
from metrics.RPM2_loss import get_center_heatmap_loss, get_box_size_loss, get_center_offset_loss, get_pose_loss
from metrics.HRRadarPose_loss import get_body_center_loss, get_keypoint_offset_loss
from utils.skeleton import get_pose_layout

def _masked_mean_over_people(metric, mask):
    # metric, mask: [B, T, K]
    if metric.shape != mask.shape:
        raise ValueError(
            f"metric and mask must have same shape, "
            f"got metric={tuple(metric.shape)}, mask={tuple(mask.shape)}"
        )

    mask = mask.to(device=metric.device, dtype=torch.bool)
    metric_num = mask.sum()

    if metric_num > 0:
        metric_sum = metric.masked_select(mask).sum()
        return metric_sum / metric_num, metric_num.item()

    return metric.masked_select(mask).sum() * 0.0, 0


class Metric:
    def __init__(
        self,
        cfg_metrics,
        xyz_limits,
        matching_bbox_l1_weight,
        matching_bbox_iou_weight,
        pose_matching_by_hip=False,
        pose_confidence_weight=0.0,
        pose_format='coco',
    ):
        self.pose_format = pose_format
        self.pose_layout = get_pose_layout(pose_format)
        if not isinstance(pose_matching_by_hip, bool):
            raise TypeError("pose_matching_by_hip must be bool")
        self.group_labels = json.loads(
            (Path(__file__).resolve().parents[2] / 'data2datasets/group_labels.json').read_text()
        )['groups']
        self.group_mpjpe = {}
        self.xyz_limits = xyz_limits
        self.matching_bbox_l1_weight = float(matching_bbox_l1_weight)
        self.matching_bbox_iou_weight = float(
            matching_bbox_iou_weight
        )
        self.pose_matching_by_hip = pose_matching_by_hip
        self.pose_confidence_weight = float(pose_confidence_weight)
        if self.pose_confidence_weight < 0:
            raise ValueError("pose_confidence_weight must be non-negative")
        self.fun_call_dict = {
            'positive_mpjpe': get_mpjpe,
            'all_gt_mpjpe': get_mpjpe,
            'center_mpjpe': get_center_mpjpe,
            'relative_pose_mpjpe': get_relative_pose_mpjpe,
            'negative_mpjpe': get_mpjpe,
            'pampjpe': get_pampjpe,
            'bone_length': get_bone_length,
            'pose_direction': get_pose_direction,
            'time_smoothness': get_time_smoothness,
            'bce': get_bce,
            'bbox_iou': get_bbox_iou,
            'bbox_l1':  get_bbox_l1,
            'objectness': get_objectness,
            'rpm2_center_heatmap': get_center_heatmap_loss,
            'rpm2_box_size': get_box_size_loss,
            'rpm2_center_offset': get_center_offset_loss,
            'rpm2_pose': get_pose_loss,
            'hrradarpose_body_center': get_body_center_loss,
            'hrradarpose_keypoint_offset': get_keypoint_offset_loss,
        }
        # 获取当前配置指标与权重
        self.cfg_metrics = {
            name.lower(): float(weight)
            for name, weight in cfg_metrics.items()
            if float(weight) != 0.0
        }

        # 检查是否匹配
        unsupported_metrics = set(self.cfg_metrics) - set(self.fun_call_dict)
        if unsupported_metrics:
            raise ValueError(
                f"存在未注册的指标: {sorted(unsupported_metrics)}。"
            )

        # 为每个epoch构建历史记录
        self.metrics_epoch_history = {
            name: []
            for name in self.cfg_metrics
        }

        # 记录当前指标状态
        self.metrics_state = {
            name: {'sum': 0.0, 'num': 0}
            for name in self.cfg_metrics
        }

    def state_dict(self):
        return {
            "metrics_epoch_history": self.metrics_epoch_history,
        }

    def load_state_dict(self, state_dict):
        # Keep initialized histories for metrics added since the checkpoint.
        self.metrics_epoch_history.update(
            state_dict.get("metrics_epoch_history", {})
        )
        
    def calculate_batch(self, pre, gt, group_ids=None, group_gt_low=None):
        # pre = {
        #     pose: pose_pre,                           # [B, T, K, J, 3]
        #     bbox,                                     # [B, T, K，6]
        #     objectness_logits,                        # [B, T, K]
        # }

        # gt = {
        #     padded,                                   # [B, T, K, J, 3]
        #     mask，                                    # [B, T, K]
        #     bbox，                                    # [B, T, K]
        # }
        pose_pre = pre.get('pose', None)
        confidence_pre = pre.get('confidence', None)
        bbox_pre = pre.get('bbox', None)
        objectness_logits_pre = pre.get('objectness_logits', None)

        pose_gt = gt.get('padded', None)
        gt_mask = gt.get('mask', None)
        bbox_gt = gt.get('bbox', None)
        

        batch_metrics = {}
        total_loss = 0.0
        bbox_matches = None
        pose_pre_for_metric = pose_pre
        confidence_target = gt_mask

        pose_metric_names = {
            'positive_mpjpe', 'all_gt_mpjpe', 'center_mpjpe', 'relative_pose_mpjpe',
            'negative_mpjpe',
            'pampjpe', 'bone_length', 'bce', 'time_smoothness', 'pose_direction'
        }
        pose_mask = gt_mask
        pose_valid = pre.get('pose_valid')
        if self.pose_matching_by_hip and pose_pre is not None and pose_gt is not None and (
            pose_metric_names & self.cfg_metrics.keys()
        ):
            if pose_pre is None or pose_gt is None or gt_mask is None:
                raise ValueError(
                    "hip pose matching requires pose prediction, pose GT and "
                    "GT mask"
                )
            pose_matches = get_pose_hungarian_match(
                pose_pre,
                pose_gt,
                gt_mask,
                confidence=confidence_pre,
                confidence_weight=self.pose_confidence_weight,
                hip_joint_indices=self.pose_layout['root_indices'],
                pose_valid=pose_valid,
            )
            pose_pre_for_metric, confidence_target = apply_pose_matches(
                pose_pre,
                pose_gt,
                pose_matches,
            )
            pose_mask = torch.zeros_like(gt_mask, dtype=torch.bool)
            for frame, (_, gt_indices) in enumerate(pose_matches):
                pose_mask.flatten(0, 1)[frame, gt_indices] = True
        elif pose_valid is not None and gt_mask is not None:
            if pose_valid.shape != gt_mask.shape:
                raise ValueError('pose_valid and GT mask must have the same shape')
            pose_mask = gt_mask.bool() & pose_valid.bool()

        for name, weight in self.cfg_metrics.items():
            if name in [
                'positive_mpjpe', 'all_gt_mpjpe', 'center_mpjpe',
                'relative_pose_mpjpe', 'pampjpe', 'bone_length'
            ]:
                if name == 'all_gt_mpjpe' and any(value is None for value in (pose_pre_for_metric, pose_gt, gt_mask)):
                    raise ValueError('all_gt_mpjpe requires predictions, GT poses and GT mask')
                if pose_pre_for_metric is None or pose_gt is None or pose_mask is None:
                    reference = pose_gt if pose_pre_for_metric is None else pose_pre_for_metric
                    batch_metrics[name] = torch.tensor(0.0, device=reference.device if reference is not None else 'cpu')
                    total_loss = total_loss + weight * batch_metrics[name]
                    continue
                evaluation_mask = gt_mask.bool() if name == 'all_gt_mpjpe' else pose_mask
                # Full-GT evaluation retains missing predictions instead of dropping their targets.
                metric = self.fun_call_dict[name](
                    pose_pre_for_metric.masked_fill(~evaluation_mask[..., None, None], 0),
                    pose_gt.masked_fill(~evaluation_mask[..., None, None], 0), type=self.pose_format
                )
                if group_ids is not None and (name == 'all_gt_mpjpe' or (
                        name == 'positive_mpjpe' and 'all_gt_mpjpe' not in self.cfg_metrics) or (
                        name == 'relative_pose_mpjpe'
                        and not {'all_gt_mpjpe', 'positive_mpjpe'} & self.cfg_metrics.keys())):
                    centered_errors = get_relative_pose_mpjpe(
                        pose_pre_for_metric.detach(), pose_gt.detach(), type=self.pose_format)
                    self._record_group_mpjpe(
                        metric.detach(), evaluation_mask, group_ids, group_gt_low, centered_errors)
                metric_value, metric_num = _masked_mean_over_people(
                    metric,
                    evaluation_mask,
                )
                self.metrics_state[name]['sum'] += (
                    metric_value.detach().item() * metric_num
                )
                self.metrics_state[name]['num'] += metric_num
            elif name == 'pose_direction':
                assert pose_pre is not None, 'pose_pre is None'
                metric = self.fun_call_dict[name](
                    pose_pre_for_metric, pose_gt, type=self.pose_format
                )
                metric_value, metric_num = _masked_mean_over_people(
                    metric, gt_mask.bool() & torch.isfinite(metric)
                )
            elif name == 'time_smoothness':
                assert pose_pre is not None, 'pose_pre is None'
                metric = self.fun_call_dict[name](
                    pose_pre_for_metric, pose_gt, type=self.pose_format
                )
                metric_value, metric_num = _masked_mean_over_people(
                    metric, gt_mask[:, 1:].bool() & gt_mask[:, :-1].bool()
                )
            elif name == 'negative_mpjpe':
                if pose_pre is None:
                    raise ValueError("negative_mpjpe requires pre['pose']")
                metric = self.fun_call_dict[name](
                    pose_pre,
                    torch.zeros_like(pose_pre),
                    type=self.pose_format,
                )
                metric_value, metric_num = _masked_mean_over_people(
                    metric,
                    ~confidence_target,
                )
            elif name in ['bbox_iou', 'bbox_l1', 'objectness']:
                assert bbox_pre is not None, 'bbox_pre is None'
                assert objectness_logits_pre is not None, 'objectness_logits_pre is None'
                if bbox_matches is None:
                    bbox_matches = get_hungarian_match(
                        bbox_pre,
                        bbox_gt,
                        gt_mask,
                        self.xyz_limits,
                        bbox_l1_weight=self.matching_bbox_l1_weight,
                        bbox_iou_weight=self.matching_bbox_iou_weight,
                    )

                if name == 'bbox_iou':
                    metric = self.fun_call_dict[name](
                        bbox_pre,
                        bbox_gt,
                        bbox_matches,
                    )
                    metric_value, metric_num = _masked_mean_over_people(
                        metric,
                        gt_mask,
                    )
                elif name == 'bbox_l1':
                    metric = self.fun_call_dict[name](
                        bbox_pre,
                        bbox_gt,
                        bbox_matches,
                        self.xyz_limits,
                    )
                    metric_value, metric_num = _masked_mean_over_people(
                        metric,
                        gt_mask,
                    )
                elif name == 'objectness':
                    metric = self.fun_call_dict[name](
                        objectness_logits_pre,
                        gt_mask,
                        bbox_matches,
                    )
                    metric_value = metric.mean()
                    metric_num = metric.numel()

                self.metrics_state[name]['sum'] += (
                    metric_value.detach().item() * metric_num
                )
                self.metrics_state[name]['num'] += metric_num
            elif name == 'bce':
                if confidence_pre is None:
                    raise ValueError("BCE requires pre['confidence']")
                metric = self.fun_call_dict[name](
                    confidence_pre,
                    confidence_target,
                )
                metric_value = metric.mean()
                metric_num = metric.numel()
            elif name.startswith('rpm2_'):
                if name == 'rpm2_center_heatmap':
                    metric_value = self.fun_call_dict[name](
                        pre['center_heatmap'],
                        pre['target_center_heatmap'],
                    )
                    metric_num = pre['target_center_heatmap'].numel()
                elif name == 'rpm2_center_offset':
                    metric_value = self.fun_call_dict[name](
                        pre['center_offset_pre'],
                        pre['target_center_offsets_sparse'],
                        pre['target_center_indices'],
                        pre['target_inside'],
                    )
                    metric_num = pre['target_inside'].sum().item()
                elif name == 'rpm2_box_size':
                    metric_value = self.fun_call_dict[name](
                        pre['center_box_pre'],
                        pre['target_center_boxes_sparse'],
                        pre['target_center_indices'],
                        pre['target_inside'],
                    )
                    metric_num = pre['target_inside'].sum().item()
                elif name == 'rpm2_pose':
                    pose_valid = pre['pose_valid'] & gt_mask
                    metric_value = self.fun_call_dict[name](
                        pose_pre,
                        pose_gt,
                        pose_valid,
                    )
                    metric_num = (
                        pose_valid.sum().item() * pose_pre.shape[-2]
                    )

                self.metrics_state[name]['sum'] += (
                    metric_value.detach().item() * metric_num
                )
                self.metrics_state[name]['num'] += metric_num
            elif name.startswith('hrradarpose_'):
                if name == 'hrradarpose_body_center':
                    metric = self.fun_call_dict[name](
                        pre['body_center'],
                        gt['body_center'],
                    )
                    metric_value = metric.mean()
                    metric_num = metric.numel()
                elif name == 'hrradarpose_keypoint_offset':
                    metric = self.fun_call_dict[name](
                        pre['keypoint_offset'],
                        gt['indices'],
                        gt['keypoint_offset'],
                        gt['hrradarpose_valid'],
                    )
                    metric_value, metric_num = _masked_mean_over_people(
                        metric,
                        gt['hrradarpose_valid'],
                    )

            self.metrics_state[name]['sum'] += metric_value.detach().item() * metric_num
            self.metrics_state[name]['num'] += metric_num

            batch_metrics[name] = metric_value
            total_loss = total_loss + weight * metric_value

        return total_loss, batch_metrics

    def _record_group_mpjpe(self, errors, mask, group_ids, gt_low, centered_errors):
        if len(group_ids) != errors.shape[0] or errors.shape != mask.shape or centered_errors.shape != errors.shape:
            raise ValueError('group_ids、MPJPE 与有效人体 mask 的 batch/shape 不一致')
        if gt_low is None or gt_low.shape != (*errors.shape, 17, 3):
            raise ValueError('分组椭圆统计需要与 MPJPE 对齐的低位机 GT [B,T,K,17,3]')
        # 使用原始低位机 GT，避免点云中心化或增广改变验收区域。
        roots = gt_low[..., [11, 12], :2].mean(dim=-2)
        inside = (((roots - roots.new_tensor([2.0, 0.0])) /
                   roots.new_tensor([1.6, 2.4])).square().sum(-1) <= 1.0)
        mask = mask.bool()
        inside = inside.to(mask.device) & mask
        sums = errors.masked_fill(~mask, 0).sum(dim=(1, 2)).cpu().tolist()
        counts = mask.sum(dim=(1, 2)).cpu().tolist()
        inside_sums = errors.masked_fill(~inside, 0).sum(dim=(1, 2)).cpu().tolist()
        inside_counts = inside.sum(dim=(1, 2)).cpu().tolist()
        centered_sums = centered_errors.masked_fill(~mask, 0).sum(dim=(1, 2)).cpu().tolist()
        centered_inside_sums = centered_errors.masked_fill(~inside, 0).sum(dim=(1, 2)).cpu().tolist()
        for group, total, count, inside_total, inside_count, centered_total, centered_inside_total in zip(
                group_ids, sums, counts, inside_sums, inside_counts, centered_sums, centered_inside_sums):
            if count:
                state = self.group_mpjpe.setdefault(group, [0.0, 0, 0.0, 0, 0.0, 0.0])
                state[0] += total
                state[1] += count
                state[2] += inside_total
                state[3] += inside_count
                state[4] += centered_total
                state[5] += centered_inside_total

    def epoch_end(self):
        epoch_metrics = {}

        for name, state_dict in self.metrics_state.items():
            if state_dict['num'] == 0:
                continue

            average = state_dict['sum'] / state_dict['num']

            epoch_metrics[name] = average
            self.metrics_epoch_history[name].append(average)

            # 为下一个 epoch 清零
            state_dict['sum'] = 0.0
            state_dict['num'] = 0

        if self.group_mpjpe:
            labels = {}
            groups = {}
            for group, (total, count, inside_total, inside_count, centered_total, centered_inside_total) in sorted(self.group_mpjpe.items()):
                label = self.group_labels.get(group, {'label': 'unlabelled'})['label']
                groups[group] = {'mpjpe': total / count, 'count': count, 'label': label,
                                 'mpjpe_inside': inside_total / inside_count if inside_count else None,
                                 'count_inside': inside_count,
                                 'centered_mpjpe': centered_total / count,
                                 'centered_mpjpe_inside': centered_inside_total / inside_count if inside_count else None}
                state = labels.setdefault(label, [0.0, 0])
                state[0] += total
                state[1] += count
            epoch_metrics['mpjpe_by_label'] = {
                label: {'mpjpe': total / count, 'count': count}
                for label, (total, count) in sorted(labels.items())
            }
            epoch_metrics['mpjpe_by_group'] = groups
            self.group_mpjpe.clear()
        return epoch_metrics
