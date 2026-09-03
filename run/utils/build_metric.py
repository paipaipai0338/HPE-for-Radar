import torch
from metrics.pose import (
    apply_pose_matches,
    get_bce,
    get_bone_length,
    get_mpjpe,
    get_pampjpe,
    get_pose_hungarian_match,
)
from metrics.detection import get_hungarian_match, get_bbox_iou, get_bbox_l1, get_objectness
from metrics.RPM2_loss import get_center_heatmap_loss, get_box_size_loss, get_center_offset_loss, get_pose_loss
from metrics.HRRadarPose_loss import get_body_center_loss, get_keypoint_offset_loss

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

    return metric.sum() * 0.0, 0


class Metric:
    def __init__(
        self,
        cfg_metrics,
        xyz_limits,
        matching_bbox_l1_weight,
        matching_bbox_iou_weight,
        pose_matching_by_hip=False,
    ):
        if not isinstance(pose_matching_by_hip, bool):
            raise TypeError("pose_matching_by_hip must be bool")
        self.xyz_limits = xyz_limits
        self.matching_bbox_l1_weight = float(matching_bbox_l1_weight)
        self.matching_bbox_iou_weight = float(
            matching_bbox_iou_weight
        )
        self.pose_matching_by_hip = pose_matching_by_hip
        self.fun_call_dict = {
            'mpjpe': get_mpjpe,
            'pampjpe': get_pampjpe,
            'bone_length': get_bone_length,
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
        self.metrics_epoch_history = state_dict.get(
            "metrics_epoch_history",
            self.metrics_epoch_history,
        )
        
    def calculate_batch(self, pre, gt):
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

        pose_metric_names = {'mpjpe', 'pampjpe', 'bone_length', 'bce'}
        if self.pose_matching_by_hip and (
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
            )
            pose_pre_for_metric, confidence_target = apply_pose_matches(
                pose_pre,
                pose_gt,
                pose_matches,
            )

        for name, weight in self.cfg_metrics.items():
            if name in ['mpjpe', 'pampjpe', 'bone_length']:
                assert pose_pre is not None, 'pose_pre is None'
                metric = self.fun_call_dict[name](
                    pose_pre_for_metric, pose_gt, type='coco'
                )
                metric_value, metric_num = _masked_mean_over_people(
                    metric,
                    gt_mask,
                )
                self.metrics_state[name]['sum'] += (
                    metric_value.detach().item() * metric_num
                )
                self.metrics_state[name]['num'] += metric_num
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

        return epoch_metrics
