import torch
from metrics.pose import get_mpjpe, get_pampjpe, get_bone_length
from metrics.detection import get_hungarian_match, get_bbox_iou, get_bbox_l1, get_objectness

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
        point_cloud_range,
        matching_bbox_l1_weight,
        matching_bbox_iou_weight,
    ):
        self.point_cloud_range = point_cloud_range
        self.matching_bbox_l1_weight = float(matching_bbox_l1_weight)
        self.matching_bbox_iou_weight = float(
            matching_bbox_iou_weight
        )
        self.fun_call_dict = {
            'mpjpe': get_mpjpe,
            'pampjpe': get_pampjpe,
            'bone_length': get_bone_length,
            'bbox_iou': get_bbox_iou,
            'bbox_l1':  get_bbox_l1,
            'objectness': get_objectness,
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
        bbox_pre = pre.get('bbox', None)
        objectness_logits_pre = pre.get('objectness_logits', None)

        pose_gt = gt.get('padded', None)
        gt_mask = gt.get('mask', None)
        bbox_gt = gt.get('bbox', None)
        

        batch_metrics = {}
        total_loss = 0.0
        matches = None

        for name, weight in self.cfg_metrics.items():
            if name in ['mpjpe', 'pampjpe', 'bone_length']:
                assert pose_pre is not None, 'pose_pre is None'
                metric = self.fun_call_dict[name](pose_pre, pose_gt, type='coco')
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
                if matches is None:
                    matches = get_hungarian_match(
                        bbox_pre,
                        bbox_gt,
                        gt_mask,
                        self.point_cloud_range,
                        bbox_l1_weight=self.matching_bbox_l1_weight,
                        bbox_iou_weight=self.matching_bbox_iou_weight,
                    )

                if name == 'bbox_iou':
                    metric = self.fun_call_dict[name](
                        bbox_pre,
                        bbox_gt,
                        matches,
                    )
                    metric_value, metric_num = _masked_mean_over_people(
                        metric,
                        gt_mask,
                    )
                elif name == 'bbox_l1':
                    metric = self.fun_call_dict[name](
                        bbox_pre,
                        bbox_gt,
                        matches,
                        self.point_cloud_range,
                    )
                    metric_value, metric_num = _masked_mean_over_people(
                        metric,
                        gt_mask,
                    )
                elif name == 'objectness':
                    metric = self.fun_call_dict[name](
                        objectness_logits_pre,
                        gt_mask,
                        matches,
                    )
                    metric_value = metric.mean()
                    metric_num = metric.numel()

                self.metrics_state[name]['sum'] += (
                    metric_value.detach().item() * metric_num
                )
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
