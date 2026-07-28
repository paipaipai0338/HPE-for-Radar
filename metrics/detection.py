"""
记录与人体检测相关的指标
"""

from collections.abc import Iterator
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


LegacyMatchIndices = List[Tuple[torch.Tensor, torch.Tensor]]


class PackedMatchIndices:
    """GPU 上的打包匹配索引，同时兼容原逐帧列表接口。"""

    def __init__(
        self,
        frame_indices: torch.Tensor,
        pred_indices: torch.Tensor,
        gt_indices: torch.Tensor,
        frame_offsets: Sequence[int],
    ) -> None:
        tensors = (frame_indices, pred_indices, gt_indices)
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("Packed match indices must be one-dimensional")
        if len({tensor.numel() for tensor in tensors}) != 1:
            raise ValueError("Packed match index lengths differ")
        if len(frame_offsets) == 0 or int(frame_offsets[0]) != 0:
            raise ValueError("frame_offsets must start at zero")
        if int(frame_offsets[-1]) != frame_indices.numel():
            raise ValueError(
                "Last frame offset must equal the packed match count"
            )
        if any(
            int(frame_offsets[idx]) > int(frame_offsets[idx + 1])
            for idx in range(len(frame_offsets) - 1)
        ):
            raise ValueError("frame_offsets must be non-decreasing")

        self.frame_indices = frame_indices
        self.pred_indices = pred_indices
        self.gt_indices = gt_indices
        self.frame_offsets = tuple(int(value) for value in frame_offsets)

    def __len__(self) -> int:
        return len(self.frame_offsets) - 1

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        LegacyMatchIndices,
    ]:
        if isinstance(index, slice):
            return [
                self[frame_idx]
                for frame_idx in range(*index.indices(len(self)))
            ]

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.frame_offsets[index]
        end = self.frame_offsets[index + 1]
        return (
            self.pred_indices[start:end],
            self.gt_indices[start:end],
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for frame_idx in range(len(self)):
            yield self[frame_idx]


MatchIndices = Union[PackedMatchIndices, LegacyMatchIndices]


def _get_packed_match_indices(
    matches: MatchIndices,
    expected_frames: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(matches) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frame matches, got {len(matches)}"
        )
    if isinstance(matches, PackedMatchIndices):
        packed = (
            matches.frame_indices,
            matches.pred_indices,
            matches.gt_indices,
        )
        if any(tensor.device != device for tensor in packed):
            raise ValueError("Packed matches and predictions must share device")
        return packed

    frame_parts = []
    pred_parts = []
    gt_parts = []
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() != gt_idx.numel():
            raise ValueError(
                f"Frame {frame_idx} has different prediction and GT "
                "match counts"
            )
        if pred_idx.numel() == 0:
            continue
        pred_idx = pred_idx.to(device=device, dtype=torch.long)
        gt_idx = gt_idx.to(device=device, dtype=torch.long)
        frame_parts.append(
            torch.full_like(pred_idx, frame_idx, device=device)
        )
        pred_parts.append(pred_idx)
        gt_parts.append(gt_idx)

    if not pred_parts:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty
    return (
        torch.cat(frame_parts),
        torch.cat(pred_parts),
        torch.cat(gt_parts),
    )


def _batched_pairwise_axis_aligned_iou_3d(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算 ``[F,Q,6]`` 与 ``[F,K,6]`` 的逐帧两两 IoU。"""
    if boxes_a.ndim != 3 or boxes_a.shape[-1] != 6:
        raise ValueError(
            f"boxes_a must be [F, Q, 6], got {tuple(boxes_a.shape)}"
        )
    if boxes_b.ndim != 3 or boxes_b.shape[-1] != 6:
        raise ValueError(
            f"boxes_b must be [F, K, 6], got {tuple(boxes_b.shape)}"
        )
    if boxes_a.shape[0] != boxes_b.shape[0]:
        raise ValueError("boxes_a and boxes_b frame dimensions differ")

    intersection_min = torch.maximum(
        boxes_a[:, :, None, :3],
        boxes_b[:, None, :, :3],
    )
    intersection_max = torch.minimum(
        boxes_a[:, :, None, 3:],
        boxes_b[:, None, :, 3:],
    )
    intersection = (
        (intersection_max - intersection_min)
        .clamp_min(0)
        .prod(dim=-1)
    )
    volume_a = (
        (boxes_a[..., 3:] - boxes_a[..., :3])
        .clamp_min(0)
        .prod(dim=-1)
    )
    volume_b = (
        (boxes_b[..., 3:] - boxes_b[..., :3])
        .clamp_min(0)
        .prod(dim=-1)
    )
    union = (
        volume_a[:, :, None]
        + volume_b[:, None, :]
        - intersection
    )
    return intersection / union.clamp_min(eps)


def pairwise_axis_aligned_iou_3d(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算两组轴对齐 3D 框两两之间的 IoU。

    Args:
        boxes_a: ``[N, 6]``，最后一维为 ``xyz_min, xyz_max``。
        boxes_b: ``[M, 6]``，最后一维为 ``xyz_min, xyz_max``。

    Returns:
        ``[N, M]`` 的 IoU 矩阵。
    """
    if boxes_a.ndim != 2 or boxes_a.shape[-1] != 6:
        raise ValueError(f"boxes_a must be [N, 6], got {tuple(boxes_a.shape)}")
    if boxes_b.ndim != 2 or boxes_b.shape[-1] != 6:
        raise ValueError(f"boxes_b must be [M, 6], got {tuple(boxes_b.shape)}")

    # 交集的最小角点 = 两个框最小角点的最大值
    intersection_min = torch.maximum(boxes_a[:, None, :3], boxes_b[None, :, :3])    # N, M, 3
    # 交集的最大角点 = 两个框最大角点的最小值  
    intersection_max = torch.minimum(boxes_a[:, None, 3:], boxes_b[None, :, 3:])    # N, M, 3
    intersection_size = (intersection_max - intersection_min).clamp_min(0)          # N, M, 3
    # 求体积
    intersection = intersection_size.prod(dim=-1)                                   # N, M

    volume_a = (boxes_a[:, 3:] - boxes_a[:, :3]).clamp_min(0).prod(dim=-1)
    volume_b = (boxes_b[:, 3:] - boxes_b[:, :3]).clamp_min(0).prod(dim=-1)
    union = volume_a[:, None] + volume_b[None, :] - intersection
    return intersection / union.clamp_min(eps)


def paired_axis_aligned_iou_3d(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算已经配对的轴对齐 3D 框 IoU，返回 ``[N]``。"""
    if boxes_a.shape != boxes_b.shape:
        raise ValueError(
            "Paired boxes must have the same shape, "
            f"got {tuple(boxes_a.shape)} and {tuple(boxes_b.shape)}"
        )
    if boxes_a.ndim != 2 or boxes_a.shape[-1] != 6:
        raise ValueError(f"boxes must be [N, 6], got {tuple(boxes_a.shape)}")

    intersection_min = torch.maximum(boxes_a[:, :3], boxes_b[:, :3])
    intersection_max = torch.minimum(boxes_a[:, 3:], boxes_b[:, 3:])
    intersection = (
        (intersection_max - intersection_min).clamp_min(0).prod(dim=-1)
    )
    volume_a = (boxes_a[:, 3:] - boxes_a[:, :3]).clamp_min(0).prod(dim=-1)
    volume_b = (boxes_b[:, 3:] - boxes_b[:, :3]).clamp_min(0).prod(dim=-1)
    union = volume_a + volume_b - intersection
    return intersection / union.clamp_min(eps)


def _normalize_bbox(
    bbox: torch.Tensor,
    point_cloud_range: Sequence[float],
) -> torch.Tensor:
    point_cloud_range = torch.as_tensor(
        point_cloud_range, dtype=bbox.dtype, device=bbox.device
    )
    if point_cloud_range.numel() != 6:
        raise ValueError("point_cloud_range must contain 6 values")
    pc_min = point_cloud_range[:3]
    extent = (point_cloud_range[3:] - pc_min).clamp_min(1e-6)
    return torch.cat(
        [
            (bbox[..., :3] - pc_min) / extent,
            (bbox[..., 3:] - pc_min) / extent,
        ],
        dim=-1,
    )


def get_hungarian_match(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    gt_mask: torch.Tensor,
    point_cloud_range: Sequence[float],
    bbox_l1_weight: float = 5.0,
    bbox_iou_weight: float = 2.0,
) -> MatchIndices:
    """批量构造代价后逐帧进行 CPU 匈牙利匹配。

    GPU 上一次构造 ``[B*T,Q,K]`` 代价并一次传到 CPU；CPU 完成各帧
    Hungarian 后，将打包索引一次传回 GPU。返回对象兼容原来的逐帧
    ``matches[frame_idx]`` 和迭代接口。
    """
    if pred_bbox.ndim != 4 or pred_bbox.shape[-1] != 6:
        raise ValueError(
            f"pred_bbox must be [B, T, Q, 6], got {tuple(pred_bbox.shape)}"
        )
    if gt_bbox.ndim != 4 or gt_bbox.shape[-1] != 6:
        raise ValueError(
            f"gt_bbox must be [B, T, K, 6], got {tuple(gt_bbox.shape)}"
        )
    if gt_mask.shape != gt_bbox.shape[:-1]:
        raise ValueError(
            "gt_mask must match gt_bbox B,T,K dimensions, "
            f"got {tuple(gt_mask.shape)} and {tuple(gt_bbox.shape)}"
        )
    if pred_bbox.shape[:2] != gt_bbox.shape[:2]:
        raise ValueError("Prediction and GT B,T dimensions differ")
    if pred_bbox.device != gt_bbox.device or pred_bbox.device != gt_mask.device:
        raise ValueError("Prediction, GT boxes and GT mask must share device")

    flat_pred_bbox = pred_bbox.flatten(0, 1)
    flat_gt_bbox = gt_bbox.flatten(0, 1)
    flat_gt_mask = gt_mask.bool().flatten(0, 1)
    num_frames, num_queries = flat_pred_bbox.shape[:2]

    with torch.no_grad():
        pred_norm = _normalize_bbox(flat_pred_bbox, point_cloud_range)
        gt_norm = _normalize_bbox(flat_gt_bbox, point_cloud_range)
        l1_cost = (
            pred_norm[:, :, None, :] - gt_norm[:, None, :, :]
        ).abs().sum(dim=-1)
        iou_cost = 1.0 - _batched_pairwise_axis_aligned_iou_3d(
            flat_pred_bbox,
            flat_gt_bbox,
        )
        matching_cost = (
            float(bbox_l1_weight) * l1_cost
            + float(bbox_iou_weight) * iou_cost
        )

        # 将 mask 作为最后一行一起传输，确保整个 matching 仅有一次
        # GPU -> CPU copy/synchronization。
        cpu_payload = torch.cat(
            [
                matching_cost,
                flat_gt_mask[:, None, :].to(matching_cost.dtype),
            ],
            dim=1,
        ).cpu().numpy()

    cost_cpu = cpu_payload[:, :num_queries, :]
    valid_mask_cpu = cpu_payload[:, num_queries, :] > 0.5
    frame_offsets = [0]
    packed_frame_indices = []
    packed_pred_indices = []
    packed_gt_indices = []

    for frame_idx in range(num_frames):
        valid_gt_idx = np.flatnonzero(valid_mask_cpu[frame_idx])
        if valid_gt_idx.size > num_queries:
            raise ValueError(
                f"Frame {frame_idx} contains {valid_gt_idx.size} people, "
                f"but only {num_queries} queries are available."
            )
        if valid_gt_idx.size == 0:
            frame_offsets.append(frame_offsets[-1])
            continue

        frame_cost = cost_cpu[frame_idx][:, valid_gt_idx]
        if not np.isfinite(frame_cost).all():
            raise ValueError(
                f"Frame {frame_idx} matching cost contains NaN or Inf"
            )
        pred_idx, local_gt_idx = linear_sum_assignment(
            frame_cost
        )
        gt_idx = valid_gt_idx[local_gt_idx]
        packed_frame_indices.append(
            np.full(pred_idx.shape, frame_idx, dtype=np.int64)
        )
        packed_pred_indices.append(pred_idx.astype(np.int64, copy=False))
        packed_gt_indices.append(gt_idx.astype(np.int64, copy=False))
        frame_offsets.append(frame_offsets[-1] + pred_idx.size)

    if packed_pred_indices:
        packed_cpu = np.stack(
            [
                np.concatenate(packed_frame_indices),
                np.concatenate(packed_pred_indices),
                np.concatenate(packed_gt_indices),
            ],
            axis=0,
        )
        # 所有匹配索引一次 CPU -> GPU copy。
        packed_gpu = torch.as_tensor(
            packed_cpu,
            dtype=torch.long,
            device=pred_bbox.device,
        )
    else:
        packed_gpu = torch.empty(
            (3, 0),
            dtype=torch.long,
            device=pred_bbox.device,
        )

    return PackedMatchIndices(
        frame_indices=packed_gpu[0],
        pred_indices=packed_gpu[1],
        gt_indices=packed_gpu[2],
        frame_offsets=frame_offsets,
    )


def get_objectness(
    objectness_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    matches: MatchIndices,
) -> torch.Tensor:
    """计算每个固定 query 的人体存在性 BCE，返回 ``[B, T, Q]``。"""
    if objectness_logits.ndim != 3:
        raise ValueError(
            "objectness_logits must be [B, T, Q], "
            f"got {tuple(objectness_logits.shape)}"
        )
    if gt_mask.shape[:2] != objectness_logits.shape[:2]:
        raise ValueError("objectness_logits and gt_mask B,T dimensions differ")

    flat_logits = objectness_logits.flatten(0, 1)
    frame_idx, pred_idx, _ = _get_packed_match_indices(
        matches,
        expected_frames=flat_logits.shape[0],
        device=objectness_logits.device,
    )
    target = torch.zeros_like(flat_logits)
    target[frame_idx, pred_idx] = 1.0
    return F.binary_cross_entropy_with_logits(
        flat_logits, target, reduction="none"
    ).reshape_as(objectness_logits)


def get_bbox_l1(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    matches: MatchIndices,
    point_cloud_range: Sequence[float],
) -> torch.Tensor:
    """计算匹配框的归一化 L1，返回 ``[B, T, K]``。

    误差按照匹配到的 GT 槽位写回；无效 GT 槽位保持为 0，调用方应使用
    ``gt_mask`` 计算 masked mean。
    """
    flat_pred_bbox = pred_bbox.flatten(0, 1)
    flat_gt_bbox = gt_bbox.flatten(0, 1)
    frame_idx, pred_idx, gt_idx = _get_packed_match_indices(
        matches,
        expected_frames=flat_pred_bbox.shape[0],
        device=pred_bbox.device,
    )
    bbox_l1 = pred_bbox.new_zeros(flat_gt_bbox.shape[:-1])
    if pred_idx.numel() > 0:
        matched_pred = flat_pred_bbox[frame_idx, pred_idx]
        matched_gt = flat_gt_bbox[frame_idx, gt_idx]
        matched_l1 = F.l1_loss(
            _normalize_bbox(matched_pred, point_cloud_range),
            _normalize_bbox(matched_gt, point_cloud_range),
            reduction="none",
        ).mean(dim=-1)
        bbox_l1[frame_idx, gt_idx] = matched_l1
    return bbox_l1.reshape(gt_bbox.shape[:-1])


def get_bbox_iou(
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    matches: MatchIndices,
) -> torch.Tensor:
    """计算匹配框的 ``1 - IoU``，返回 ``[B, T, K]``。

    误差按照匹配到的 GT 槽位写回；无效 GT 槽位保持为 0，调用方应使用
    ``gt_mask`` 计算 masked mean。
    """
    flat_pred_bbox = pred_bbox.flatten(0, 1)
    flat_gt_bbox = gt_bbox.flatten(0, 1)
    frame_idx, pred_idx, gt_idx = _get_packed_match_indices(
        matches,
        expected_frames=flat_pred_bbox.shape[0],
        device=pred_bbox.device,
    )
    bbox_iou_loss = pred_bbox.new_zeros(flat_gt_bbox.shape[:-1])
    if pred_idx.numel() > 0:
        matched_pred = flat_pred_bbox[frame_idx, pred_idx]
        matched_gt = flat_gt_bbox[frame_idx, gt_idx]
        bbox_iou_loss[frame_idx, gt_idx] = (
            1.0 - paired_axis_aligned_iou_3d(matched_pred, matched_gt)
        )
    return bbox_iou_loss.reshape(gt_bbox.shape[:-1])

def get_tp_fp_fn_tn(
    objectness_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_bbox: torch.Tensor,
    gt_bbox: torch.Tensor,
    matches: MatchIndices,
    score_thresh: float,
    iou_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    匈牙利匹配只负责在预测 query 与有效 GT 之间确定最小代价的一对一分配，不代表检测成功。
    除此之外需要额外验证是否检测成功，具体逻辑为：
    TP 匈牙利匹配结果中 objectness_logits.sigmoid() >= score_thresh 且 IoU >= iou_thresh
    FN 匈牙利匹配结果中 objectness_logits.sigmoid() < score_thresh 或 IoU < iou_thresh
    不存在于匈牙利匹配结果默认为负样本，由于不存在匹配数据并没有 gt 只能通过置信度判决
    TN 非匈牙利匹配结果 objectness_logits.sigmoid() < score_thresh
    FP 非匈牙利匹配结果 objectness_logits.sigmoid() >= score_thresh
    """
    if objectness_logits.ndim != 3:
        raise ValueError(
            "objectness_logits must be [B, T, Q], "
            f"got {tuple(objectness_logits.shape)}"
        )
    if pred_bbox.ndim != 4 or pred_bbox.shape[-1] != 6:
        raise ValueError(
            "pred_bbox must be [B, T, Q, 6], "
            f"got {tuple(pred_bbox.shape)}"
        )
    if gt_bbox.ndim != 4 or gt_bbox.shape[-1] != 6:
        raise ValueError(
            "gt_bbox must be [B, T, K, 6], "
            f"got {tuple(gt_bbox.shape)}"
        )
    if gt_mask.ndim != 3:
        raise ValueError(
            f"gt_mask must be [B, T, K], got {tuple(gt_mask.shape)}"
        )
    if pred_bbox.shape[:3] != objectness_logits.shape:
        raise ValueError(
            "pred_bbox B,T,Q dimensions must match objectness_logits, "
            f"got pred_bbox={tuple(pred_bbox.shape)} and "
            f"objectness_logits={tuple(objectness_logits.shape)}"
        )
    if gt_bbox.shape[:3] != gt_mask.shape:
        raise ValueError(
            "gt_bbox B,T,K dimensions must match gt_mask, "
            f"got gt_bbox={tuple(gt_bbox.shape)} and "
            f"gt_mask={tuple(gt_mask.shape)}"
        )
    if gt_mask.shape[:2] != objectness_logits.shape[:2]:
        raise ValueError(
            "prediction and GT B,T dimensions differ, "
            f"got prediction={tuple(objectness_logits.shape[:2])} and "
            f"GT={tuple(gt_mask.shape[:2])}"
        )
    devices = {
        objectness_logits.device,
        pred_bbox.device,
        gt_bbox.device,
        gt_mask.device,
    }
    if len(devices) != 1:
        raise ValueError(
            "objectness_logits, pred_bbox, gt_bbox and gt_mask must be "
            "on the same device"
        )
    if not torch.isfinite(objectness_logits).all():
        raise ValueError("objectness_logits must contain only finite values")
    if not torch.isfinite(pred_bbox).all():
        raise ValueError("pred_bbox must contain only finite values")
    if not torch.isfinite(gt_bbox).all():
        raise ValueError("gt_bbox must contain only finite values")
    if not 0.0 <= float(score_thresh) <= 1.0:
        raise ValueError(
            f"score_thresh must be in [0, 1], got {score_thresh}"
        )
    if not 0.0 <= float(iou_thresh) <= 1.0:
        raise ValueError(
            f"iou_thresh must be in [0, 1], got {iou_thresh}"
        )

    flat_logits = objectness_logits.flatten(0, 1)
    flat_pred_bbox = pred_bbox.flatten(0, 1)
    flat_gt_bbox = gt_bbox.flatten(0, 1)
    flat_gt_mask = gt_mask.bool().flatten(0, 1)
    if len(matches) != flat_logits.shape[0]:
        raise ValueError(
            f"Expected {flat_logits.shape[0]} frame matches, "
            f"got {len(matches)}"
        )

    target = torch.zeros_like(flat_logits, dtype=torch.bool)
    iou = torch.zeros_like(flat_logits, dtype=torch.bool)
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.ndim != 1 or gt_idx.ndim != 1:
            raise ValueError("match indices must be one-dimensional")
        if pred_idx.numel() != gt_idx.numel():
            raise ValueError(
                f"Frame {frame_idx} has different prediction and GT "
                "match counts"
            )
        if pred_idx.numel() == 0:
            continue
        if (
            (pred_idx < 0).any()
            or (pred_idx >= flat_pred_bbox.shape[1]).any()
        ):
            raise IndexError(
                f"Frame {frame_idx} contains out-of-range prediction indices"
            )
        if (
            (gt_idx < 0).any()
            or (gt_idx >= flat_gt_bbox.shape[1]).any()
        ):
            raise IndexError(
                f"Frame {frame_idx} contains out-of-range GT indices"
            )
        if pred_idx.unique().numel() != pred_idx.numel():
            raise ValueError(
                f"Frame {frame_idx} contains duplicate prediction matches"
            )
        if gt_idx.unique().numel() != gt_idx.numel():
            raise ValueError(
                f"Frame {frame_idx} contains duplicate GT matches"
            )
        if not flat_gt_mask[frame_idx, gt_idx].all():
            raise ValueError(
                f"Frame {frame_idx} contains matches to invalid GT entries"
            )

        target[frame_idx, pred_idx] = True
        matched_pred = flat_pred_bbox[frame_idx, pred_idx]
        matched_gt = flat_gt_bbox[frame_idx, gt_idx]
        iou[frame_idx, pred_idx] = (paired_axis_aligned_iou_3d(matched_pred, matched_gt) >= iou_thresh)

    prediction_for_matched = ((flat_logits.sigmoid() >= float(score_thresh)) & iou)
    prediction_for_all = flat_logits.sigmoid() >= float(score_thresh)
    tp = (prediction_for_matched & target).sum()
    fp = (prediction_for_all & ~target).sum()
    fn = (~prediction_for_matched & target).sum()
    tn = (~prediction_for_all & ~target).sum()
    return tp, fp, fn, tn

def get_acc(tp, fp, fn, tn, eps=1e-6):
    return (tp + tn) / (tp + fp + fn + tn + eps)


def get_precision(tp, fp, fn, tn, eps=1e-6):
    return tp / (tp + fp + eps)


def get_recall(tp, fp, fn, tn, eps=1e-6):
    return tp / (tp + fn + eps)


if __name__ == "__main__":
    pred_bbox = torch.rand((2, 3, 4, 6))
    pred_bbox[..., 3:] += pred_bbox[..., :3]
    objectness_logits = torch.rand((2, 3, 4))
    gt_bbox = torch.rand((2, 3, 4, 6))
    gt_bbox[..., 3:] += gt_bbox[..., :3]
    gt_mask = torch.rand((2, 3, 4)) > 0.5
    point_cloud_range = (0.0, -3.0, -2.0, 6.0, 3.0, 2.0)

    matches = get_hungarian_match(
        pred_bbox, gt_bbox, gt_mask, point_cloud_range
    )
    objectness = get_objectness(objectness_logits, gt_mask, matches)
    bbox_l1 = get_bbox_l1(
        pred_bbox, gt_bbox, matches, point_cloud_range
    )
    bbox_iou = get_bbox_iou(pred_bbox, gt_bbox, matches)
    print("objectness", objectness.shape)
    print("bbox_l1", bbox_l1.shape)
    print("bbox_iou", bbox_iou.shape)
