import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from metrics.detection import (
    _normalize_bbox,
    get_bbox_iou,
    get_bbox_l1,
    get_hungarian_match,
    get_objectness,
    get_tp_fp_fn_tn,
    paired_axis_aligned_iou_3d,
    pairwise_axis_aligned_iou_3d,
)


POINT_CLOUD_RANGE = (0.0, -3.0, -2.0, 6.0, 3.0, 2.0)


def make_boxes(shape, generator):
    center = torch.rand((*shape, 3), generator=generator)
    center[..., 0] *= 5.0
    center[..., 1] = center[..., 1] * 4.0 - 2.0
    center[..., 2] = center[..., 2] * 2.0 - 1.0
    size = torch.rand((*shape, 3), generator=generator) * 0.8 + 0.2
    return torch.cat([center - size * 0.5, center + size * 0.5], dim=-1)


def reference_matches(pred_bbox, gt_bbox, gt_mask):
    flat_pred = pred_bbox.flatten(0, 1)
    flat_gt = gt_bbox.flatten(0, 1)
    flat_mask = gt_mask.flatten(0, 1)
    matches = []
    for frame_idx in range(flat_pred.shape[0]):
        valid_gt_idx = flat_mask[frame_idx].nonzero().flatten()
        if valid_gt_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            matches.append((empty, empty))
            continue
        current_gt = flat_gt[frame_idx, valid_gt_idx]
        l1_cost = torch.cdist(
            _normalize_bbox(flat_pred[frame_idx], POINT_CLOUD_RANGE),
            _normalize_bbox(current_gt, POINT_CLOUD_RANGE),
            p=1,
        )
        iou_cost = 1.0 - pairwise_axis_aligned_iou_3d(
            flat_pred[frame_idx],
            current_gt,
        )
        pred_idx, local_gt_idx = linear_sum_assignment(
            (5.0 * l1_cost + 2.0 * iou_cost).numpy()
        )
        matches.append(
            (
                torch.as_tensor(pred_idx, dtype=torch.long),
                valid_gt_idx[
                    torch.as_tensor(local_gt_idx, dtype=torch.long)
                ],
            )
        )
    return matches


def reference_losses(pred_bbox, logits, gt_bbox, matches):
    batch_size, frames, queries = logits.shape
    target = torch.zeros_like(logits)
    bbox_l1 = pred_bbox.new_zeros(gt_bbox.shape[:-1])
    bbox_iou = pred_bbox.new_zeros(gt_bbox.shape[:-1])
    for batch_idx in range(batch_size):
        for time_idx in range(frames):
            frame_idx = batch_idx * frames + time_idx
            pred_idx, gt_idx = matches[frame_idx]
            target[batch_idx, time_idx, pred_idx] = 1.0
            if pred_idx.numel() == 0:
                continue
            matched_pred = pred_bbox[batch_idx, time_idx, pred_idx]
            matched_gt = gt_bbox[batch_idx, time_idx, gt_idx]
            bbox_l1[batch_idx, time_idx, gt_idx] = F.l1_loss(
                _normalize_bbox(matched_pred, POINT_CLOUD_RANGE),
                _normalize_bbox(matched_gt, POINT_CLOUD_RANGE),
                reduction="none",
            ).mean(dim=-1)
            bbox_iou[batch_idx, time_idx, gt_idx] = (
                1.0
                - paired_axis_aligned_iou_3d(matched_pred, matched_gt)
            )
    objectness = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return bbox_l1, bbox_iou, objectness


def test_vectorized_matching_losses_and_gradients_match_reference():
    generator = torch.Generator().manual_seed(1234)
    pred = make_boxes((2, 3, 4), generator)
    gt = make_boxes((2, 3, 4), generator)
    mask = torch.tensor(
        [
            [
                [False, False, False, False],
                [True, False, True, False],
                [True, True, True, True],
            ],
            [
                [False, True, False, False],
                [True, True, False, True],
                [False, False, False, False],
            ],
        ],
        dtype=torch.bool,
    )

    expected_matches = reference_matches(pred, gt, mask)
    actual_matches = get_hungarian_match(
        pred,
        gt,
        mask,
        POINT_CLOUD_RANGE,
    )
    assert len(actual_matches) == len(expected_matches)
    assert len(actual_matches[:2]) == 2
    for actual, expected in zip(actual_matches, expected_matches):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    pred_new = pred.clone().requires_grad_(True)
    logits_new = torch.randn(
        (2, 3, 4),
        generator=generator,
        requires_grad=True,
    )
    actual_l1 = get_bbox_l1(
        pred_new,
        gt,
        actual_matches,
        POINT_CLOUD_RANGE,
    )
    actual_iou = get_bbox_iou(pred_new, gt, actual_matches)
    actual_objectness = get_objectness(
        logits_new,
        mask,
        actual_matches,
    )

    pred_reference = pred.clone().requires_grad_(True)
    logits_reference = logits_new.detach().clone().requires_grad_(True)
    expected_l1, expected_iou, expected_objectness = reference_losses(
        pred_reference,
        logits_reference,
        gt,
        expected_matches,
    )

    torch.testing.assert_close(actual_l1, expected_l1)
    torch.testing.assert_close(actual_iou, expected_iou)
    torch.testing.assert_close(actual_objectness, expected_objectness)

    actual_loss = (
        actual_l1.sum()
        + actual_iou.sum()
        + actual_objectness.sum()
    )
    expected_loss = (
        expected_l1.sum()
        + expected_iou.sum()
        + expected_objectness.sum()
    )
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(pred_new.grad, pred_reference.grad)
    torch.testing.assert_close(logits_new.grad, logits_reference.grad)

    packed_counts = get_tp_fp_fn_tn(
        logits_new.detach(),
        mask,
        pred,
        gt,
        actual_matches,
        score_thresh=0.5,
        iou_thresh=0.3,
    )
    legacy_counts = get_tp_fp_fn_tn(
        logits_new.detach(),
        mask,
        pred,
        gt,
        expected_matches,
        score_thresh=0.5,
        iou_thresh=0.3,
    )
    for actual_count, expected_count in zip(
        packed_counts,
        legacy_counts,
    ):
        torch.testing.assert_close(actual_count, expected_count)


def test_matching_rejects_more_gt_than_queries():
    generator = torch.Generator().manual_seed(7)
    pred = make_boxes((1, 1, 2), generator)
    gt = make_boxes((1, 1, 3), generator)
    mask = torch.ones((1, 1, 3), dtype=torch.bool)
    try:
        get_hungarian_match(pred, gt, mask, POINT_CLOUD_RANGE)
    except ValueError as exc:
        assert "only 2 queries" in str(exc)
    else:
        raise AssertionError("Expected too-many-GT ValueError")


if __name__ == "__main__":
    test_vectorized_matching_losses_and_gradients_match_reference()
    test_matching_rejects_more_gt_than_queries()
    print("detection vectorization tests passed")
