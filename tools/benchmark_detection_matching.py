#!/usr/bin/env python3
"""Microbenchmark the detection matching/loss path without loading data."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable, Dict

import torch

from metrics.detection import (
    get_bbox_iou,
    get_bbox_l1,
    get_hungarian_match,
    get_objectness,
)
from run.utils.build_metric import Metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def benchmark(
    function: Callable[[], object],
    device: torch.device,
    repeats: int,
) -> Dict[str, float]:
    for _ in range(10):
        function()
    torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
        del result
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def make_boxes(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    center = torch.empty((*shape, 3), device=device).uniform_(-1.0, 1.0)
    center[..., 0].add_(3.0)
    size = torch.empty((*shape, 3), device=device).uniform_(0.2, 1.0)
    return torch.cat([center - size * 0.5, center + size * 0.5], dim=-1)


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    batch_size, frames, queries, max_gt = 8, 8, 6, 6
    point_cloud_range = (0.0, -3.0, -2.0, 6.0, 3.0, 2.0)

    results = {}
    for valid_people in (1, 3, 6):
        pred_bbox = make_boxes(
            (batch_size, frames, queries), device
        ).requires_grad_(True)
        gt_bbox = make_boxes((batch_size, frames, max_gt), device)
        gt_mask = torch.zeros(
            (batch_size, frames, max_gt),
            dtype=torch.bool,
            device=device,
        )
        gt_mask[..., :valid_people] = True
        logits = torch.randn(
            (batch_size, frames, queries),
            device=device,
            requires_grad=True,
        )
        matches = get_hungarian_match(
            pred_bbox,
            gt_bbox,
            gt_mask,
            point_cloud_range,
        )
        torch.cuda.synchronize(device)

        metric = Metric(
            {
                "bbox_iou": 1,
                "bbox_l1": 1,
                "objectness": 1,
            },
            point_cloud_range,
            5.0,
            2.0,
        )
        prediction = {
            "bbox": pred_bbox,
            "objectness_logits": logits,
        }
        target = {
            "bbox": gt_bbox,
            "mask": gt_mask,
            "padded": torch.empty(0, device=device),
        }

        case = {
            "hungarian": benchmark(
                lambda: get_hungarian_match(
                    pred_bbox,
                    gt_bbox,
                    gt_mask,
                    point_cloud_range,
                ),
                device,
                args.repeats,
            ),
            "bbox_iou": benchmark(
                lambda: get_bbox_iou(pred_bbox, gt_bbox, matches),
                device,
                args.repeats,
            ),
            "bbox_l1": benchmark(
                lambda: get_bbox_l1(
                    pred_bbox,
                    gt_bbox,
                    matches,
                    point_cloud_range,
                ),
                device,
                args.repeats,
            ),
            "objectness": benchmark(
                lambda: get_objectness(logits, gt_mask, matches),
                device,
                args.repeats,
            ),
            "metric_total": benchmark(
                lambda: metric.calculate_batch(prediction, target),
                device,
                args.repeats,
            ),
        }
        results[str(valid_people)] = case
        print(
            json.dumps(
                {"valid_people_per_frame": valid_people, **case},
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(args.device),
                "shape": {
                    "B": batch_size,
                    "T": frames,
                    "Q": queries,
                    "K": max_gt,
                },
                "repeats": args.repeats,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
