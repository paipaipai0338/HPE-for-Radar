#!/usr/bin/env python3
"""Short, isolated VoxelNeXt training run with component-level timing."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from data2datasets.dataset_for_detection import (
    HPE_Dataset,
    collate_detection_fn,
)
from run.utils.build_metric import Metric
from run.utils.build_model import build_model
from run.utils.load_config import load_config
from run.utils.model_init import model_init
from run.utils.set_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/VoxelNeXt/20260724_231418/config/config.yaml",
    )
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--detail-steps", type=int, default=10)
    parser.add_argument("--profiler-steps", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to experiments/VoxelNeXt/profiling/<timestamp>.",
    )
    return parser.parse_args()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_counts(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def next_batch(
    iterator: Any,
    dataloader: DataLoader,
) -> Tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def prepare_batch(
    samples: Dict[str, Any],
    cfg_task: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    input_key = cfg_task["input"]
    target_key = cfg_task["output"]
    model_input = {
        "input": samples[input_key]["padded"].to(device, non_blocking=True),
        "mask": samples[input_key]["mask"].to(device, non_blocking=True),
    }
    gt = {
        "padded": samples[target_key]["padded"].to(
            device, non_blocking=True
        ),
        "mask": samples[target_key]["mask"].to(device, non_blocking=True),
        "bbox": samples[target_key]["bbox"].to(device, non_blocking=True),
    }
    return model_input, gt


def calculate_loss(
    prediction: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    metric: Metric,
) -> torch.Tensor:
    loss, _ = metric.calculate_batch(prediction, gt)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
    return loss


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    samples: Dict[str, Any],
    cfg_task: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    model_input, gt = prepare_batch(samples, cfg_task, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(model_input)
    loss = calculate_loss(prediction, gt, metric)
    loss.backward()
    optimizer.step()
    return loss


def synchronized_call(
    device: torch.device,
    function: Any,
) -> Tuple[Any, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def detailed_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: Metric,
    samples: Dict[str, Any],
    cfg_task: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    timings: Dict[str, float] = {}

    (model_input, gt), timings["host_to_device_ms"] = synchronized_call(
        device,
        lambda: prepare_batch(samples, cfg_task, device),
    )
    optimizer.zero_grad(set_to_none=True)

    points = model_input["input"]
    mask = model_input["mask"]
    batch_size, num_frames, num_points, channels = points.shape
    frame_batch_size = batch_size * num_frames
    frame_points = points.reshape(
        frame_batch_size, 1, num_points, channels
    )
    frame_mask = mask.reshape(frame_batch_size, 1, num_points)

    batch_dict, timings["voxelizer_ms"] = synchronized_call(
        device,
        lambda: model.voxelizer(frame_points, frame_mask),
    )
    input_voxels = int(batch_dict["voxel_features"].shape[0])

    if input_voxels == 0:
        encoded = None
        timings["sparse_backbone_ms"] = 0.0
        output_tokens = 0
    else:
        batch_dict, timings["sparse_backbone_ms"] = synchronized_call(
            device,
            lambda: model.backbone_3d(batch_dict),
        )
        encoded = batch_dict["encoded_spconv_tensor"]
        output_tokens = int(encoded.features.shape[0])

    (bbox, logits), timings["query_head_ms"] = synchronized_call(
        device,
        lambda: model.query_head(encoded, frame_batch_size),
    )
    prediction = {
        "bbox": bbox.reshape(
            batch_size, num_frames, model.max_people, 6
        ),
        "objectness_logits": logits.reshape(
            batch_size, num_frames, model.max_people
        ),
    }

    loss, timings["matching_and_loss_ms"] = synchronized_call(
        device,
        lambda: calculate_loss(prediction, gt, metric),
    )
    _, timings["backward_ms"] = synchronized_call(
        device,
        loss.backward,
    )
    _, timings["optimizer_ms"] = synchronized_call(
        device,
        optimizer.step,
    )
    timings["compute_total_ms"] = sum(timings.values())

    counts = {
        "valid_points": int(mask.sum().item()),
        "input_voxels": input_voxels,
        "backbone_output_tokens": output_tokens,
        "frame_batch_size": frame_batch_size,
    }
    return timings, counts


def main() -> None:
    args = parse_args()
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {args.device} unavailable; "
            f"device_count={torch.cuda.device_count()}"
        )

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    cfg_data = cfg["data"]
    cfg_task = cfg["task"]
    if cfg["model"]["name"] != "VoxelNeXt":
        raise ValueError(
            f"Expected VoxelNeXt config, got {cfg['model']['name']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments/VoxelNeXt/profiling") / timestamp
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    set_seed(cfg_task["seed"])
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    dataset_start = time.perf_counter()
    dataset = HPE_Dataset(
        root_path=cfg_data["root_path"],
        sensor_config=cfg_data["sensor_config"],
        mode="train",
        base_source=cfg_data["base_source"],
        split_method=cfg_data["split_method"],
        ratio=cfg_data["ratio"],
        T=cfg_data["T"],
        preload_cache=cfg_data.get("preload_cache", False),
    )
    dataset_init_seconds = time.perf_counter() - dataset_start

    dataloader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "collate_fn": partial(
            collate_detection_fn,
            max_points=cfg_data["max_points"],
            max_people=cfg_data["max_people"],
        ),
        "shuffle": cfg_task["train"]["shuffle"],
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    dataloader = DataLoader(**dataloader_kwargs)

    model = model_init(build_model("VoxelNeXt")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_task["train"]["init_lr"],
        betas=(0.9, 0.999),
    )
    matching_cfg = cfg_task["matching_for_hungarian"]
    metric = Metric(
        cfg_task["train"]["metrics"],
        cfg_data["point_cloud_range"],
        matching_cfg["bbox_l1_weight"],
        matching_cfg["bbox_iou_weight"],
    )
    model.train()

    iterator = iter(dataloader)
    print(
        json.dumps(
            {
                "phase": "setup_complete",
                "output_dir": str(output_dir),
                "dataset_samples": len(dataset),
                "dataset_init_seconds": dataset_init_seconds,
                "device": torch.cuda.get_device_name(args.device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for step in range(args.warmup_steps):
        samples, iterator = next_batch(iterator, dataloader)
        train_step(
            model, optimizer, metric, samples, cfg_task, device
        )
        torch.cuda.synchronize(device)
        print(
            json.dumps(
                {"phase": "warmup", "step": step + 1},
                ensure_ascii=False,
            ),
            flush=True,
        )

    end_to_end_ms: List[float] = []
    data_wait_ms: List[float] = []
    for step in range(args.timed_steps):
        wait_start = time.perf_counter()
        samples, iterator = next_batch(iterator, dataloader)
        data_wait_ms.append((time.perf_counter() - wait_start) * 1000.0)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = train_step(
            model, optimizer, metric, samples, cfg_task, device
        )
        torch.cuda.synchronize(device)
        end_to_end_ms.append((time.perf_counter() - start) * 1000.0)
        print(
            json.dumps(
                {
                    "phase": "timed",
                    "step": step + 1,
                    "compute_ms": end_to_end_ms[-1],
                    "data_wait_ms": data_wait_ms[-1],
                    "loss": float(loss.detach().cpu()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    detailed: Dict[str, List[float]] = {}
    counts: List[Dict[str, int]] = []
    for step in range(args.detail_steps):
        samples, iterator = next_batch(iterator, dataloader)
        step_timings, step_counts = detailed_train_step(
            model,
            optimizer,
            metric,
            samples,
            cfg_task,
            device,
        )
        for name, value in step_timings.items():
            detailed.setdefault(name, []).append(value)
        counts.append(step_counts)
        print(
            json.dumps(
                {
                    "phase": "detail",
                    "step": step + 1,
                    **step_timings,
                    **step_counts,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    profiler_table_cuda = ""
    profiler_table_cpu = ""
    trace_path = output_dir / "torch_profiler_trace.json"
    if args.profiler_steps > 0:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for step in range(args.profiler_steps):
                samples, iterator = next_batch(iterator, dataloader)
                train_step(
                    model, optimizer, metric, samples, cfg_task, device
                )
                torch.cuda.synchronize(device)
                prof.step()
                print(
                    json.dumps(
                        {"phase": "profiler", "step": step + 1},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        prof.export_chrome_trace(str(trace_path))
        profiler_table_cuda = prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
        profiler_table_cpu = prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=40,
        )
        (output_dir / "profiler_cuda.txt").write_text(
            profiler_table_cuda, encoding="utf-8"
        )
        (output_dir / "profiler_cpu.txt").write_text(
            profiler_table_cpu, encoding="utf-8"
        )

    result = {
        "status": "completed",
        "timestamp": timestamp,
        "command_args": vars(args),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
        },
        "dataset": {
            "samples": len(dataset),
            "init_seconds": dataset_init_seconds,
            "T": cfg_data["T"],
            "max_points": cfg_data["max_points"],
            "max_people": cfg_data["max_people"],
        },
        "end_to_end_compute": summarize(end_to_end_ms),
        "data_wait": summarize(data_wait_ms),
        "components": {
            name: summarize(values)
            for name, values in detailed.items()
        },
        "counts": {
            name: summarize_counts(
                [float(step_counts[name]) for step_counts in counts]
            )
            for name in counts[0]
        } if counts else {},
        "trace_path": str(trace_path) if trace_path.exists() else None,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if profiler_table_cuda:
        print(profiler_table_cuda, flush=True)
    if profiler_table_cpu:
        print(profiler_table_cpu, flush=True)


if __name__ == "__main__":
    main()
