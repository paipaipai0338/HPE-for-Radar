from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.labels import LABEL_NAMES
from src.common.losses import MulticlassFocalLoss, sequence_supervision_loss
from src.common.metrics import metrics_from_confusion
from src.pointcloud.data import BehaviorDataset, PointAugmentationConfig
from src.pointcloud.models import BehaviorModel, BehaviorModelConfig


CONFIG_PATH = PROJECT_ROOT / "src" / "pointcloud" / "config" / "train.yaml"


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    per_class: list[dict[str, str | int | float]]
    confusion_matrix: list[list[int]]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def class_weights(dataset: BehaviorDataset, class_count: int) -> torch.Tensor:
    labels = torch.from_numpy(dataset.target_labels)
    counts = torch.bincount(labels, minlength=class_count).float()
    present = counts > 0
    if not present.all():
        missing = torch.nonzero(~present).flatten().tolist()
        print(f"Warning: training split has no samples for classes {missing}")
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def print_dataset_stats(name: str, dataset: BehaviorDataset) -> None:
    labels = np.bincount(dataset.target_labels, minlength=len(LABEL_NAMES))
    points = dataset.associated_point_counts
    label_counts = ", ".join(f"{label}={count}" for label, count in zip(LABEL_NAMES, labels, strict=True))
    index_cache = f"{dataset.sample_index_cache_hit_groups}/{dataset.sample_index_cache_written_groups}"
    point_cache = f"{dataset.persistent_cache_hit_groups}/{dataset.persistent_cache_written_groups}"
    point_summary = (
        f"{points.min()}/{np.median(points):.0f}/"
        f"{np.percentile(points, 95):.0f}/{points.max()}"
    )
    print(
        f"{name}: candidate_frames={dataset.candidate_sample_count}, usable_frames={len(dataset.samples)}, "
        f"unreadable_frames={len(dataset.unreadable_point_paths)}, "
        f"sample_index_cache[hit/write]={index_cache}, persistent_cache[hit/write]={point_cache}, "
        f"windows={len(dataset)}, stride={dataset.window_stride}, labels[{label_counts}], "
        f"points[min/median/p95/max]={point_summary}"
    )
    if dataset.unreadable_point_paths:
        print(f"{name}: first_unreadable={dataset.unreadable_point_paths[0]}")


def evaluate(
    model: BehaviorModel,
    loader: DataLoader,
    device: torch.device,
    label_names: tuple[str, ...],
    criterion: nn.Module,
) -> EvaluationMetrics:
    class_count = len(label_names)
    model.eval()
    total_loss = torch.zeros((), device=device)
    loss_normalizer = torch.zeros((), device=device)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            points = batch["points"].to(device, non_blocking=True)
            mask = batch["point_mask"].to(device, non_blocking=True)
            shape_statistics = batch["shape_statistics"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(points, mask, shape_statistics)
            batch_normalizer = criterion.weight[labels].sum() if criterion.weight is not None else len(labels)
            total_loss += criterion(logits, labels) * batch_normalizer
            loss_normalizer += batch_normalizer
            predictions = logits.argmax(dim=1)
            indices = labels * class_count + predictions
            confusion.view(-1).scatter_add_(0, indices, torch.ones_like(indices))
    metrics = metrics_from_confusion(confusion.cpu(), label_names)
    return EvaluationMetrics(
        (total_loss / loss_normalizer).item(),
        metrics.accuracy,
        metrics.macro_f1,
        metrics.per_class,
        metrics.confusion_matrix,
    )


def print_evaluation_details(metrics: EvaluationMetrics) -> None:
    details = "  ".join(
        f"{item['label']}[P={item['precision']:.3f} R={item['recall']:.3f} "
        f"F1={item['f1']:.3f} N={item['support']}]"
        for item in metrics.per_class
    )
    print(f"val_per_class: {details}")
    print("val_confusion_matrix rows=true cols=predicted")
    print(" " * 13 + " ".join(f"{label:>10}" for label in LABEL_NAMES))
    for label, row in zip(LABEL_NAMES, metrics.confusion_matrix, strict=True):
        print(f"{label:>12} " + " ".join(f"{value:>10}" for value in row))


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_config = config["dataset"]
    training_config = config["training"]
    set_seed(training_config["random_seed"])
    device = select_device(training_config["device"])
    model_config = BehaviorModelConfig(
        sequence_length=dataset_config["sequence_length"],
        **config["model"],
    )

    common_dataset_args = {
        "dataset_root": dataset_config["root"],
        "max_points": dataset_config["max_points"],
        "sequence_length": dataset_config["sequence_length"],
        "max_sync_delta_seconds": dataset_config["max_sync_delta_seconds"],
        "max_frame_gap_seconds": dataset_config["max_frame_gap_seconds"],
        "include_shape_statistics": model_config.use_shape_statistics,
        "include_compact_statistics": model_config.use_compact_statistics,
        "include_point_count": model_config.use_point_count,
        "shape_statistic_dim": model_config.shape_statistic_dim,
        "cache_associated_points": dataset_config["cache_associated_points"],
        "sample_index_cache_dir": PROJECT_ROOT / dataset_config["sample_index_cache_dir"],
        "association_cache_dir": PROJECT_ROOT / dataset_config["association_cache_dir"],
        "association_mode": dataset_config.get("association_mode", "low_pose_association"),
        "coordinate_frame": dataset_config.get("coordinate_frame", "low"),
        "preserve_z_height": dataset_config.get("preserve_z_height", False),
        "high_pose_box_padding": dataset_config.get("high_pose_box_padding", 0.15),
        "high_pose_box_min_points": dataset_config.get("high_pose_box_min_points", 10),
        "high_pose_box_xy_radius": dataset_config.get("high_pose_box_xy_radius"),
        "high_pose_box_z_radius": dataset_config.get("high_pose_box_z_radius", 0.25),
    }
    print("Loading training dataset...")
    train_dataset = BehaviorDataset(
        dates=dataset_config["train_dates"],
        group_start_index=dataset_config["group_start_index"],
        window_stride=dataset_config["train_window_stride"],
        training=True,
        show_progress=True,
        augmentation=PointAugmentationConfig(**config["augmentation"]),
        **common_dataset_args,
    )
    print_dataset_stats("train", train_dataset)
    print("Loading validation dataset...")
    val_dataset = BehaviorDataset(
        dates=dataset_config["val_dates"],
        group_start_index=dataset_config["group_start_index"],
        window_stride=dataset_config["val_window_stride"],
        training=False,
        show_progress=True,
        **common_dataset_args,
    )
    print_dataset_stats("val", val_dataset)
    loader_args = {"batch_size": training_config["batch_size"], "num_workers": training_config["num_workers"]}
    loader_args["pin_memory"] = device.type == "cuda"
    if loader_args["num_workers"] > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=training_config.get("prefetch_factor", 2))
    log_interval = max(1, int(training_config.get("log_interval", 20)))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = BehaviorModel(model_config).to(device)
    weights = class_weights(train_dataset, model_config.class_count).to(device)
    focal_gamma = float(training_config["focal_gamma"])
    criterion = MulticlassFocalLoss(focal_gamma, weights)
    auxiliary_loss_weight = float(training_config["auxiliary_loss_weight"])
    if auxiliary_loss_weight < 0:
        raise ValueError("auxiliary_loss_weight must be non-negative")
    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_config["scheduler_factor"],
        patience=training_config["scheduler_patience"],
        min_lr=training_config["min_learning_rate"],
    )

    output_dir = PROJECT_ROOT / training_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    print(
        f"device={device}, train_windows={len(train_dataset)}, val_windows={len(val_dataset)}, "
        f"auxiliary_loss_weight={auxiliary_loss_weight}, focal_gamma={focal_gamma}"
    )
    for epoch in range(1, training_config["epochs"] + 1):
        model.train()
        running_loss = torch.zeros((), device=device)
        loss_normalizer = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch}/{training_config['epochs']}")
        for step, batch in enumerate(progress, 1):
            points = batch["points"].to(device, non_blocking=True)
            mask = batch["point_mask"].to(device, non_blocking=True)
            shape_statistics = batch["shape_statistics"].to(device, non_blocking=True)
            sequence_labels = batch["sequence_labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            sequence_logits = model.forward_sequence(points, mask, shape_statistics)
            loss = sequence_supervision_loss(
                sequence_logits,
                sequence_labels,
                criterion,
                auxiliary_loss_weight,
            )
            loss.backward()
            optimizer.step()
            batch_normalizer = len(sequence_labels)
            running_loss += loss.detach() * batch_normalizer
            loss_normalizer += batch_normalizer
            if step % log_interval == 0 or step == len(train_loader):
                progress.set_postfix(loss=f"{(running_loss / loss_normalizer).item():.4f}")

        train_loss = (running_loss / loss_normalizer).item()
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            LABEL_NAMES,
            criterion,
        )
        val_macro_f1 = val_metrics.macro_f1
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_accuracy={val_metrics.accuracy:.4f} val_macro_f1={val_macro_f1:.4f} "
            f"lr={learning_rate:.2e}"
        )
        print_evaluation_details(val_metrics)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_macro_f1,
                "val_per_class": val_metrics.per_class,
                "val_confusion_matrix": val_metrics.confusion_matrix,
            }
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "val_per_class": val_metrics.per_class,
                    "val_confusion_matrix": val_metrics.confusion_matrix,
                },
                output_dir / "best.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(val_macro_f1)
        if epochs_without_improvement >= training_config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    with (output_dir / "training_record.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
                "model_config": asdict(model_config),
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "history": history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
