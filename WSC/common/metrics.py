"""Classification metrics shared by behavior-recognition routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class: list[dict[str, str | int | float]]
    confusion_matrix: list[list[int]]


def classification_metrics(
    labels: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
    label_names: tuple[str, ...],
) -> ClassificationMetrics:
    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    predictions = torch.as_tensor(predictions, dtype=torch.long).flatten()
    if labels.shape != predictions.shape or not len(labels):
        raise ValueError("labels and predictions must be non-empty arrays with equal shapes")
    class_count = len(label_names)
    invalid_indices = (
        labels.min() < 0
        or predictions.min() < 0
        or labels.max() >= class_count
        or predictions.max() >= class_count
    )
    if invalid_indices:
        raise ValueError("labels and predictions must contain valid class indices")
    indices = labels * class_count + predictions
    confusion = torch.bincount(indices, minlength=class_count**2).reshape(class_count, class_count)
    return metrics_from_confusion(confusion, label_names)


def metrics_from_confusion(
    confusion: torch.Tensor,
    label_names: tuple[str, ...],
) -> ClassificationMetrics:
    class_count = len(label_names)
    if confusion.shape != (class_count, class_count):
        raise ValueError("confusion matrix shape must match label_names")
    true_positive = confusion.diag().float()
    predicted = confusion.sum(dim=0).float()
    support = confusion.sum(dim=1).float()
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0
    return ClassificationMetrics(
        accuracy=(true_positive.sum() / support.sum().clamp_min(1)).item(),
        macro_f1=f1[supported].mean().item() if supported.any() else 0.0,
        per_class=[
            {
                "label": label,
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "support": int(support[index].item()),
            }
            for index, label in enumerate(label_names)
        ],
        confusion_matrix=confusion.tolist(),
    )
