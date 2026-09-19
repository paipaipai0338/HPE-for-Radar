from __future__ import annotations

from pathlib import Path

import numpy as np


LABEL_COUNT = 4
LABEL_NAMES = ("stand", "sit_squat", "lie", "other")


def load_action_labels(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load one action-label frame while preserving its original person slots."""
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != LABEL_COUNT or valid.shape != (len(labels),):
        raise ValueError(f"Invalid action-label shapes in {path}")
    return labels, valid


def action_label_index(label: np.ndarray, path: Path | str) -> int:
    """Validate one one-hot action label and return its class index."""
    if not np.isfinite(label).all() or not np.isclose(label.sum(), 1.0):
        raise ValueError(f"Invalid one-hot action label in {path}")
    return int(label.argmax())
