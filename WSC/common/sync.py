from __future__ import annotations

from pathlib import Path

import numpy as np


def timestamp_seconds(path: Path) -> float:
    seconds, nanoseconds = path.stem.split("_")
    return int(seconds) + int(nanoseconds) * 1e-9


def match_nearest_paths(
    reference_paths: list[Path],
    candidate_paths: list[Path],
    max_delta_seconds: float,
) -> list[tuple[Path | None, float]]:
    """Match each reference timestamp to the nearest candidate path."""
    if not candidate_paths:
        raise FileNotFoundError("No candidate files to synchronize")

    candidate_times = np.asarray([timestamp_seconds(path) for path in candidate_paths])
    matches: list[tuple[Path | None, float]] = []
    for reference_path in reference_paths:
        delta = np.abs(candidate_times - timestamp_seconds(reference_path))
        index = int(delta.argmin())
        delta_seconds = float(delta[index])
        matches.append(
            (candidate_paths[index] if delta_seconds <= max_delta_seconds else None, delta_seconds)
        )
    return matches
