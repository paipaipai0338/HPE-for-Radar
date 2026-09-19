from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from WSC.common.pose_io import (
    transform_points_between_radars,
)


POINT_STATE_COLUMN = 5


@dataclass(frozen=True, slots=True)
class PoseAssociationConfig:
    """Core association and conservative completeness limits."""

    core_bbox_padding: tuple[float, float, float] = (0.1, 0.1, 0.1)
    expansion_padding: tuple[float, float, float] = (0.4, 0.4, 0.3)
    association_distance: float = 0.35
    growth_radius: float = 0.25
    max_core_distance: float = 0.65
    max_growth_steps: int = 2
    min_core_points: int = 12
    cluster_radius: float = 0.2
    min_cluster_points: int = 20
    min_capture_ratio: float = 0.7
    max_temporal_xy_offset: float = 0.35


@dataclass(frozen=True, slots=True)
class PoseAssociationResult:
    """Associated raw point rows, including the directly accepted core points."""

    points: np.ndarray
    point_indices: np.ndarray
    core_indices: np.ndarray
    bbox: np.ndarray


class TemporalPoseAssociator:
    """Retry failed associations with the last reliable pose-to-cloud XY offset."""

    def __init__(self, config: PoseAssociationConfig, max_frame_gap_seconds: float) -> None:
        self.config = config
        self.max_frame_gap_seconds = max_frame_gap_seconds
        self._last_timestamp: float | None = None
        self._xy_offset: np.ndarray | None = None

    def associate(
        self,
        points: np.ndarray,
        pose: np.ndarray,
        timestamp: float,
        *,
        skeleton: list[list[int]] | tuple[tuple[int, int], ...],
    ) -> PoseAssociationResult:
        continuous = (
            self._last_timestamp is not None
            and 0.0 <= timestamp - self._last_timestamp <= self.max_frame_gap_seconds
        )
        if not continuous:
            self._xy_offset = None

        result = associate_human_points(points, pose, skeleton=skeleton, config=self.config)
        if not len(result.points) and self._xy_offset is not None:
            shifted_pose = np.asarray(pose, dtype=np.float64).copy()
            shifted_pose[:, :2] += self._xy_offset
            result = associate_human_points(
                points,
                shifted_pose,
                skeleton=skeleton,
                config=self.config,
            )

        self._last_timestamp = timestamp
        if len(result.points):
            valid_pose = np.isfinite(pose).all(axis=1)
            offset = np.median(result.points[:, :2], axis=0) - np.median(pose[valid_pose, :2], axis=0)
            distance = np.linalg.norm(offset)
            if distance > self.config.max_temporal_xy_offset:
                offset *= self.config.max_temporal_xy_offset / distance
            self._xy_offset = offset
        return result


def transform_high_radar_to_low_radar(
    points: np.ndarray,
    low_extrinsic: tuple[np.ndarray, np.ndarray],
    high_extrinsic: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Transform high-radar xyz directly into low-radar coordinates."""
    return transform_points_between_radars(points, high_extrinsic, low_extrinsic)


def load_high_point_groups_low_radar(
    path: Path | None,
    low_extrinsic: tuple[np.ndarray, np.ndarray],
    high_extrinsic: tuple[np.ndarray, np.ndarray],
    tlv_value: int | None = None,
    tlv_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load high-radar xyz in the low-radar frame filtered by TLV type."""
    empty = np.empty((0, 3), dtype=np.float64)
    if path is None:
        return empty
    if tlv_value is not None and tlv_range is not None:
        raise ValueError("Use either tlv_value or tlv_range, not both")
    if tlv_range is not None and tlv_range[0] > tlv_range[1]:
        raise ValueError("tlv_range must be ordered as (minimum, maximum)")

    rows = load_high_point_rows_low_radar(path, low_extrinsic, high_extrinsic)
    points = rows[:, :3]
    tlv = np.rint(rows[:, POINT_STATE_COLUMN]).astype(np.int16)
    if tlv_value is not None:
        return points[tlv == tlv_value]
    if tlv_range is not None:
        minimum, maximum = tlv_range
        return points[(tlv >= minimum) & (tlv <= maximum)]
    return points


def load_high_point_rows_low_radar(
    path: Path | str,
    low_extrinsic: tuple[np.ndarray, np.ndarray],
    high_extrinsic: tuple[np.ndarray, np.ndarray],
    expected_columns: int | None = None,
) -> np.ndarray:
    """Load finite high-radar rows and transform xyz into the low-radar frame."""
    raw_points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if raw_points.size == 0:
        columns = expected_columns or (raw_points.shape[1] if raw_points.ndim == 2 else 6)
        return np.empty((0, columns), dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] <= POINT_STATE_COLUMN:
        raise ValueError(f"Expected N x M point cloud with state column in {path}")
    if expected_columns is not None and raw_points.shape[1] != expected_columns:
        raise ValueError(f"Expected N x {expected_columns} point cloud in {path}")

    finite = np.isfinite(raw_points[:, :3]).all(axis=1)
    finite &= np.isfinite(raw_points[:, POINT_STATE_COLUMN])
    raw_points = raw_points[finite]
    xyz_low = transform_high_radar_to_low_radar(raw_points[:, :3], low_extrinsic, high_extrinsic)
    return np.concatenate((xyz_low, raw_points[:, 3:]), axis=1)


def load_high_point_rows_high_radar(
    path: Path | str,
    expected_columns: int | None = None,
) -> np.ndarray:
    """Load finite high-radar point rows without a coordinate transform."""
    raw_points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if raw_points.size == 0:
        columns = expected_columns or (raw_points.shape[1] if raw_points.ndim == 2 else 6)
        return np.empty((0, columns), dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] <= POINT_STATE_COLUMN:
        raise ValueError(f"Expected N x M point cloud with state column in {path}")
    if expected_columns is not None and raw_points.shape[1] != expected_columns:
        raise ValueError(f"Expected N x {expected_columns} point cloud in {path}")
    finite = np.isfinite(raw_points[:, :3]).all(axis=1)
    finite &= np.isfinite(raw_points[:, POINT_STATE_COLUMN])
    return raw_points[finite]


def select_complete_pose_box_points(
    points: np.ndarray,
    pose: np.ndarray,
    *,
    padding: float = 0.15,
    minimum_points: int = 10,
) -> np.ndarray:
    """Return points inside a fixed-padding box around one complete pose."""
    raw_points = np.asarray(points)
    pose = np.asarray(pose, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError("points must have shape N x M with M >= 3")
    if pose.shape != (17, 3):
        raise ValueError("pose must have shape 17 x 3")
    if padding < 0 or minimum_points < 1:
        raise ValueError("padding must be non-negative and minimum_points must be positive")
    if not np.isfinite(pose).all():
        return raw_points[:0]
    finite = np.isfinite(raw_points[:, :3]).all(axis=1)
    lower = pose.min(axis=0) - padding
    upper = pose.max(axis=0) + padding
    finite_points = raw_points[finite]
    inside = np.all(
        (finite_points[:, :3] >= lower) & (finite_points[:, :3] <= upper), axis=1
    )
    selected = finite_points[inside]
    return selected if len(selected) >= minimum_points else raw_points[:0]


def expand_box_seed_indices(
    points: np.ndarray,
    seed_indices: np.ndarray,
    box_corners: np.ndarray,
    xy_radius: float,
    z_radius: float,
) -> np.ndarray:
    """Add points directly neighboring box seeds within limited XY and Z bounds."""
    raw_points = np.asarray(points)
    seeds = np.asarray(seed_indices, dtype=int)
    corners = np.asarray(box_corners, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError("points must have shape N x M with M >= 3")
    xyz = raw_points[:, :3]
    if corners.shape != (8, 3):
        raise ValueError("box_corners must have shape 8 x 3")
    if min(xy_radius, z_radius) <= 0:
        raise ValueError("xy_radius and z_radius must be positive")
    if not len(seeds):
        return seeds
    if seeds.min() < 0 or seeds.max() >= len(xyz):
        raise ValueError("seed_indices are outside points")

    lower, upper = corners.min(axis=0), corners.max(axis=0)
    candidate_mask = np.all((xyz[:, :2] >= lower[:2]) & (xyz[:, :2] <= upper[:2]), axis=1)
    candidate_mask &= (xyz[:, 2] >= lower[2] - z_radius) & (xyz[:, 2] <= upper[2] + z_radius)
    candidate_mask[seeds] = True
    candidate_indices = np.flatnonzero(candidate_mask)
    offsets = xyz[candidate_indices, None, :] - xyz[seeds, :]
    near_seed = (
        (np.linalg.norm(offsets[..., :2], axis=-1) <= xy_radius)
        & (np.abs(offsets[..., 2]) <= z_radius)
    ).any(axis=1)
    return candidate_indices[near_seed]


def resolve_person_point_overlap(
    points: np.ndarray,
    seeds: list[np.ndarray],
    candidates: list[np.ndarray],
    centers: np.ndarray,
    ambiguity_margin: float = 0.10,
) -> list[np.ndarray]:
    """Resolve shared candidates in low-radar XY; preserve unambiguous box seeds."""
    if len(candidates) < 2:
        return candidates
    if len(seeds) != len(candidates) or np.asarray(centers).shape != (len(seeds), 3):
        raise ValueError("seeds, candidates and centers must describe the same people")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")
    claims = np.zeros((len(candidates), len(points)), dtype=bool)
    seed_claims = np.zeros_like(claims)
    for row, (seed, candidate) in enumerate(zip(seeds, candidates, strict=True)):
        claims[row, candidate] = True
        seed_claims[row, seed] = True
    shared = claims.sum(axis=0) > 1
    if not shared.any():
        return candidates
    seed_counts = seed_claims.sum(axis=0)
    references = np.asarray(centers, dtype=float)[:, :2].copy()
    for row in range(len(seeds)):
        exclusive = seed_claims[row] & (seed_counts == 1)
        # Very sparse exclusive points may be a limb or an outlier.
        if exclusive.sum() >= 5:
            references[row] = np.median(points[exclusive, :2], axis=0)
    for index in np.flatnonzero(shared):
        owners = np.flatnonzero(claims[:, index])
        claims[:, index] = False
        if seed_counts[index] == 1:
            claims[np.flatnonzero(seed_claims[:, index])[0], index] = True
            continue
        distances = np.linalg.norm(references[owners] - points[index, :2], axis=1)
        order = np.argsort(distances)
        if distances[order[1]] - distances[order[0]] > ambiguity_margin:
            claims[owners[order[0]], index] = True
    return [np.flatnonzero(mask) for mask in claims]


def _distance_to_skeleton(
    points: np.ndarray,
    pose: np.ndarray,
    skeleton: list[list[int]] | tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Return each point's shortest distance to a valid skeleton segment."""
    segments = [
        (pose[start], pose[end])
        for start, end in skeleton
        if start < len(pose)
        and end < len(pose)
        and np.isfinite(pose[start]).all()
        and np.isfinite(pose[end]).all()
    ]
    if not segments:
        joints = pose[np.isfinite(pose).all(axis=1)]
        return np.linalg.norm(points[:, None, :] - joints[None, :, :], axis=-1).min(axis=1)

    starts = np.asarray([segment[0] for segment in segments])
    vectors = np.asarray([end - start for start, end in segments])
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    offsets = points[:, None, :] - starts[None, :, :]
    positions = np.clip(
        np.einsum("nsi,si->ns", offsets, vectors) / np.maximum(lengths_squared, 1e-12),
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + positions[:, :, None] * vectors[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def _nearest_distance(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return nearest Euclidean distance without introducing a spatial-index dependency."""
    return np.linalg.norm(points[:, None, :] - references[None, :, :], axis=-1).min(axis=1)


def _grow_xy_cluster(points: np.ndarray, seed_mask: np.ndarray, radius: float) -> np.ndarray:
    """Grow the complete XY-connected component containing the trusted seeds."""
    cluster = seed_mask.copy()
    frontier = seed_mask.copy()
    while frontier.any():
        remaining = ~cluster
        if not remaining.any():
            break
        connected = _nearest_distance(points[remaining, :2], points[frontier, :2]) <= radius
        frontier = np.zeros(len(points), dtype=bool)
        frontier[np.flatnonzero(remaining)[connected]] = True
        cluster |= frontier
    return cluster


def associate_human_points(
    points: np.ndarray,
    pose: np.ndarray,
    *,
    skeleton: list[list[int]] | tuple[tuple[int, int], ...],
    config: PoseAssociationConfig = PoseAssociationConfig(),
) -> PoseAssociationResult:
    """Associate a complete core-supported human cluster or reject the frame."""
    raw_points = np.asarray(points)
    pose_array = np.asarray(pose, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError(f"points must have shape N x M (M >= 3), got {raw_points.shape}")
    if pose_array.shape != (17, 3):
        raise ValueError(f"pose must have shape 17 x 3, got {pose_array.shape}")

    valid_joints = np.isfinite(pose_array).all(axis=1)
    empty_points = raw_points[:0].copy()
    empty_indices = np.empty(0, dtype=np.int64)
    if not valid_joints.any():
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, np.full(6, np.nan))

    core_padding = np.asarray(config.core_bbox_padding, dtype=np.float64)
    expansion_padding = np.asarray(config.expansion_padding, dtype=np.float64)

    pose_points = pose_array[valid_joints]
    core_min = pose_points.min(axis=0) - core_padding
    core_max = pose_points.max(axis=0) + core_padding
    expansion_min = pose_points.min(axis=0) - expansion_padding
    expansion_max = pose_points.max(axis=0) + expansion_padding
    bbox = np.concatenate((expansion_min, expansion_max))

    finite_points = np.isfinite(raw_points[:, :3]).all(axis=1)
    original_indices = np.flatnonzero(finite_points)
    xyz = raw_points[finite_points, :3].astype(np.float64, copy=False)
    if len(xyz) == 0:
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, bbox)

    in_expansion = np.all((xyz >= expansion_min) & (xyz <= expansion_max), axis=1)
    candidate_indices = np.flatnonzero(in_expansion)
    if len(candidate_indices) == 0:
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, bbox)

    candidates = xyz[candidate_indices]
    core_mask = np.all((candidates >= core_min) & (candidates <= core_max), axis=1)
    if core_mask.sum() < config.min_core_points:
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, bbox)

    accepted = core_mask.copy()
    frontier = core_mask.copy()
    core_points = candidates[core_mask]
    outside_core = ~core_mask
    allowed = np.zeros(len(candidates), dtype=bool)
    allowed[outside_core] = (
        _distance_to_skeleton(candidates[outside_core], pose_array, skeleton) <= config.association_distance
    )
    close_to_core = np.zeros(len(candidates), dtype=bool)
    close_to_core[outside_core] = (
        _nearest_distance(candidates[outside_core], core_points) <= config.max_core_distance
    )
    for _ in range(config.max_growth_steps):
        remaining = ~accepted
        if not remaining.any():
            break
        # Require a short local connection and a bounded distance from the core box.
        close_to_frontier = _nearest_distance(candidates[remaining], candidates[frontier]) <= config.growth_radius
        new_points = allowed[remaining] & close_to_frontier & close_to_core[remaining]
        frontier = np.zeros(len(candidates), dtype=bool)
        frontier[np.flatnonzero(remaining)[new_points]] = True
        if not frontier.any():
            break
        accepted |= frontier

    clusters = []
    remaining_core = core_mask.copy()
    while remaining_core.any():
        seed = np.zeros(len(candidates), dtype=bool)
        seed[np.flatnonzero(remaining_core)[0]] = True
        cluster = _grow_xy_cluster(candidates, seed, config.cluster_radius)
        clusters.append(cluster)
        remaining_core &= ~cluster

    core_support = np.asarray([(cluster & core_mask).sum() for cluster in clusters])
    best_support = core_support.max()
    if best_support < config.min_core_points or np.count_nonzero(core_support == best_support) != 1:
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, bbox)

    cluster = clusters[int(core_support.argmax())]
    cluster_count = int(cluster.sum())
    capture_ratio = (accepted & cluster).sum() / cluster_count
    if cluster_count < config.min_cluster_points or capture_ratio < config.min_capture_ratio:
        return PoseAssociationResult(empty_points, empty_indices, empty_indices, bbox)

    selected_candidate_indices = candidate_indices[cluster | core_mask]
    selected_indices = original_indices[selected_candidate_indices]
    core_indices = original_indices[candidate_indices[core_mask]]
    return PoseAssociationResult(raw_points[selected_indices], selected_indices, core_indices, bbox)
