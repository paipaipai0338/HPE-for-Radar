from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def load_coco_skeleton_config(path: Path | str) -> dict:
    """Load the COCO-17 joint order and skeleton edges."""
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("joint_count") != 17 or not isinstance(config.get("skeleton"), list):
        raise ValueError(f"Invalid COCO skeleton configuration: {path}")
    return config


def ensure_pose_n_j_3(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    return poses[None] if poses.ndim == 2 else poses


def complete_pose_mask(poses: np.ndarray) -> np.ndarray:
    """Return the people whose joint coordinates are all finite."""
    poses = ensure_pose_n_j_3(poses)
    return np.isfinite(poses).all(axis=(1, 2))


def load_pose_pickle(path: Path | str) -> np.ndarray:
    """Load one camera-frame pose array as N x 17 x 3."""
    with Path(path).open("rb") as file:
        poses = ensure_pose_n_j_3(pickle.load(file))
    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(f"Expected N x 17 x 3 poses in {path}, got {poses.shape}")
    return poses


def load_extrinsic_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load p_radar = R_est @ p_camera + t_est."""
    with np.load(path) as data:
        rotation = np.asarray(data["R_est"], dtype=np.float64)
        translation = np.asarray(data["t_est"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Unexpected extrinsic shapes in {path}")
    return rotation, translation


def transform_points_camera_to_radar(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform xyz points with p_radar = R @ p_camera + t."""
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def transform_points_radar_to_camera(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Invert p_radar = R @ p_camera + t."""
    return (np.asarray(points, dtype=np.float64) - translation) @ rotation


def transform_points_between_radars(
    points: np.ndarray,
    source_extrinsic: tuple[np.ndarray, np.ndarray],
    target_extrinsic: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Transform radar points directly between two calibrated radar frames."""
    source_rotation, source_translation = source_extrinsic
    target_rotation, target_translation = target_extrinsic
    relative_rotation = target_rotation @ source_rotation.T
    relative_translation = target_translation - relative_rotation @ source_translation
    return (
        np.asarray(points, dtype=np.float64) @ relative_rotation.T
        + relative_translation
    )


def load_pose_sequence_radar(
    pose_dir: Path | str,
    extrinsic_path: Path | str,
) -> tuple[list[Path], list[np.ndarray]]:
    """Load camera poses and transform every frame to one radar frame."""
    pose_paths = sorted(Path(pose_dir).glob("*.pkl"))
    rotation, translation = load_extrinsic_npz(extrinsic_path)
    frames = [
        transform_points_camera_to_radar(load_pose_pickle(path), rotation, translation)
        for path in pose_paths
    ]
    return pose_paths, frames
