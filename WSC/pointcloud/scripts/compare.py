from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-association-comparison")

import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from tqdm import tqdm


# ------------------------------- Configuration -------------------------------
DATASET_ROOT = Path("/mnt/huawei")
DATE = "20260723"
GROUP = "group_030"
FPS = 10
MAX_MATCH_DELTA_SECONDS = 0.10
MAX_FRAME_GAP_SECONDS = 0.18
HIGH_POSE_BOX_PADDING = 0.15
HIGH_POSE_BOX_MIN_POINTS = 10
HIGH_POSE_BOX_XY_RADIUS = 0.20
HIGH_POSE_BOX_Z_RADIUS = 0.25
WEB_HOST = "0.0.0.0"
WEB_PORT = 8988
AXIS_LIMITS = ((0, 5), (-3, 3), (-2, 2))
SHOW_ALL_POINTS = True


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.pose_io import (
    load_coco_skeleton_config,
    load_extrinsic_npz,
    load_pose_pickle,
    load_pose_sequence_radar,
    transform_points_between_radars,
    transform_points_camera_to_radar,
)
from src.common.sync import match_nearest_paths, timestamp_seconds
from src.pointcloud.utils.pointcloud_io import (
    POINT_STATE_COLUMN,
    PoseAssociationConfig,
    TemporalPoseAssociator,
    expand_box_seed_indices,
    load_high_point_rows_high_radar,
)
from src.pointcloud.utils.skeleton_plot import configure_skeleton_axes, draw_skeleton


ASSOCIATION_CONFIG = PoseAssociationConfig()
BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)

matplotlib.rcParams.update(
    {
        "webagg.address": WEB_HOST,
        "webagg.port": WEB_PORT,
        "webagg.port_retries": 20,
        "webagg.open_in_browser": False,
    }
)


@dataclass(frozen=True, slots=True)
class FrameComparison:
    point_path: Path | None
    delta_seconds: float
    old_indices: np.ndarray
    new_indices: np.ndarray
    box_corners: tuple[np.ndarray, ...]


def pose_box(pose: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = pose.min(axis=0) - padding
    upper = pose.max(axis=0) + padding
    corners = np.asarray(
        [[x, y, z] for x in (lower[0], upper[0])
         for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
    )
    return lower, upper, corners


def draw_box(ax, corners: np.ndarray) -> None:
    for start, end in BOX_EDGES:
        ax.plot(*corners[[start, end]].T, color="#2ca02c", linewidth=1.2)


def prepare_comparisons(
    pose_paths: list[Path],
    point_matches,
    low_extrinsic,
    high_extrinsic,
    skeleton,
) -> list[FrameComparison]:
    associators: dict[int, TemporalPoseAssociator] = {}
    comparisons = []
    frames = zip(pose_paths, point_matches, strict=True)
    for pose_path, (point_path, delta_seconds) in tqdm(
        frames, total=len(pose_paths), desc="compare associations", unit="frame"
    ):
        if point_path is None:
            associators.clear()
            comparisons.append(FrameComparison(None, delta_seconds, np.empty(0, int), np.empty(0, int), ()))
            continue

        rows_high = load_high_point_rows_high_radar(point_path, expected_columns=6)
        rows_low = rows_high.copy()
        rows_low[:, :3] = transform_points_between_radars(
            rows_high[:, :3], high_extrinsic, low_extrinsic
        )
        poses_camera = load_pose_pickle(pose_path)
        poses_low = transform_points_camera_to_radar(poses_camera, *low_extrinsic)
        poses_high = transform_points_camera_to_radar(poses_camera, *high_extrinsic)
        old_indices, new_indices, boxes = [], [], []

        for person_index, (pose_low, pose_high) in enumerate(zip(poses_low, poses_high, strict=True)):
            old = associators.setdefault(
                person_index,
                TemporalPoseAssociator(ASSOCIATION_CONFIG, MAX_FRAME_GAP_SECONDS),
            ).associate(
                rows_low,
                pose_low,
                timestamp_seconds(point_path),
                skeleton=skeleton,
            )
            old_indices.append(old.point_indices)
            if np.isfinite(pose_high).all():
                lower, upper, corners = pose_box(pose_high, HIGH_POSE_BOX_PADDING)
                inside = np.all(
                    (rows_high[:, :3] >= lower) & (rows_high[:, :3] <= upper), axis=1
                )
                selected = np.flatnonzero(inside)
                if len(selected) >= HIGH_POSE_BOX_MIN_POINTS:
                    corners_low = transform_points_between_radars(
                        corners, high_extrinsic, low_extrinsic
                    )
                    new_indices.append(expand_box_seed_indices(
                        rows_low,
                        selected,
                        corners_low,
                        HIGH_POSE_BOX_XY_RADIUS,
                        HIGH_POSE_BOX_Z_RADIUS,
                    ))
                boxes.append(transform_points_between_radars(
                    corners, high_extrinsic, low_extrinsic
                ))

        combine = lambda values: np.unique(np.concatenate(values)) if values else np.empty(0, dtype=int)
        comparisons.append(FrameComparison(
            point_path,
            delta_seconds,
            combine(old_indices),
            combine(new_indices),
            tuple(boxes),
        ))
    return comparisons


def main() -> None:
    group_root = DATASET_ROOT / DATE / "data_collection" / GROUP
    calibration_root = DATASET_ROOT / DATE / "calib"
    pose_paths, pose_frames = load_pose_sequence_radar(
        group_root / "camera results" / "smoothed 3D",
        calibration_root / "extrinsic_img_to_radar_low.npz",
    )
    point_paths = sorted((group_root / "dpct高位机" / "PC").glob("*.npy"))
    if not pose_paths or not point_paths:
        raise FileNotFoundError(f"Missing pose or point cloud data in {group_root}")

    low_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_low.npz")
    high_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_high.npz")
    skeleton = load_coco_skeleton_config(
        PROJECT_ROOT / "src/pointcloud/config/coco_skeleton.json"
    )["skeleton"]
    point_matches = match_nearest_paths(pose_paths, point_paths, MAX_MATCH_DELTA_SECONDS)
    comparisons = prepare_comparisons(
        pose_paths, point_matches, low_extrinsic, high_extrinsic, skeleton
    )

    figure = plt.figure(figsize=(18, 6.5), dpi=105)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    figure.subplots_adjust(bottom=0.18, wspace=0.08)
    slider = Slider(
        figure.add_axes([0.18, 0.06, 0.64, 0.035]),
        "frame", 0, len(pose_paths) - 1, valinit=0, valstep=1,
    )
    playing = [True]

    def scatter(ax, points: np.ndarray, indices: np.ndarray, color: str, size=8) -> None:
        if len(indices):
            ax.scatter(*points[indices].T, s=size, c=color, alpha=0.75, depthshade=False)

    def scatter_by_state(ax, points: np.ndarray, rows: np.ndarray, indices: np.ndarray) -> None:
        if not len(indices):
            return
        states = np.rint(rows[indices, POINT_STATE_COLUMN]).astype(np.int8)
        scatter(ax, points, indices[states == 1], "#d62728")
        scatter(ax, points, indices[(states >= 2) & (states <= 7)], "#1f77b4")

    def draw_frame(value: float) -> None:
        index = int(value)
        comparison = comparisons[index]
        points = np.empty((0, 3))
        rows = np.empty((0, 6))
        if comparison.point_path is not None:
            rows = load_high_point_rows_high_radar(comparison.point_path, expected_columns=6)
            points = transform_points_between_radars(
                rows[:, :3], high_extrinsic, low_extrinsic
            )
        old_only = np.setdiff1d(comparison.old_indices, comparison.new_indices)
        new_only = np.setdiff1d(comparison.new_indices, comparison.old_indices)
        common = np.intersect1d(comparison.old_indices, comparison.new_indices)
        status = "playing" if playing[0] else "paused"
        titles = (
            f"Old association  points={len(comparison.old_indices)}",
            f"High-pose box  points={len(comparison.new_indices)}",
            f"Difference  common={len(common)}  old-only={len(old_only)}  new-only={len(new_only)}",
        )

        for ax, title in zip(axes, titles, strict=True):
            ax.cla()
            configure_skeleton_axes(
                ax,
                AXIS_LIMITS,
                f"{title}\n{GROUP}  {index + 1}/{len(pose_paths)}  {status}  "
                f"Δt={comparison.delta_seconds * 1_000:.1f} ms",
            )
            if SHOW_ALL_POINTS and len(points):
                ax.scatter(*points.T, s=3, c="#d0d0d0", alpha=0.25, depthshade=False)
            for pose in pose_frames[index]:
                draw_skeleton(ax, pose, skeleton)

        point_counts = (
            len(comparison.old_indices),
            len(comparison.new_indices),
            len(old_only) + len(new_only),
        )
        for ax, point_count in zip(axes, point_counts, strict=True):
            ax.text2D(
                0.03,
                0.95,
                f"points: {point_count}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
            )

        scatter_by_state(axes[0], points, rows, comparison.old_indices)
        scatter_by_state(axes[1], points, rows, comparison.new_indices)
        for corners in comparison.box_corners:
            draw_box(axes[1], corners)
        scatter(axes[2], points, old_only, "#1f77b4", 11)
        scatter(axes[2], points, new_only, "#d62728", 11)
        axes[0].legend(handles=[
            Line2D([], [], marker=".", linestyle="", color="#d62728", label="TLV1 dynamic"),
            Line2D([], [], marker=".", linestyle="", color="#1f77b4", label="TLV2-7 micro/static"),
            Line2D([], [], color="#222222", label="pose"),
        ])
        axes[1].legend(handles=[
            Line2D([], [], marker=".", linestyle="", color="#d62728", label="TLV1 dynamic"),
            Line2D([], [], marker=".", linestyle="", color="#1f77b4", label="TLV2-7 micro/static"),
            Line2D([], [], color="#2ca02c", label="high-radar box"),
            Line2D([], [], color="#222222", label="pose"),
        ])
        axes[2].legend(handles=[
            Line2D([], [], marker=".", linestyle="", color="#1f77b4", label="old only"),
            Line2D([], [], marker=".", linestyle="", color="#d62728", label="new only"),
            Line2D([], [], color="#222222", label="pose"),
        ])
        figure.canvas.draw_idle()

    def on_key_press(event) -> None:
        index = int(slider.val)
        if event.key == "left":
            slider.set_val(max(0, index - 1))
        elif event.key == "right":
            slider.set_val(min(len(pose_paths) - 1, index + 1))
        elif event.key in {" ", "space"}:
            playing[0] = not playing[0]
            draw_frame(slider.val)

    def advance() -> None:
        if playing[0]:
            slider.set_val((int(slider.val) + 1) % len(pose_paths))

    slider.on_changed(draw_frame)
    figure.canvas.mpl_connect("key_press_event", on_key_press)
    timer = figure.canvas.new_timer(interval=round(1_000 / FPS))
    timer.add_callback(advance)
    timer.start()
    draw_frame(0)
    print("WebAgg starting; open the URL below. Space pauses playback.")
    plt.show()


if __name__ == "__main__":
    main()
