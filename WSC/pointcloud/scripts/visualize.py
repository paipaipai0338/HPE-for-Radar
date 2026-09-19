from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar-visualize")

import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider


# ------------------------------- Configuration -------------------------------
DATASET_ROOT = Path("/mnt/huawei")
DATE = "20260703"
GROUP = "group_029"
FPS = 10
MAX_MATCH_DELTA_SECONDS = 0.12
WEB_HOST = "0.0.0.0"
WEB_PORT = 8988
AXIS_LIMITS = ((0, 5), (-3, 3), (-2, 2))
POINT_TLV_VALUE = None
POINT_TLV_RANGE = None
SHOW_UNASSOCIATED_POINTS = True
ASSOCIATION_MODE = "high_pose_box"
HIGH_POSE_BOX_PADDING = 0.15
HIGH_POSE_BOX_MIN_POINTS = 10
HIGH_POSE_BOX_XY_RADIUS = 0.20
HIGH_POSE_BOX_Z_RADIUS = 0.25


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
    PoseAssociationConfig,
    TemporalPoseAssociator,
    expand_box_seed_indices,
    load_high_point_rows_high_radar,
    load_high_point_groups_low_radar,
)
from src.pointcloud.utils.skeleton_plot import configure_skeleton_axes, draw_skeleton

ASSOCIATION_CONFIG = PoseAssociationConfig()
BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


matplotlib.rcParams.update(
    {
        "webagg.address": WEB_HOST,
        "webagg.port": WEB_PORT,
        "webagg.port_retries": 20,
        "webagg.open_in_browser": False,
    }
)


def pose_box_corners(pose: np.ndarray, padding: float) -> np.ndarray:
    """Return the eight corners of a padded axis-aligned pose box."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (17, 3) or not np.isfinite(pose).all():
        return np.empty((0, 3), dtype=np.float64)
    lower = pose.min(axis=0) - padding
    upper = pose.max(axis=0) + padding
    return np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def draw_box(ax, corners: np.ndarray) -> None:
    """Draw a box represented by eight corners in the displayed coordinate frame."""
    for start, end in BOX_EDGES:
        ax.plot(*corners[[start, end]].T, color="#d62728", linewidth=1.2, alpha=0.9)


def main() -> None:
    group_root = DATASET_ROOT / DATE / "data_collection" / GROUP
    calibration_root = DATASET_ROOT / DATE / "calib"
    pose_paths, pose_frames = load_pose_sequence_radar(
        group_root / "camera results" / "smoothed 3D",
        calibration_root / "extrinsic_img_to_radar_low.npz",
    )
    if not pose_paths:
        raise FileNotFoundError(f"No pose files in {group_root}")

    point_paths = sorted((group_root / "dpct高位机" / "PC").glob("*.npy"))
    point_matches = match_nearest_paths(pose_paths, point_paths, MAX_MATCH_DELTA_SECONDS)
    low_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_low.npz")
    high_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_high.npz")
    skeleton = load_coco_skeleton_config(
        PROJECT_ROOT / "src/pointcloud/config/coco_skeleton.json"
    )["skeleton"]

    figure = plt.figure(figsize=(8, 7), dpi=110)
    ax = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(bottom=0.18)
    slider = Slider(figure.add_axes([0.15, 0.06, 0.7, 0.035]), "frame", 0, len(pose_paths) - 1, valinit=0, valstep=1)
    playing = [True]
    associators: dict[int, TemporalPoseAssociator] = {}

    def draw_frame(value: float) -> None:
        index = int(value)
        point_path, delta_seconds = point_matches[index]
        if ASSOCIATION_MODE == "high_pose_box":
            point_rows_high = (
                load_high_point_rows_high_radar(point_path, expected_columns=6)
                if point_path is not None
                else np.empty((0, 6), dtype=np.float64)
            )
            points = transform_points_between_radars(
                point_rows_high[:, :3], high_extrinsic, low_extrinsic
            )
            poses_high = transform_points_camera_to_radar(
                load_pose_pickle(pose_paths[index]), *high_extrinsic
            )
            box_corners = [
                transform_points_between_radars(
                    corners, high_extrinsic, low_extrinsic
                )
                for pose_high in poses_high
                if len(corners := pose_box_corners(pose_high, HIGH_POSE_BOX_PADDING))
            ]
            selected_indices = []
            for pose_high, corners_low in zip(poses_high, box_corners, strict=True):
                lower = pose_high.min(axis=0) - HIGH_POSE_BOX_PADDING
                upper = pose_high.max(axis=0) + HIGH_POSE_BOX_PADDING
                seeds = np.flatnonzero(np.all(
                    (point_rows_high[:, :3] >= lower) & (point_rows_high[:, :3] <= upper), axis=1
                ))
                if len(seeds) >= HIGH_POSE_BOX_MIN_POINTS:
                    selected_indices.append(expand_box_seed_indices(
                        np.column_stack((points, point_rows_high[:, 3:])),
                        seeds,
                        corners_low,
                        HIGH_POSE_BOX_XY_RADIUS,
                        HIGH_POSE_BOX_Z_RADIUS,
                    ))
            associated_points = points[
                np.unique(np.concatenate(selected_indices))
            ] if selected_indices else np.empty((0, 3), dtype=np.float64)
            core_points = np.empty((0, 3), dtype=np.float64)
        else:
            box_corners = []
            points = load_high_point_groups_low_radar(
                point_path,
                low_extrinsic,
                high_extrinsic,
                tlv_value=POINT_TLV_VALUE,
                tlv_range=POINT_TLV_RANGE,
            )
            associated_indices = []
            core_indices = []
            for pose in pose_frames[index]:
                person_index = len(associated_indices)
                associator = associators.setdefault(
                    person_index,
                    TemporalPoseAssociator(ASSOCIATION_CONFIG, MAX_MATCH_DELTA_SECONDS),
                )
                association = associator.associate(
                    points,
                    pose,
                    timestamp_seconds(pose_paths[index]),
                    skeleton=skeleton,
                )
                associated_indices.append(association.point_indices)
                core_indices.append(association.core_indices)
            associated_indices = (
                np.unique(np.concatenate(associated_indices))
                if associated_indices
                else np.empty(0, dtype=np.int64)
            )
            core_indices = (
                np.unique(np.concatenate(core_indices))
                if core_indices
                else np.empty(0, dtype=np.int64)
            )
            associated_points = points[associated_indices]
            core_points = points[core_indices]
        status = "playing" if playing[0] else "paused"
        title = (
            f"{GROUP}  {index + 1}/{len(pose_paths)}  {ASSOCIATION_MODE}  "
            f"{status}  Δt={delta_seconds * 1_000:.1f} ms"
        )
        if point_path is None:
            title += "  (no synchronized point cloud)"

        ax.cla()
        configure_skeleton_axes(ax, AXIS_LIMITS, title)
        if SHOW_UNASSOCIATED_POINTS and len(points):
            ax.scatter(
                *points.T,
                s=4,
                c="#bdbdbd",
                alpha=0.38,
                depthshade=False,
            )
        if len(associated_points):
            ax.scatter(
                *associated_points.T,
                s=7,
                c="#2ca02c",
                alpha=0.6,
                depthshade=False,
            )
        if len(core_points):
            ax.scatter(
                *core_points.T,
                s=14,
                c="#ffbf00",
                alpha=0.95,
                depthshade=False,
            )
        for corners in box_corners:
            draw_box(ax, corners)
        for pose in pose_frames[index]:
            draw_skeleton(ax, pose, skeleton)
        handles = [
            Line2D([], [], marker=".", linestyle="", color="#2ca02c", label="associated human point"),
            Line2D([], [], color="#222222", label="camera pose"),
        ]
        if ASSOCIATION_MODE == "low_pose_association":
            handles.insert(
                1,
                Line2D([], [], marker=".", linestyle="", color="#ffbf00", label="core-box point"),
            )
        else:
            handles.insert(
                1,
                Line2D([], [], color="#d62728", label="high-radar pose box"),
            )
        if SHOW_UNASSOCIATED_POINTS:
            label = "all high-radar points" if ASSOCIATION_MODE == "high_pose_box" else "unassociated point"
            handles.insert(
                0,
                Line2D([], [], marker=".", linestyle="", color="#bdbdbd", label=label),
            )
        ax.legend(handles=handles, loc="upper right")
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
    print("WebAgg starting; open http://<server-ip>:<port shown below>. Space pauses playback.")
    plt.show()


if __name__ == "__main__":
    main()
