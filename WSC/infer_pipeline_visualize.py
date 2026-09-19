from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar-pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.widgets import Slider
from tqdm import tqdm

from src.common.gif import export_gif
from src.common.labels import LABEL_NAMES, action_label_index, load_action_labels
from src.common.metrics import classification_metrics
from src.common.pose_io import (
    complete_pose_mask,
    load_coco_skeleton_config,
    load_extrinsic_npz,
    load_pose_pickle,
    transform_points_between_radars,
    transform_points_camera_to_radar,
)
from src.common.sync import match_nearest_paths, timestamp_seconds
from src.detection.inference import Detection, FullTracker, detect_people, load_detector
from src.detection.src.models.modules.processor import PocPaddingProcessor
from src.pointcloud.data import prepare_behavior_window
from src.pointcloud.models import BehaviorModel, BehaviorModelConfig
from src.pointcloud.utils.pointcloud_io import (
    POINT_STATE_COLUMN, expand_box_seed_indices, resolve_person_point_overlap,
)
from src.pointcloud.utils.postprocess import (
    CausalBehaviorPostprocessor,
    StaticInferenceGate,
    point_postprocess_features,
)
from src.pointcloud.utils.skeleton_plot import draw_skeleton


# ------------------------------- Configuration -------------------------------
DATASET_ROOT = Path("/mnt/huawei")
DATE = "20260912"
GROUP = "group_108"
DETECTION_CHECKPOINT = Path("src/detection/ckpt/ckpt_exp5")
DETECTION_VOXEL_CONFIG = Path("src/detection/src/configs/voxel.yaml")
BEHAVIOR_CHECKPOINT = Path(
    "outputs/pointcloud/runs/model_pointnet_8dim_t16_xy_expansion_0719_exp2_z/best.pt"
)
DEVICE = "cuda:1"
DETECTION_SCORE_THRESHOLD = 0.40
DETECTION_BOX_PADDING = (0.10, 0.10, 0.10)
DETECTION_NMS_IOU_THRESHOLD = 0.25
TRACK_MAX_AGE_SECONDS = 0.5
MAX_FRAME_GAP_SECONDS = 0.50
PERSON_XY_EXPANSION_RADIUS = 0.20
PERSON_Z_EXPANSION_RADIUS = 0.25
PERSON_OVERLAP_MARGIN = 0.10
MAX_POINTS = 128
ENABLE_POSTPROCESS = True
SHOW_INFERENCE_DETAILS = False
MAX_POSE_DELTA_SECONDS = 0.05
PRESERVE_Z_HEIGHT = True
ENABLE_REGION_FILTER = True
REGION_X_LIMITS = (0.0, 4.0)
REGION_Y_LIMITS = (-3.0, 3.0)
FPS = 10
GIF_DPI = 70
AXIS_LIMITS = ((0, 6), (-3, 3), (-2, 2))
WEB_HOST = "0.0.0.0"
WEB_PORT = 8988
EXPORT_GIF = True
GIF_PATH = Path("outputs/pointcloud/gifs/108.gif")


matplotlib.rcParams.update(
    {
        "webagg.address": WEB_HOST,
        "webagg.port": WEB_PORT,
        "webagg.port_retries": 20,
        "webagg.open_in_browser": False,
    }
)


@dataclass(slots=True)
class Track:
    identifier: int
    center: np.ndarray
    timestamp: float
    frames: list[np.ndarray] = field(default_factory=list)
    postprocessor: CausalBehaviorPostprocessor = field(
        default_factory=CausalBehaviorPostprocessor
    )
    label: int | None = None
    raw_label: int | None = None
    confidence: float | None = None
    inference_gate: StaticInferenceGate = field(default_factory=StaticInferenceGate)
    model_ran: bool = False


@dataclass(frozen=True, slots=True)
class PersonResult:
    detection: Detection
    behavior_point_indices: np.ndarray
    box_corners: np.ndarray
    track_id: int
    label: int | None
    raw_label: int | None
    confidence: float | None
    holding: bool = False


def select_device(name: str) -> torch.device:
    device = torch.device(name)
    index = device.index or 0
    if device.type == "cuda" and (
        not torch.cuda.is_available() or index >= torch.cuda.device_count()
    ):
        print(f"Warning: {name} is unavailable; using CPU")
        return torch.device("cpu")
    return device


def load_behavior_model(path: Path, device: torch.device) -> tuple[BehaviorModel, BehaviorModelConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = BehaviorModelConfig(**checkpoint["model_config"])
    model = BehaviorModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), config


def classify_track(
    model: BehaviorModel,
    config: BehaviorModelConfig,
    track: Track,
    points: np.ndarray,
    device: torch.device,
) -> tuple[int | None, int | None, float | None]:
    track.model_ran = False
    track.frames.append(points)
    track.frames = track.frames[-config.sequence_length :]
    if len(track.frames) < config.sequence_length:
        return track.label, track.raw_label, track.confidence
    if ENABLE_POSTPROCESS and not track.inference_gate.should_infer(
        points,
        track.postprocessor.established,
        track.postprocessor.transition_pending,
    ):
        return track.label, track.raw_label, track.confidence
    point_tensor, mask, statistics = prepare_behavior_window(
        track.frames,
        config,
        MAX_POINTS,
        preserve_z_height=PRESERVE_Z_HEIGHT,
    )
    with torch.inference_mode():
        probabilities = model(
            point_tensor[None].to(device),
            mask[None].to(device),
            statistics[None].to(device),
        ).softmax(dim=1)[0].cpu().numpy()
    track.model_ran = True
    raw_label = int(probabilities.argmax())
    label = raw_label
    if ENABLE_POSTPROCESS:
        dynamic_ratio, point_count, robust_height = point_postprocess_features(points)
        label = track.postprocessor.update(
            probabilities, dynamic_ratio, point_count, robust_height
        )
    track.label = label
    track.raw_label = raw_label
    track.confidence = float(probabilities[label])
    return track.label, track.raw_label, track.confidence


def classify_available_points(
    model: BehaviorModel,
    config: BehaviorModelConfig,
    track: Track,
    points: np.ndarray,
    device: torch.device,
) -> tuple[int | None, int | None, float | None]:
    """Restart behavior inference after a tracked detection loses all points."""
    if not len(points):
        track.frames.clear()
        track.postprocessor.reset()
        track.inference_gate.reset()
        track.label = track.raw_label = track.confidence = None
        track.model_ran = False
        return track.label, track.raw_label, track.confidence
    return classify_track(model, config, track, points, device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def box_edges(corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    return [(corners[start], corners[end]) for start, end in indices]


def box_corners(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [box[x], box[1 + y], box[2 + z]]
            for x in (0, 3)
            for y in (0, 3)
            for z in (0, 3)
        ]
    )


def load_point_frames(
    path: Path,
    low_extrinsic: tuple[np.ndarray, np.ndarray],
    high_extrinsic: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if high.ndim != 2 or high.shape[1] != 6:
        raise ValueError(f"Expected N x 6 point cloud in {path}")
    finite = np.isfinite(high[:, :4]).all(axis=1) & np.isfinite(high[:, POINT_STATE_COLUMN])
    high = high[finite]
    low = high.copy()
    low[:, :3] = transform_points_between_radars(
        high[:, :3], high_extrinsic, low_extrinsic
    )
    return high, low


def filter_region_points(
    high_points: np.ndarray,
    low_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not ENABLE_REGION_FILTER or not len(low_points):
        return high_points, low_points
    x_min, x_max = REGION_X_LIMITS
    y_min, y_max = REGION_Y_LIMITS
    inside = (
        (low_points[:, 0] >= x_min) & (low_points[:, 0] <= x_max)
        & (low_points[:, 1] >= y_min) & (low_points[:, 1] <= y_max)
    )
    return high_points[inside], low_points[inside]


def pose_in_region(pose: np.ndarray) -> bool:
    if not ENABLE_REGION_FILTER:
        return True
    finite = np.isfinite(pose).all(axis=1)
    if not finite.any():
        return False
    center = pose[finite].mean(axis=0)
    return (
        REGION_X_LIMITS[0] <= center[0] <= REGION_X_LIMITS[1]
        and REGION_Y_LIMITS[0] <= center[1] <= REGION_Y_LIMITS[1]
    )


def transform_box_corners(
    box: np.ndarray,
    low_extrinsic: tuple[np.ndarray, np.ndarray],
    high_extrinsic: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    corners = box_corners(box)
    return transform_points_between_radars(
        corners, high_extrinsic, low_extrinsic
    ).astype(np.float32)


def print_behavior_metrics(
    name: str,
    labels: list[int],
    predictions: list[int],
) -> None:
    if not labels:
        print(f"{name}: no matched action labels")
        return
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions), LABEL_NAMES)
    details = "  ".join(
        f"{item['label']}[F1={item['f1']:.4f} N={item['support']}]"
        for item in metrics.per_class
    )
    print(f"{name}: accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")
    print(f"{name}_per_class: {details}")
    print(f"{name}_confusion_matrix rows=true cols=predicted")
    print(" " * 13 + " ".join(f"{label:>10}" for label in LABEL_NAMES))
    for label, row in zip(LABEL_NAMES, metrics.confusion_matrix, strict=True):
        print(f"{label:>12} " + " ".join(f"{value:>10}" for value in row))


def evaluate_pipeline_predictions(
    group_root: Path,
    point_paths: list[Path],
    frame_results: list[list[PersonResult]],
    pose_matches: list[tuple[Path | None, float]],
    low_extrinsic: tuple[np.ndarray, np.ndarray],
) -> None:
    labels: list[int] = []
    raw_predictions: list[int] = []
    predictions: list[int] = []
    for path, results, (pose_path, _) in zip(point_paths, frame_results, pose_matches, strict=True):
        if pose_path is None:
            continue
        label_path = group_root / "camera results" / "action label" / f"{pose_path.stem}.npz"
        if not label_path.is_file():
            continue
        try:
            action_labels, valid = load_action_labels(label_path)
            poses = load_pose_pickle(pose_path)
        except (EOFError, OSError, ValueError):
            continue
        person_indices = np.asarray(
            [
                index
                for index in np.flatnonzero(valid & complete_pose_mask(poses))
                if pose_in_region(
                    transform_points_camera_to_radar(poses[index], *low_extrinsic)
                )
            ],
            dtype=int,
        )
        if len(action_labels) != len(poses) or not len(person_indices):
            continue
        pose_centers = transform_points_camera_to_radar(
            poses[person_indices].mean(axis=1), *low_extrinsic
        )
        result_centers = np.asarray([result.box_corners.mean(axis=0) for result in results])
        if not len(result_centers):
            continue
        distances = np.linalg.norm(result_centers[:, None, :2] - pose_centers[None, :, :2], axis=2)
        matched_pose_indices: set[int] = set()
        for result_index in np.argsort(distances.min(axis=1)):
            available = [index for index in range(len(person_indices)) if index not in matched_pose_indices]
            if not available:
                break
            pose_index = min(available, key=lambda index: distances[result_index, index])
            result = results[int(result_index)]
            if result.label is not None and result.raw_label is not None:
                person_index = int(person_indices[pose_index])
                labels.append(action_label_index(action_labels[person_index], label_path))
                raw_predictions.append(result.raw_label)
                predictions.append(result.label)
            matched_pose_indices.add(pose_index)
    print(f"label_matches={len(labels)}")
    print_behavior_metrics("raw", labels, raw_predictions)
    print_behavior_metrics("postprocessed", labels, predictions)


def main() -> None:
    group_root = DATASET_ROOT / DATE / "data_collection" / GROUP
    point_paths = sorted((group_root / "dpct高位机" / "PC").glob("*.npy"))
    if not point_paths:
        raise FileNotFoundError(f"No high-radar point clouds in {group_root}")
    calibration_root = DATASET_ROOT / DATE / "calib"
    low_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_low.npz")
    high_extrinsic = load_extrinsic_npz(calibration_root / "extrinsic_img_to_radar_high.npz")
    device = select_device(DEVICE)
    detector = load_detector(PROJECT_ROOT / DETECTION_CHECKPOINT, device)
    with (PROJECT_ROOT / DETECTION_VOXEL_CONFIG).open(encoding="utf-8") as file:
        processor = PocPaddingProcessor(yaml.safe_load(file))
    behavior_model, behavior_config = load_behavior_model(
        PROJECT_ROOT / BEHAVIOR_CHECKPOINT, device
    )

    frame_points: list[np.ndarray] = []
    frame_results: list[list[PersonResult]] = []
    tracks: dict[int, Track] = {}
    tracker = FullTracker()
    detection_times = []
    behavior_times = []
    frame_times = []
    for path in tqdm(point_paths, desc="detect and classify", unit="frame"):
        synchronize(device)
        frame_start = time.perf_counter()
        try:
            high_points, points = load_point_frames(path, low_extrinsic, high_extrinsic)
            high_points, points = filter_region_points(high_points, points)
        except (EOFError, OSError, ValueError):
            high_points = np.empty((0, 6), dtype=np.float32)
            points = np.empty((0, 6), dtype=np.float32)
        synchronize(device)
        detection_start = time.perf_counter()
        detections = (
            detect_people(
                detector,
                processor,
                high_points,
                device,
                score_threshold=DETECTION_SCORE_THRESHOLD,
                box_padding=DETECTION_BOX_PADDING,
                nms_iou_threshold=DETECTION_NMS_IOU_THRESHOLD,
            )
            if len(points)
            else []
        )
        synchronize(device)
        detection_times.append(time.perf_counter() - detection_start)
        timestamp = timestamp_seconds(path)
        tracked_people = tracker.update(detections, high_points, timestamp)
        results = []
        active_ids = set()
        boxes = [transform_box_corners(p.box, low_extrinsic, high_extrinsic) for p in tracked_people]
        seeds = [np.flatnonzero(np.all(
            (high_points[:, :3] >= p.box[:3]) & (high_points[:, :3] <= p.box[3:]), axis=1
        )) for p in tracked_people]
        candidates = [expand_box_seed_indices(
            points, seed, corners, PERSON_XY_EXPANSION_RADIUS, PERSON_Z_EXPANSION_RADIUS
        ) for seed, corners in zip(seeds, boxes, strict=True)]
        assignments = resolve_person_point_overlap(
            points, seeds, candidates, np.asarray([box.mean(axis=0) for box in boxes]),
            PERSON_OVERLAP_MARGIN,
        )
        for person, corners, seed, behavior_indices in zip(
            tracked_people, boxes, seeds, assignments, strict=True
        ):
            active_ids.add(person.identifier)
            high_detection = Detection(person.box, 1.0, seed)
            center = corners.mean(axis=0)
            track = tracks.setdefault(
                person.identifier,
                Track(person.identifier, center, timestamp),
            )
            if timestamp - track.timestamp > MAX_FRAME_GAP_SECONDS:
                track.frames.clear()
                track.postprocessor.reset()
                track.inference_gate.reset()
                track.label = track.raw_label = track.confidence = None
            track.center = center
            track.timestamp = timestamp
            synchronize(device)
            behavior_start = time.perf_counter()
            label, raw_label, confidence = classify_available_points(
                behavior_model,
                behavior_config,
                track,
                points[behavior_indices],
                device,
            )
            synchronize(device)
            if track.model_ran:
                behavior_times.append(time.perf_counter() - behavior_start)
            results.append(
                PersonResult(
                    high_detection,
                    behavior_indices,
                    corners,
                    track.identifier,
                    label,
                    raw_label,
                    confidence,
                    track.inference_gate.holding if ENABLE_POSTPROCESS else False,
                )
            )
        stale = [
            identifier
            for identifier, track in tracks.items()
            if identifier not in active_ids
            and timestamp - track.timestamp > TRACK_MAX_AGE_SECONDS
        ]
        for identifier in stale:
            del tracks[identifier]
        frame_points.append(points)
        frame_results.append(results)
        synchronize(device)
        frame_times.append(time.perf_counter() - frame_start)

    def timing_summary(name: str, values: list[float]) -> None:
        if values:
            milliseconds = np.asarray(values) * 1000
            print(
                f"{name}: mean={milliseconds.mean():.1f} ms  "
                f"p95={np.quantile(milliseconds, 0.95):.1f} ms  n={len(values)}"
            )

    timing_summary("detection/frame", detection_times)
    timing_summary("behavior/person", behavior_times)
    timing_summary("complete/frame", frame_times)

    pose_paths = sorted((group_root / "camera results" / "smoothed 3D").glob("*.pkl"))
    pose_matches = match_nearest_paths(point_paths, pose_paths, MAX_POSE_DELTA_SECONDS)
    evaluate_pipeline_predictions(
        group_root,
        point_paths,
        frame_results,
        pose_matches,
        low_extrinsic,
    )
    skeleton = load_coco_skeleton_config(
        PROJECT_ROOT / "src/pointcloud/config/coco_skeleton.json"
    )["skeleton"]
    figure = plt.figure(figsize=(9, 7), dpi=110)
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(bottom=0.16)
    slider = Slider(
        figure.add_axes([0.15, 0.06, 0.7, 0.035]),
        "frame", 0, len(point_paths) - 1, valinit=0, valstep=1,
    )
    playing = [True]

    def draw_frame(value: float) -> None:
        index = int(value)
        points = frame_points[index]
        results = frame_results[index]
        selected = np.zeros(len(points), dtype=bool)
        axis.cla()
        for result in results:
            selected[result.behavior_point_indices] = True
        if (~selected).any():
            axis.scatter(*points[~selected, :3].T, s=3, c="#bdbdbd", alpha=0.25)
        colors = plt.cm.tab10.colors
        for result in results:
            color = colors[result.track_id % len(colors)]
            person_points = points[result.behavior_point_indices, :3]
            axis.scatter(*person_points.T, s=8, color=color, alpha=0.75)
            for start, end in box_edges(result.box_corners):
                axis.plot(*np.stack((start, end)).T, color=color, linewidth=1.2)
            label = "waiting"
            if result.label is not None:
                label = LABEL_NAMES[result.label]
                if SHOW_INFERENCE_DETAILS:
                    label += f" {result.confidence:.0%}"
                    if result.holding:
                        label += " (hold, last confidence)"
                    if result.raw_label != result.label:
                        label += f" (raw={LABEL_NAMES[result.raw_label]})"
            position = result.box_corners.mean(axis=0)
            position[2] = result.box_corners[:, 2].max() + 0.15
            axis.text(*position, f"ID {result.track_id}: {label}", color=color, ha="center")
        pose_path, _ = pose_matches[index]
        if pose_path is not None:
            poses = load_pose_pickle(pose_path)
            poses = transform_points_camera_to_radar(poses[complete_pose_mask(poses)], *low_extrinsic)
            for pose in poses:
                if pose_in_region(pose):
                    draw_skeleton(axis, pose, skeleton)
        axis.set(
            xlim=AXIS_LIMITS[0], ylim=AXIS_LIMITS[1], zlim=AXIS_LIMITS[2],
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
            title=f"{DATE}/{GROUP}  {index + 1}/{len(point_paths)}  "
            f"{'playing' if playing[0] else 'paused'}  people={len(results)}",
        )
        figure.canvas.draw_idle()

    def on_key(event) -> None:
        index = int(slider.val)
        if event.key == " ":
            playing[0] = not playing[0]
            draw_frame(index)
        elif event.key == "left":
            slider.set_val(max(0, index - 1))
        elif event.key == "right":
            slider.set_val(min(len(point_paths) - 1, index + 1))

    def advance() -> None:
        if playing[0]:
            slider.set_val((int(slider.val) + 1) % len(point_paths))

    slider.on_changed(draw_frame)
    figure.canvas.mpl_connect("key_press_event", on_key)
    if EXPORT_GIF:
        export_gif(
            figure, draw_frame, len(point_paths), PROJECT_ROOT / GIF_PATH, FPS, dpi=GIF_DPI
        )
        print(f"GIF saved to {PROJECT_ROOT / GIF_PATH}")
    timer = figure.canvas.new_timer(interval=round(1000 / FPS))
    timer.add_callback(advance)
    timer.start()
    draw_frame(0)
    plt.show()


if __name__ == "__main__":
    main()
