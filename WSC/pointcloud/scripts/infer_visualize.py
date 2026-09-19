from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar-inference")

import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------- Configuration -------------------------------
DATASET_ROOT = Path("/mnt/huawei")
DATE = "20260913"
GROUP = "group_020"
CHECKPOINT_PATH = Path("outputs/pointcloud/runs/model_pointnet_8dim_t16_xy_expansion_0719_exp2_z/best.pt")
DEVICE = "cuda:1"
BATCH_SIZE = 256
MAX_POINTS = 128
MAX_SYNC_DELTA_SECONDS = 0.10
MAX_FRAME_GAP_SECONDS = 0.50
ASSOCIATION_MODE = "high_pose_box"
COORDINATE_FRAME = "low"
PRESERVE_Z_HEIGHT = True
ASSOCIATION_CACHE_DIR = Path(
    "outputs/pointcloud/cache/association_high_pose_box_xy_expansion"
)
HIGH_POSE_BOX_PADDING = 0.15
HIGH_POSE_BOX_MIN_POINTS = 1
HIGH_POSE_BOX_XY_RADIUS = 0.20
HIGH_POSE_BOX_Z_RADIUS = 0.25
REQUIRE_SINGLE_PERSON = False
ENABLE_REGION_FILTER = True
REGION_X_LIMITS = (0.0, 4.0)
REGION_Y_LIMITS = (-3.0, 3.0)
FPS = 10
GIF_DPI = 70
WEB_HOST = "0.0.0.0"
WEB_PORT = 8988
AXIS_LIMITS = ((0, 5), (-3, 3), (-2, 2))
SHOW_UNASSOCIATED_POINTS = True
LABEL_HEIGHT_OFFSET = 0.2
ENABLE_POSTPROCESS = True
SHOW_INFERENCE_DETAILS = False
EXPORT_GIF = True
GIF_PATH = Path("outputs/pointcloud/gifs/20.gif")
PERSON_COLORS = ("#2ca02c", "#ff7f0e", "#9467bd", "#17becf")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.gif import export_gif
from src.common.labels import LABEL_NAMES
from src.common.metrics import classification_metrics
from src.common.pose_io import (
    load_coco_skeleton_config,
    load_pose_pickle,
    transform_points_camera_to_radar,
    transform_points_between_radars,
)
from src.pointcloud.data import BehaviorDataset
from src.pointcloud.models import BehaviorModel, BehaviorModelConfig
from src.pointcloud.utils.postprocess import (
    CausalBehaviorPostprocessor, StaticInferenceGate, point_postprocess_features,
)
from src.pointcloud.utils.pointcloud_io import (
    PoseAssociationConfig,
    load_high_point_groups_low_radar,
)
from src.pointcloud.utils.skeleton_plot import configure_skeleton_axes, draw_skeleton


ASSOCIATION_CONFIG = PoseAssociationConfig()


def region_points(points: np.ndarray) -> np.ndarray:
    if not ENABLE_REGION_FILTER or not len(points):
        return points
    x_min, x_max = REGION_X_LIMITS
    y_min, y_max = REGION_Y_LIMITS
    inside = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    return points[inside]


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

matplotlib.rcParams.update(
    {
        "webagg.address": WEB_HOST,
        "webagg.port": WEB_PORT,
        "webagg.port_retries": 20,
        "webagg.open_in_browser": False,
    }
)


def select_device(name: str) -> torch.device:
    requested = torch.device(name)
    index = requested.index or 0
    if requested.type == "cuda" and (not torch.cuda.is_available() or index >= torch.cuda.device_count()):
        print(f"Warning: {name} is unavailable; using CPU")
        return torch.device("cpu")
    return requested


def predict(
    model: BehaviorModel,
    dataset: BehaviorDataset,
    device: torch.device,
) -> np.ndarray:
    probabilities = []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="infer", unit="batch"):
            logits = model(
                batch["points"].to(device),
                batch["point_mask"].to(device),
                batch["shape_statistics"].to(device),
            )
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def print_metrics(name: str, labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    metrics = classification_metrics(labels, predictions, LABEL_NAMES)
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
    return metrics.accuracy, metrics.macro_f1


def associated_points_low(dataset: BehaviorDataset, target) -> np.ndarray:
    """Use gravity coordinates for display and quality metrics without changing model input."""
    points = dataset._associated_points(target)
    if dataset.coordinate_frame == "high":
        low, high = dataset.extrinsics_by_date[target.date]
        points = points.copy()
        points[:, :3] = transform_points_between_radars(points[:, :3], high, low)
    return points


def main() -> None:
    checkpoint_path = CHECKPOINT_PATH if CHECKPOINT_PATH.is_absolute() else PROJECT_ROOT / CHECKPOINT_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = BehaviorModelConfig(**checkpoint["model_config"])
    device = select_device(DEVICE)
    model = BehaviorModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    cache_dir = (
        ASSOCIATION_CACHE_DIR
        if ASSOCIATION_CACHE_DIR.is_absolute()
        else PROJECT_ROOT / ASSOCIATION_CACHE_DIR
    )

    dataset = BehaviorDataset(
        dataset_root=DATASET_ROOT,
        dates=DATE,
        group_names=[GROUP],
        max_points=MAX_POINTS,
        sequence_length=model_config.sequence_length,
        max_sync_delta_seconds=MAX_SYNC_DELTA_SECONDS,
        max_frame_gap_seconds=MAX_FRAME_GAP_SECONDS,
        training=False,
        include_shape_statistics=model_config.use_shape_statistics,
        include_compact_statistics=model_config.use_compact_statistics,
        include_point_count=model_config.use_point_count,
        shape_statistic_dim=model_config.shape_statistic_dim,
        sample_index_cache_dir=PROJECT_ROOT / "outputs/pointcloud/cache/sample_index",
        association_cache_dir=cache_dir,
        association_mode=ASSOCIATION_MODE,
        coordinate_frame=COORDINATE_FRAME,
        preserve_z_height=PRESERVE_Z_HEIGHT,
        region_limits=(REGION_X_LIMITS, REGION_Y_LIMITS) if ENABLE_REGION_FILTER else None,
        high_pose_box_padding=HIGH_POSE_BOX_PADDING,
        high_pose_box_min_points=HIGH_POSE_BOX_MIN_POINTS,
        high_pose_box_xy_radius=HIGH_POSE_BOX_XY_RADIUS,
        high_pose_box_z_radius=HIGH_POSE_BOX_Z_RADIUS,
        require_single_person=REQUIRE_SINGLE_PERSON,
        show_progress=True,
        association_config=ASSOCIATION_CONFIG,
    )
    probabilities = predict(model, dataset, device)
    raw_predictions = probabilities.argmax(axis=1)
    targets = [window[-1] for window in dataset.windows]
    print(
        f"association={ASSOCIATION_MODE}, coordinate_frame={COORDINATE_FRAME}, "
        f"cache={cache_dir.name}"
    )
    postprocess_features = np.asarray(
        [point_postprocess_features(associated_points_low(dataset, target)) for target in targets]
    )
    dynamic_ratios, point_counts, robust_heights = postprocess_features.T
    predictions = raw_predictions.copy()
    holding = np.zeros(len(targets), dtype=bool)
    if ENABLE_POSTPROCESS:
        states: dict[tuple[str, str, int], tuple[CausalBehaviorPostprocessor, StaticInferenceGate, float]] = {}
        for index, target in enumerate(targets):
            key = target.date, target.group_name, target.person_index
            postprocessor, gate, previous_timestamp = states.get(
                key, (CausalBehaviorPostprocessor(), StaticInferenceGate(), -np.inf)
            )
            if target.timestamp_seconds - previous_timestamp > dataset.max_frame_gap_seconds:
                postprocessor.reset()
                gate.reset()
            points = associated_points_low(dataset, target)
            if gate.should_infer(
                points, postprocessor.established, postprocessor.transition_pending
            ):
                predictions[index] = postprocessor.update(probabilities[index], *postprocess_features[index])
            else:
                predictions[index] = postprocessor.state
                holding[index] = True
            states[key] = postprocessor, gate, target.timestamp_seconds
        print(f"static_hold: {holding.sum()}/{len(holding)} windows ({holding.mean():.1%}); "
              "raw comparison still runs inference on every window")
    labels = np.asarray([target.label for target in targets])
    raw_accuracy, raw_macro_f1 = print_metrics("raw", labels, raw_predictions)
    if ENABLE_POSTPROCESS:
        post_accuracy, post_macro_f1 = print_metrics("postprocessed", labels, predictions)
        print(
            f"postprocess_delta: accuracy={post_accuracy - raw_accuracy:+.4f} "
            f"macro_f1={post_macro_f1 - raw_macro_f1:+.4f}"
        )

    low_extrinsic, high_extrinsic = dataset.extrinsics_by_date[DATE]
    skeleton = load_coco_skeleton_config(
        PROJECT_ROOT / "src/pointcloud/config/coco_skeleton.json"
    )["skeleton"]
    frame_indices_by_path: dict[Path, list[int]] = {}
    for index, target in enumerate(targets):
        frame_indices_by_path.setdefault(target.point_path, []).append(index)
    frame_indices = sorted(
        frame_indices_by_path.values(), key=lambda indices: targets[indices[0]].timestamp_seconds
    )
    figure = plt.figure(figsize=(8, 7), dpi=110)
    ax = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(bottom=0.18)
    slider = Slider(figure.add_axes([0.15, 0.06, 0.7, 0.035]), "frame", 0, len(frame_indices) - 1, valinit=0, valstep=1)
    playing = [True]

    def draw_frame(value: float) -> None:
        indices = frame_indices[int(value)]
        frame_targets = [targets[index] for index in indices]
        target = frame_targets[0]
        points = region_points(load_high_point_groups_low_radar(
            target.point_path, low_extrinsic, high_extrinsic
        ))
        status = "playing" if playing[0] else "paused"
        title = f"{DATE}/{GROUP}  {int(value) + 1}/{len(frame_indices)}  {status}"
        if SHOW_INFERENCE_DETAILS:
            results = []
            for index, frame_target in zip(indices, frame_targets, strict=True):
                prediction = int(predictions[index])
                raw_prediction = int(raw_predictions[index])
                confidence = probabilities[index, raw_prediction]
                result = "correct" if prediction == frame_target.label else "wrong"
                results.append(
                    f"person={frame_target.person_index}: {LABEL_NAMES[prediction]} "
                    f"raw={LABEL_NAMES[raw_prediction]} ({confidence:.0%}) "
                    f"truth={LABEL_NAMES[frame_target.label]} {result}"
                )
            title += "\n" + "\n".join(results)

        ax.cla()
        configure_skeleton_axes(ax, AXIS_LIMITS, title)
        if SHOW_UNASSOCIATED_POINTS and len(points):
            ax.scatter(*points.T, s=4, c="#bdbdbd", alpha=0.35, depthshade=False)
        probability_lines = []
        for color_index, (index, frame_target) in enumerate(zip(indices, frame_targets, strict=True)):
            pose = transform_points_camera_to_radar(
                load_pose_pickle(frame_target.pose_path)[frame_target.person_index], *low_extrinsic
            )
            if not pose_in_region(pose):
                continue
            associated = associated_points_low(dataset, frame_target)[:, :3]
            color = PERSON_COLORS[color_index % len(PERSON_COLORS)]
            if len(associated):
                ax.scatter(*associated.T, s=8, c=color, alpha=0.65, depthshade=False)
            draw_skeleton(ax, pose, skeleton)
            prediction = int(predictions[index])
            raw_prediction = int(raw_predictions[index])
            confidence = probabilities[index, raw_prediction]
            head = pose[:5]
            head = head[np.isfinite(head).all(axis=1)]
            if len(head):
                label_position = (*np.median(head[:, :2], axis=0), head[:, 2].max() + LABEL_HEIGHT_OFFSET)
                label = f"P{frame_target.person_index}: {LABEL_NAMES[prediction]}"
                if SHOW_INFERENCE_DETAILS:
                    label += f"\nraw {LABEL_NAMES[raw_prediction]} {confidence:.0%}"
                ax.text(
                    *label_position,
                    label,
                    color=color,
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                )
            if SHOW_INFERENCE_DETAILS:
                probability_lines.append(
                    f"P{frame_target.person_index}: "
                    + " ".join(f"{label}={probability:.0%}" for label, probability in zip(LABEL_NAMES, probabilities[index], strict=True))
                    + f" | tlv1={dynamic_ratios[index]:.0%} points={point_counts[index]:.0f} "
                    f"height={robust_heights[index]:.2f} {'hold' if holding[index] else 'infer'}"
                )
        probability_text = "\n".join(probability_lines)
        if probability_text:
            ax.text2D(0.02, 0.98, probability_text, transform=ax.transAxes, va="top")

        handles = [
            Line2D([], [], marker=".", linestyle="", color="#bdbdbd", label="all points"),
            Line2D([], [], marker=".", linestyle="", color="#2ca02c", label="associated person points"),
            Line2D([], [], color="#222222", label="camera pose"),
        ]
        if not SHOW_UNASSOCIATED_POINTS:
            handles.pop(0)
        ax.legend(handles=handles, loc="upper right")
        figure.canvas.draw_idle()

    def on_key_press(event) -> None:
        index = int(slider.val)
        if event.key == "left":
            slider.set_val(max(0, index - 1))
        elif event.key == "right":
            slider.set_val(min(len(frame_indices) - 1, index + 1))
        elif event.key in {" ", "space"}:
            playing[0] = not playing[0]
            draw_frame(slider.val)

    def advance() -> None:
        if playing[0]:
            slider.set_val((int(slider.val) + 1) % len(frame_indices))

    slider.on_changed(draw_frame)
    figure.canvas.mpl_connect("key_press_event", on_key_press)
    if EXPORT_GIF:
        export_gif(
            figure, draw_frame, len(frame_indices), PROJECT_ROOT / GIF_PATH, FPS, dpi=GIF_DPI
        )
        print(f"GIF saved to {PROJECT_ROOT / GIF_PATH}")
    timer = figure.canvas.new_timer(interval=round(1_000 / FPS))
    timer.add_callback(advance)
    timer.start()
    draw_frame(0)
    print(f"checkpoint={checkpoint_path}, epoch={checkpoint['epoch']}, val_macro_f1={checkpoint['val_macro_f1']:.4f}")
    print("WebAgg starting; open http://<server-ip>:<port shown below>. Space pauses playback.")
    plt.show()


if __name__ == "__main__":
    main()
