from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from WSC.common.labels import LABEL_NAMES
from WSC.common.pose_io import load_extrinsic_npz, transform_points_between_radars
from WSC.pointcloud.data.inference import prepare_behavior_window
from WSC.pointcloud.models.model import BehaviorModel, BehaviorModelConfig
from WSC.pointcloud.utils.pointcloud_io import (
    POINT_STATE_COLUMN,
    expand_box_seed_indices,
    resolve_person_point_overlap,
)
from WSC.pointcloud.utils.postprocess import (
    CausalBehaviorPostprocessor,
    StaticInferenceGate,
    point_postprocess_features,
)


@dataclass(frozen=True, slots=True)
class PersonBox:
    """One tracked high-radar box in [xmin, ymin, zmin, xmax, ymax, zmax]."""

    track_id: int
    box: np.ndarray


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    track_id: int
    label_index: int | None
    label: str
    confidence: float | None
    point_count: int
    holding: bool
    model_ran: bool
    window_size: int


@dataclass(slots=True)
class _TrackState:
    timestamp: float
    frames: list[np.ndarray] = field(default_factory=list)
    postprocessor: CausalBehaviorPostprocessor = field(
        default_factory=CausalBehaviorPostprocessor
    )
    gate: StaticInferenceGate = field(default_factory=StaticInferenceGate)
    label: int | None = None
    confidence: float | None = None
    model_ran: bool = False

    def reset(self) -> None:
        self.frames.clear()
        self.postprocessor.reset()
        self.gate.reset()
        self.label = None
        self.confidence = None
        self.model_ran = False


class BehaviorRecognizer:
    """Recognize each tracked person's behavior from high-radar point clouds."""

    def __init__(
        self,
        checkpoint: Path | str,
        low_extrinsic: Path | str,
        high_extrinsic: Path | str,
        *,
        device: str | torch.device | None = None,
        max_points: int = 128,
        max_frame_gap_seconds: float = 0.5,
        track_timeout_seconds: float = 0.5,
        box_expansion_xy: float = 0.20,
        box_expansion_z: float = 0.25,
        overlap_margin: float = 0.10,
        preserve_z_height: bool = True,
    ) -> None:
        if min(
            max_points,
            max_frame_gap_seconds,
            track_timeout_seconds,
            box_expansion_xy,
            box_expansion_z,
        ) <= 0 or overlap_margin < 0:
            raise ValueError("limits must be positive and overlap_margin non-negative")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.config = BehaviorModelConfig(**saved["model_config"])
        self.model = BehaviorModel(self.config).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.low_extrinsic = load_extrinsic_npz(low_extrinsic)
        self.high_extrinsic = load_extrinsic_npz(high_extrinsic)
        self.max_points = max_points
        self.max_frame_gap_seconds = max_frame_gap_seconds
        self.track_timeout_seconds = track_timeout_seconds
        self.box_expansion_xy = box_expansion_xy
        self.box_expansion_z = box_expansion_z
        self.overlap_margin = overlap_margin
        self.preserve_z_height = preserve_z_height
        self._tracks: dict[int, _TrackState] = {}
        self._last_timestamp: float | None = None

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._tracks.clear()
            self._last_timestamp = None
        else:
            self._tracks.pop(track_id, None)

    def update(
        self,
        high_points: np.ndarray,
        detections: Sequence[PersonBox],
        timestamp: float,
        associated_low_points: dict[int, np.ndarray] | None = None,
        frame_windows: dict[int, list[np.ndarray]] | None = None,
    ) -> list[BehaviorResult]:
        """Process one radar frame and return results in detection order."""
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = timestamp
        boxes = self._validate_boxes(detections)
        if associated_low_points is None:
            if frame_windows is not None:
                raise ValueError("frame_windows requires associated_low_points")
            points_high = self._validate_points(high_points)
            points_low = points_high.copy()
            points_low[:, :3] = transform_points_between_radars(
                points_high[:, :3], self.high_extrinsic, self.low_extrinsic
            )
            corners = [self._low_box_corners(box.box) for box in boxes]
            seeds = [
                np.flatnonzero(np.all(
                    (points_high[:, :3] >= box.box[:3])
                    & (points_high[:, :3] <= box.box[3:]), axis=1,
                )) for box in boxes
            ]
            candidates = [
                expand_box_seed_indices(points_low, seed, corners_for_box,
                                        self.box_expansion_xy, self.box_expansion_z)
                for seed, corners_for_box in zip(seeds, corners, strict=True)
            ]
            assignments = resolve_person_point_overlap(
                points_low, seeds, candidates,
                np.asarray([corners_for_box.mean(axis=0) for corners_for_box in corners]),
                self.overlap_margin,
            )
            person_clouds = [points_low[indices] for indices in assignments]
        else:
            person_clouds = []
            for box in boxes:
                cloud = np.asarray(associated_low_points[box.track_id], dtype=np.float32)
                if cloud.ndim != 2 or cloud.shape[1] != 6 or not np.isfinite(cloud).all():
                    raise ValueError("associated low-radar points must have shape N x 6 and be finite")
                person_clouds.append(cloud)
                if frame_windows is not None:
                    window = frame_windows[box.track_id]
                    if len(window) > self.config.sequence_length or any(
                        frame.ndim != 2 or frame.shape[1] != 6 or not len(frame)
                        or not np.isfinite(frame).all() for frame in window
                    ):
                        raise ValueError("frame_windows must contain finite non-empty N x 6 frames")

        results: list[BehaviorResult | None] = [None] * len(boxes)
        pending: list[
            tuple[
                int,
                _TrackState,
                np.ndarray,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for index, (box, person_points) in enumerate(zip(boxes, person_clouds, strict=True)):
            state = self._tracks.setdefault(box.track_id, _TrackState(timestamp))
            state.model_ran = False
            if timestamp - state.timestamp > self.max_frame_gap_seconds or not len(person_points):
                state.reset()
            state.timestamp = timestamp
            if not len(person_points):
                results[index] = self._result(box.track_id, state, 0)
                continue
            if frame_windows is None:
                state.frames.append(person_points)
                state.frames = state.frames[-self.config.sequence_length:]
            else:
                state.frames = frame_windows[box.track_id]
            if len(state.frames) < self.config.sequence_length:
                results[index] = self._result(box.track_id, state, len(person_points))
                continue
            if not state.gate.should_infer(
                person_points,
                state.postprocessor.established,
                state.postprocessor.transition_pending,
            ):
                results[index] = self._result(box.track_id, state, len(person_points))
                continue
            point_tensor, point_mask, statistics = prepare_behavior_window(
                state.frames,
                self.config,
                self.max_points,
                preserve_z_height=self.preserve_z_height,
            )
            pending.append(
                (index, state, person_points, point_tensor, point_mask, statistics)
            )

        if pending:
            point_tensors = torch.stack([item[3] for item in pending]).to(self.device)
            point_masks = torch.stack([item[4] for item in pending]).to(self.device)
            statistics = torch.stack([item[5] for item in pending]).to(self.device)
            with torch.inference_mode():
                probabilities = self.model(
                    point_tensors, point_masks, statistics
                ).softmax(dim=1).cpu().numpy()
            for item, probability in zip(pending, probabilities, strict=True):
                index, state, person_points = item[:3]
                state.model_ran = True
                state.label = state.postprocessor.update(
                    probability,
                    *point_postprocess_features(person_points),
                )
                state.confidence = float(probability[state.label])
                results[index] = self._result(
                    boxes[index].track_id, state, len(person_points)
                )

        active_ids = {box.track_id for box in boxes}
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if track_id in active_ids
            or timestamp - state.timestamp <= self.track_timeout_seconds
        }
        return [result for result in results if result is not None]

    @staticmethod
    def _validate_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError("high_points must have shape N x 6")
        finite = np.isfinite(points[:, :4]).all(axis=1)
        finite &= np.isfinite(points[:, POINT_STATE_COLUMN])
        return points[finite]

    @staticmethod
    def _validate_boxes(detections: Sequence[PersonBox]) -> list[PersonBox]:
        boxes = list(detections)
        if len({box.track_id for box in boxes}) != len(boxes):
            raise ValueError("track_id must be unique within one frame")
        normalized = []
        for detection in boxes:
            box = np.asarray(detection.box, dtype=np.float32)
            if (
                box.shape != (6,)
                or not np.isfinite(box).all()
                or np.any(box[:3] > box[3:])
            ):
                raise ValueError(
                    "each box must be finite [xmin, ymin, zmin, xmax, ymax, zmax]"
                )
            normalized.append(PersonBox(detection.track_id, box))
        return normalized

    def _low_box_corners(self, box: np.ndarray) -> np.ndarray:
        corners = np.asarray([
            [box[x], box[1 + y], box[2 + z]]
            for x in (0, 3) for y in (0, 3) for z in (0, 3)
        ])
        return transform_points_between_radars(
            corners, self.high_extrinsic, self.low_extrinsic
        ).astype(np.float32)

    @staticmethod
    def _result(track_id: int, state: _TrackState, point_count: int) -> BehaviorResult:
        return BehaviorResult(
            track_id=track_id,
            label_index=state.label,
            label="waiting" if state.label is None else LABEL_NAMES[state.label],
            confidence=state.confidence,
            point_count=point_count,
            holding=state.gate.holding,
            model_ran=state.model_ran,
            window_size=len(state.frames),
        )
