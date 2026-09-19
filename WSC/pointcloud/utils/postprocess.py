from __future__ import annotations

from collections import deque

import numpy as np

from WSC.pointcloud.utils.pointcloud_io import POINT_STATE_COLUMN


OTHER_LABEL_INDEX = 3


def point_postprocess_features(points: np.ndarray) -> tuple[float, float, float]:
    """Return TLV1 ratio, point count, and robust height for postprocessing."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] <= POINT_STATE_COLUMN:
        raise ValueError("points must include the TLV state column")
    if not len(points):
        raise ValueError("points must not be empty")
    states = np.rint(points[:, POINT_STATE_COLUMN]).astype(np.int8)
    dynamic_count = np.count_nonzero(states == 1)
    state_count = np.count_nonzero((states >= 1) & (states <= 7))
    dynamic_ratio = float((dynamic_count + 1) / (state_count + 2)) if state_count else 0.0
    robust_height = float(np.quantile(points[:, 2], 0.90) - np.quantile(points[:, 2], 0.10))
    return dynamic_ratio, float(len(points)), robust_height


def dynamic_point_ratio(points: np.ndarray) -> float:
    """Return the smoothed TLV1 share among associated TLV1--7 points."""
    return point_postprocess_features(points)[0]


class StaticInferenceGate:
    """Decide whether a tracked person's current window needs model inference."""

    def __init__(
        self, *, minimum_points: int = 60, static_ratio: float = 0.15,
        dynamic_ratio: float = 0.25, minimum_dynamic_points: int = 10,
        count_growth: float = 1.8, center_shift: float = 0.3,
        active_frames: int = 16, recheck_frames: int = 8,
    ) -> None:
        if min(minimum_points, minimum_dynamic_points, active_frames, recheck_frames) < 1:
            raise ValueError("gate counts must be positive")
        if not 0 <= static_ratio < dynamic_ratio <= 1 or count_growth <= 1 or center_shift <= 0:
            raise ValueError("invalid gate motion thresholds")
        self.minimum_points = minimum_points
        self.static_ratio = static_ratio
        self.dynamic_ratio = dynamic_ratio
        self.minimum_dynamic_points = minimum_dynamic_points
        self.count_growth = count_growth
        self.center_shift = center_shift
        self.active_frames = active_frames
        self.recheck_frames = recheck_frames
        self.reset()

    def reset(self) -> None:
        self.remaining_active = 0
        self.held_frames = 0
        self.reference_count: int | None = None
        self.reference_center: np.ndarray | None = None
        self.holding = False

    def should_infer(
        self,
        points: np.ndarray,
        state_established: bool,
        transition_pending: bool = False,
    ) -> bool:
        if not len(points):
            self.reset()
            return False
        states = np.rint(points[:, POINT_STATE_COLUMN]).astype(int)
        dynamic_count = int(np.count_nonzero(states == 1))
        valid_count = int(np.count_nonzero((states >= 1) & (states <= 7)))
        ratio = dynamic_count / max(valid_count, 1)
        count = len(points)
        center = np.median(points[:, :3], axis=0)
        moved = self.reference_center is not None and np.linalg.norm(center - self.reference_center) >= self.center_shift
        grew = self.reference_count is not None and count >= max(
            self.minimum_points, self.reference_count * self.count_growth
        )
        dynamic = ratio >= self.dynamic_ratio and dynamic_count >= self.minimum_dynamic_points
        if dynamic or moved or grew:
            self.remaining_active = self.active_frames
        quiet = count < self.minimum_points or (valid_count > 0 and ratio <= self.static_ratio)
        self.holding = (
            state_established and not transition_pending and quiet
            and self.remaining_active == 0 and self.held_frames < self.recheck_frames
        )
        if self.holding:
            if self.held_frames == 0:
                self.reference_count = count
                self.reference_center = center
            self.held_frames += 1
            return False
        self.held_frames = 0
        self.remaining_active = max(0, self.remaining_active - 1)
        self.reference_count = count
        self.reference_center = center
        return True


class CausalBehaviorPostprocessor:
    """Maintain activity-aware smoothing state for one tracked person."""

    def __init__(
        self,
        *,
        dynamic_ratio_threshold: float = 0.25,
        quality_history_frames: int = 16,
        minimum_count_ratio: float = 0.70,
        minimum_height_ratio: float = 0.80,
        static_confirmation_frames: int = 8,
        dynamic_confirmation_frames: int = 2,
        other_confirmation_frames: int = 16,
        minimum_confidence: float = 0.55,
        minimum_margin: float = 0.08,
        establishment_frames: int = 4,
        motion_grace_frames: int = 4,
        strong_confidence: float = 0.8,
        use_quality_features: bool = True,
    ) -> None:
        self.dynamic_ratio_threshold = dynamic_ratio_threshold
        self.minimum_count_ratio = minimum_count_ratio
        self.minimum_height_ratio = minimum_height_ratio
        self.static_confirmation_frames = static_confirmation_frames
        self.dynamic_confirmation_frames = dynamic_confirmation_frames
        self.other_confirmation_frames = other_confirmation_frames
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self.establishment_frames = establishment_frames
        self.motion_grace_frames = motion_grace_frames
        self.strong_confidence = strong_confidence
        self.use_quality_features = use_quality_features
        self.count_history = deque(maxlen=quality_history_frames)
        self.height_history = deque(maxlen=quality_history_frames)
        self.reset()

    def reset(self) -> None:
        self.state: int | None = None
        self.candidate = -1
        self.candidate_frames = 0
        self.motion_frames_remaining = 0
        self.confirmed_frames = 0
        self.established = False
        self.count_history.clear()
        self.height_history.clear()

    @property
    def transition_pending(self) -> bool:
        return self.candidate >= 0

    def update(
        self,
        probabilities: np.ndarray,
        dynamic_ratio: float,
        point_count: float,
        robust_height: float,
    ) -> int:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if self.state is None:
            self.state = int(probabilities.argmax())
            low_quality = False
        else:
            dynamic = dynamic_ratio >= self.dynamic_ratio_threshold
            self.motion_frames_remaining = (
                self.motion_grace_frames
                if dynamic
                else max(0, self.motion_frames_remaining - 1)
            )
            low_quality = self.use_quality_features and not dynamic and (
                point_count < self.minimum_count_ratio * np.median(self.count_history)
                and robust_height < self.minimum_height_ratio * np.median(self.height_history)
            )
            challenger = int(probabilities.argmax())
            strong = (
                probabilities[challenger] >= self.strong_confidence
                and probabilities[self.state] <= 1.0 - self.strong_confidence
            )
            eligible = (
                not low_quality
                and probabilities[challenger] >= self.minimum_confidence
                and probabilities[challenger] - probabilities[self.state] >= self.minimum_margin
            )
            if challenger == self.state or not eligible:
                self.candidate = -1
                self.candidate_frames = 0
            else:
                self.candidate_frames = self.candidate_frames + 1 if challenger == self.candidate else 1
                self.candidate = challenger
                if self.motion_frames_remaining or not self.established:
                    required = self.dynamic_confirmation_frames
                elif challenger == OTHER_LABEL_INDEX:
                    required = self.other_confirmation_frames
                elif strong:
                    required = self.dynamic_confirmation_frames
                else:
                    required = self.static_confirmation_frames
                if self.candidate_frames >= required:
                    self.state = self.candidate
                    self.candidate = -1
                    self.candidate_frames = 0
                    self.confirmed_frames = 0
                    self.established = False
        agrees = int(probabilities.argmax()) == self.state
        self.confirmed_frames = self.confirmed_frames + 1 if agrees else 0
        if self.confirmed_frames >= self.establishment_frames:
            self.established = True
        if not low_quality:
            self.count_history.append(point_count)
            self.height_history.append(robust_height)
        return self.state


def smooth_behavior_predictions(
    probabilities: np.ndarray,
    dynamic_ratios: np.ndarray,
    *,
    point_counts: np.ndarray | None = None,
    robust_heights: np.ndarray | None = None,
    reset_mask: np.ndarray | None = None,
    dynamic_ratio_threshold: float = 0.25,
    quality_history_frames: int = 16,
    minimum_count_ratio: float = 0.70,
    minimum_height_ratio: float = 0.80,
    static_confirmation_frames: int = 8,
    dynamic_confirmation_frames: int = 2,
    other_confirmation_frames: int = 16,
    minimum_confidence: float = 0.55,
    minimum_margin: float = 0.08,
) -> np.ndarray:
    """Apply causal activity-aware smoothing to frame classification probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    dynamic_ratios = np.asarray(dynamic_ratios, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [frame, class]")
    if dynamic_ratios.shape != (len(probabilities),):
        raise ValueError("dynamic_ratios must have one value per frame")
    if not np.isfinite(probabilities).all() or not np.isfinite(dynamic_ratios).all():
        raise ValueError("probabilities and dynamic_ratios must be finite")
    if min(static_confirmation_frames, dynamic_confirmation_frames, other_confirmation_frames) < 1:
        raise ValueError("confirmation frame counts must be positive")
    if quality_history_frames < 1:
        raise ValueError("quality_history_frames must be positive")
    probability_parameters = (
        dynamic_ratio_threshold,
        minimum_count_ratio,
        minimum_height_ratio,
        minimum_confidence,
    )
    if any(not 0.0 <= value <= 1.0 for value in probability_parameters) or minimum_margin < 0.0:
        raise ValueError("confidence and margin thresholds must be valid probabilities")

    quality_enabled = point_counts is not None or robust_heights is not None
    if quality_enabled and (point_counts is None or robust_heights is None):
        raise ValueError("point_counts and robust_heights must be provided together")
    counts = np.ones(len(probabilities)) if point_counts is None else np.asarray(point_counts, dtype=np.float64)
    heights = np.ones(len(probabilities)) if robust_heights is None else np.asarray(robust_heights, dtype=np.float64)
    if counts.shape != (len(probabilities),) or heights.shape != (len(probabilities),):
        raise ValueError("quality features must have one value per frame")
    if not np.isfinite(counts).all() or not np.isfinite(heights).all() or np.any(counts <= 0) or np.any(heights < 0):
        raise ValueError("quality features must be finite and non-negative")

    resets = (
        np.zeros(len(probabilities), dtype=bool)
        if reset_mask is None
        else np.asarray(reset_mask, dtype=bool).copy()
    )
    if resets.shape != (len(probabilities),):
        raise ValueError("reset_mask must have one value per frame")
    if not len(probabilities):
        return np.empty(0, dtype=np.int64)
    resets[0] = True

    postprocessor = CausalBehaviorPostprocessor(
        dynamic_ratio_threshold=dynamic_ratio_threshold,
        quality_history_frames=quality_history_frames,
        minimum_count_ratio=minimum_count_ratio,
        minimum_height_ratio=minimum_height_ratio,
        static_confirmation_frames=static_confirmation_frames,
        dynamic_confirmation_frames=dynamic_confirmation_frames,
        other_confirmation_frames=other_confirmation_frames,
        minimum_confidence=minimum_confidence,
        minimum_margin=minimum_margin,
        use_quality_features=quality_enabled,
    )
    predictions = np.empty(len(probabilities), dtype=np.int64)
    frames = zip(probabilities, dynamic_ratios, counts, heights, strict=True)
    for index, (frame_probabilities, dynamic_ratio, point_count, robust_height) in enumerate(frames):
        if resets[index]:
            postprocessor.reset()
        predictions[index] = postprocessor.update(
            frame_probabilities, dynamic_ratio, point_count, robust_height
        )
    return predictions
