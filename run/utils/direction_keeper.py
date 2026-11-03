"""Per-track direction evidence; consumes predictions and current-frame counts only."""
import numpy as np

from preprocess.actionprocess import classify_actions, LIE, STAND, SIT_SQUAT
from run.utils.temporal_guard import pose_change_within


def unit(vector, minimum):
    length = np.linalg.norm(vector)
    return vector / length if np.isfinite(length) and length >= minimum else None


def directions(pose):
    head = unit(pose[:5].mean(0) - pose[[15, 16]].mean(0), .4)
    lateral = unit(((pose[6] - pose[5]) + (pose[12] - pose[11])) / 2, .1)
    torso = unit(pose[[5, 6]].mean(0) - pose[[11, 12]].mean(0), .15)
    heading = unit(np.cross(lateral, torso), .2) if lateral is not None and torso is not None else None
    return heading, head


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))) if a is not None and b is not None else np.nan


class DirectionKeeper:
    def __init__(self, rotation, translation, lying_lock=True, direction_check=True):
        self.rotation, self.translation = rotation, translation
        self.use_lying_lock = lying_lock
        self.use_direction_check = direction_check
        self.heading_ref = None
        self.head_foot_ref = None
        self.lying_locked = False
        self.previous = None
        self.motion_run = self.exit_run = 0
        self.current = (None, None)
        self.label = None
        self.reason = 'uninitialized'
        self.hard_hold = False

    def check(self, raw, dynamic, total, dt):
        self.hard_hold = False
        if raw is None or not np.isfinite(raw).all() or dynamic < 20 or total < 30:
            self.previous = None
            self.motion_run = self.exit_run = 0
            self.reason = 'point_or_raw_gate'
            return False
        self.current = directions(raw)
        previous_dirs = directions(self.previous) if self.previous is not None else (None, None)
        stable = (self.previous is not None and pose_change_within(raw, self.previous, dt, stable=True)
                  and all(np.isfinite(angle(a, b)) and angle(a, b) <= 30
                          for a, b in zip(self.current, previous_dirs)))
        strong = total >= 60 and dynamic >= 30 and dynamic / total >= .8
        self.motion_run = (self.motion_run + 1 if stable else 1) if strong else 0
        self.previous = raw.copy()
        self.label = int(classify_actions(raw @ self.rotation.T + self.translation).labels.argmax())
        upright = self.label in (STAND, SIT_SQUAT) and self.current[1] is not None
        # Confirm posture independently of translation: getting up and walking must release the lock.
        exit_stable = angle(self.current[1], previous_dirs[1]) <= 30
        self.exit_run = (self.exit_run + 1 if exit_stable else 1) if upright else 0
        if self.lying_locked:
            if self.label != LIE and self.exit_run < 3:
                self.reason, self.hard_hold = 'lying_exit_pending', True
                return False
            if self.label == LIE and (self.current[1] is None or angle(self.current[1], self.head_foot_ref) > 90):
                self.reason, self.hard_hold = 'lying_head_reversal', True
                return False
        large = any(angle(a, b) > max(100., 360. * dt)
                    for a, b in zip(self.current, (self.heading_ref, self.head_foot_ref)))
        if self.use_direction_check and large:
            self.reason = 'large_motion_confirmed' if self.motion_run >= 3 else 'large_motion_insufficient'
            return self.motion_run >= 3
        # Direction-valid small turns follow the person even if Guard rejects translation/shape.
        if self.heading_ref is not None and self.current[0] is not None:
            self.heading_ref = self.current[0].copy()
        if not self.lying_locked and self.head_foot_ref is not None and self.current[1] is not None:
            self.head_foot_ref = self.current[1].copy()
        self.reason = 'small_turn_allowed'
        return True

    def commit(self):
        if self.lying_locked and self.exit_run >= 3:
            self.lying_locked = False
        if self.current[0] is not None:
            self.heading_ref = self.current[0].copy()
        if not self.lying_locked and self.current[1] is not None:
            self.head_foot_ref = self.current[1].copy()
            if self.use_lying_lock and self.label == LIE:
                self.lying_locked = True

    def snapshot(self):
        return dict(heading_ref=None if self.heading_ref is None else self.heading_ref.copy(),
                    head_foot_ref=None if self.head_foot_ref is None else self.head_foot_ref.copy(),
                    lying_locked=self.lying_locked, reason=self.reason)
