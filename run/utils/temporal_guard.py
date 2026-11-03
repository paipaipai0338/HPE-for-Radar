"""Shared temporal guard and frame scoring for deployment and validation."""
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.skeleton import get_pose_layout


def match_poses(poses, gt, pose_format="coco"):
    """过滤非有限人体，按髋中心距离匈牙利匹配，保留匹配前人数。"""
    pred = poses[np.isfinite(poses).all(axis=(1, 2))]
    target = gt[np.isfinite(gt).all(axis=(1, 2))]
    counts = len(pred), len(target)
    roots = list(get_pose_layout(pose_format)["root_indices"])
    pred_root = pred[:, roots].mean(axis=1)
    gt_root = target[:, roots].mean(axis=1)
    rows, cols = linear_sum_assignment(
        np.linalg.norm(pred_root[:, None] - gt_root[None], axis=-1)
    )
    return pred[rows], target[cols], counts


def frame_mpjpe(poses, gt, centered=False, pose_format="coco"):
    """所选骨架的 MPJPE（mm）；centered 时仅去掉骨盆中心平移。"""
    pred, target, counts = match_poses(poses, gt, pose_format)
    if not len(pred):
        return None, 0, *counts
    if centered:
        roots = list(get_pose_layout(pose_format)["root_indices"])
        pred = pred - pred[:, roots].mean(axis=1, keepdims=True)
        target = target - target[:, roots].mean(axis=1, keepdims=True)
    error_mm = np.linalg.norm(pred - target, axis=-1).mean() * 1000
    return float(error_mm), len(pred), *counts



# 固定阈值仅使用点数和预测变化，GT 误差不参与判决。
TEMPORAL_GUARD_PARAMS = {
    "min_dynamic": 20, "min_total": 30,
    "root_speed_m_s": 2.0, "shape_speed_m_s": 1.25, "axis_jump_deg": 60.0,
    "stable_root_speed_m_s": 1.0, "stable_shape_speed_m_s": 2 / 3,
    "stable_axis_deg": 30.0, "confirm_frames": 3, "hold_frames": 2,
}


def guard_metric_protocol():
    """随指标记录实际 Guard 参数；算法口径与各报表的计分范围分开注明。"""
    p = TEMPORAL_GUARD_PARAMS
    return {
        "enabled": True, "name": "temporal_guard", "params": p.copy(),
        "stages": {"raw": "original predictions", "guarded": "temporal guard outputs"},
        "point_counts": "unaccumulated current-frame GT crop; dynamic = TLV tag 1",
        "person_ids": "GT person indices, not detector track IDs",
        "offline_backfill": True,
        "description": (
            f"动态点<{p['min_dynamic']}或总点数<{p['min_total']}、缺Raw或Raw非有限时，"
            "持续保持上一可信骨架，无可信骨架则不输出；"
            f"正常变化阈值：根节点速度≤{p['root_speed_m_s']:g}m/s、"
            f"去平移形变速度≤{p['shape_speed_m_s']:g}m/s、纵轴夹角≤{p['axis_jump_deg']:g}°；"
            f"突变候选须连续{p['confirm_frames']}帧满足相邻根速度≤{p['stable_root_speed_m_s']:g}m/s、"
            f"形变速度≤{p['stable_shape_speed_m_s']:g}m/s、夹角≤{p['stable_axis_deg']:g}°，确认后离线回填；"
            f"未确认突变最多暂留旧骨架{p['hold_frames']}帧，超出后撤销整个待确认区间；"
            "不使用GT误差作判决，不进行插值或平滑。"
        ),
    }


def pose_change(a, b, pose_format="coco"):
    roots = list(get_pose_layout(pose_format)["root_indices"])
    head, ankles = (0, [15, 16]) if pose_format == "coco" else (6, [17, 20])
    root_a = a[roots].mean(0)
    root_b = b[roots].mean(0)
    root = float(np.linalg.norm(root_a - root_b))
    shape = float(np.linalg.norm((a - root_a) - (b - root_b), axis=1).mean())
    axis_a = a[head] - a[ankles].mean(0)
    axis_b = b[head] - b[ankles].mean(0)
    norm = np.linalg.norm(axis_a) * np.linalg.norm(axis_b)
    angle = (float(np.degrees(np.arccos(np.clip(np.dot(axis_a, axis_b) / norm, -1, 1))))
             if norm > 1e-10 else 180.0)
    return root, shape, angle


def pose_change_within(a, b, dt, stable=False, pose_format="coco"):
    root, shape, angle = pose_change(a, b, pose_format)
    p = TEMPORAL_GUARD_PARAMS
    return (root <= p["stable_root_speed_m_s" if stable else "root_speed_m_s"] * dt
            and shape <= p["stable_shape_speed_m_s" if stable else "shape_speed_m_s"] * dt
            and angle <= p["stable_axis_deg" if stable else "axis_jump_deg"])


def guard_pose_records(records, prefix="", pose_format="coco", direction_factory=None):
    """低点数持续保持；点数充足的突变确认后回填三帧。

    shortcut: 人员沿用 GT 索引；接入检测轨迹时需改用稳定 track ID。
    """
    joints = len(get_pose_layout(pose_format)["names"])
    outputs = [{} for _ in records]
    states = {}
    params = TEMPORAL_GUARD_PARAMS
    if direction_factory is not None and pose_format != "coco":
        raise ValueError("Direction evidence requires COCO-17 poses")
    for i, record in enumerate(records):
        if direction_factory is not None:
            record[f"{prefix}direction_state"] = {}
        stamp = Path(record["path"]).stem.split("_")
        timestamp = int(stamp[0]) + int(stamp[1]) * 1e-9
        raw_by_person = dict(zip(record[f"{prefix}pose_indices"], record[f"{prefix}poses_raw"]))
        # 裁剪不足时可能没有 raw，但当前 GT 轨迹仍需要保持上一帧。
        people = record.get(f"{prefix}track_ids", record[f"{prefix}pose_indices"])
        for person in people:
            person = int(person)
            raw = raw_by_person.get(person)
            state = states.get(person)
            if state is None or state["last"] != i - 1:
                state = {"last": i - 1, "time": timestamp - .12, "ref": None,
                         "pending": [], "candidate": [], "protected": set(),
                         "direction": direction_factory() if direction_factory is not None else None}
                states[person] = state
            dt = timestamp - state["time"]
            if dt <= 0:
                raise ValueError("Non-increasing point-cloud timestamp")
            weak = (record["dynamic_point_counts"].get(person, 0) < params["min_dynamic"]
                    or record["total_point_counts"].get(person, 0) < params["min_total"])
            direction = state["direction"]
            permit = direction.check(raw, record["dynamic_point_counts"].get(person, 0),
                                     record["total_point_counts"].get(person, 0), dt) if direction else True
            if not permit:
                state["protected"].add(i)
            if direction:
                record[f"{prefix}direction_state"][person] = direction.snapshot()
            if weak or raw is None or not np.isfinite(raw).all():
                if state["ref"] is not None:
                    outputs[i][person] = (state["ref"].copy(), "held")
                state["pending"] = []
                state["candidate"] = []
                state["protected"].clear()
                state["last"] = i
                state["time"] = timestamp
                continue
            normal = (state["ref"] is None
                      or pose_change_within(raw, state["ref"], dt, pose_format=pose_format))
            accepted = False
            if normal and permit:
                accepted = True
                outputs[i][person] = (raw.copy(), "accepted")
                state["ref"] = raw.copy()
                state["pending"] = []
                state["candidate"] = []
                state["protected"].clear()
            else:
                state["pending"].append(i)
                candidates = state["candidate"]
                if (candidates and candidates[-1][0] == i - 1
                        and pose_change_within(raw, candidates[-1][1], dt, stable=True,
                                               pose_format=pose_format)):
                    candidates.append((i, raw.copy(), not direction or not direction.hard_hold))
                else:
                    candidates[:] = [(i, raw.copy(), not direction or not direction.hard_hold)]
                if len(candidates) >= params["confirm_frames"] and permit:
                    accepted = True
                    for j, pose, backfill in candidates[-params["confirm_frames"]:]:
                        if backfill:
                            outputs[j][person] = (pose.copy(), "confirmed")
                    state["ref"] = raw.copy()
                    state["pending"] = []
                    state["candidate"] = []
                    state["protected"].clear()
                else:
                    if state["ref"] is not None and (not permit or len(state["pending"]) <= params["hold_frames"]):
                        outputs[i][person] = (state["ref"].copy(), "held")
                    if permit and len(state["pending"]) > params["hold_frames"]:
                        # 离线评估撤销整个未确认区间，避免旧骨架被算作新观测。
                        for j in state["pending"]:
                            if j not in state["protected"]:
                                outputs[j].pop(person, None)
            if direction:
                if accepted:
                    direction.commit()
                record[f"{prefix}direction_state"][person] = direction.snapshot()
            state["last"] = i
            state["time"] = timestamp
    for record, output in zip(records, outputs):
        people = list(output)
        record[f"{prefix}poses"] = np.asarray(
            [output[p][0] for p in people], dtype=np.float32).reshape(-1, joints, 3)
        record[f"{prefix}pose_track_ids"] = np.asarray(people, dtype=int)
        record[f"{prefix}orientation_held"] = np.asarray(
            [output[p][1] == "held" for p in people], dtype=bool)
        record[f"{prefix}pose_status"] = {
            int(p): output[int(p)][1] if int(p) in output else "excluded"
            for p in record.get(f"{prefix}track_ids", record[f"{prefix}pose_indices"])
        }
