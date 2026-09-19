"""保存选定组的点云、姿态/行为真值和模型原始姿态推理结果。"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import Inference_selected_group as base


MODEL_POSE_PATH = base.PROJECT_ROOT / "experiments/P4Transformer/20260916_140833"


GROUPS = (
    "20260615/group_021",
    # "20260912/group_028", "20260912/group_029", "20260912/group_030",
    # "20260912/group_031", "20260912/group_032", "20260912/group_046",
    # "20260912/group_049", "20260912/group_050", "20260912/group_051",
    # "20260912/group_096", "20260912/group_097", "20260912/group_099",
    # "20260912/group_100", "20260912/group_101", "20260912/group_102",
    # "20260912/group_105", "20260912/group_106", "20260912/group_108",
    # "20260912/group_109", "20260912/group_110",
    # "20260913/group_011", "20260913/group_012", "20260913/group_013",
    # "20260913/group_014", "20260913/group_015", "20260913/group_016",
)


def configure_pose_model(path):
    """使用所选实验的输入窗口，输出窗口最后一帧。"""
    from run.utils.load_config import load_config

    base.MODEL_POSE_PATH = Path(path).resolve()
    data = load_config(base.MODEL_POSE_PATH / "config/config.yaml")["data"]
    base.T = int(data["T"])
    base.TRAIN_MAX_POINTS = int(data["max_points"])
    base.POSE_OUTPUT_POSITION = base.T - 1
    base.POSE_LOOKAHEAD = 0
    return data


def load_action_gt(gt_pose_path):
    """读取与姿态 GT 同时间戳、同人员顺序的行为 one-hot 真值。"""
    if gt_pose_path is None:
        return np.empty((0, 4), dtype=np.float32)
    gt_pose_path = Path(gt_pose_path)
    action_dir = gt_pose_path.parent.parent / "action label"
    action_path = next(
        (action_dir / f"{gt_pose_path.stem}{suffix}"
         for suffix in (".pkl", ".npz")
         if (action_dir / f"{gt_pose_path.stem}{suffix}").is_file()),
        None,
    )
    if action_path is None:
        raise FileNotFoundError(f"姿态 GT 对应的行为真值不存在：{action_dir / gt_pose_path.name}")
    if action_path.suffix == ".npz":
        with np.load(action_path) as source:
            data = source["labels"]
    else:
        with action_path.open("rb") as source:
            loaded = pickle.load(source)
        data = loaded["labels"] if isinstance(loaded, dict) else loaded
    labels = np.asarray(data, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] != 4 or not np.isfinite(labels).all():
        raise ValueError(f"行为真值应为有限的 [N,4] 数组：{action_path}, shape={labels.shape}")
    return labels


@torch.inference_mode()
def infer_accumulated_results(app):
    """按 GT 人员索引复用 RHL 的历史静点补充和姿态窗口。"""
    tracks, timestamps, results = {}, {}, []
    for index, record in enumerate(app.records):
        points, skip_reason = base.load_point_cloud(record["path"])
        seconds, nanoseconds = Path(record["path"]).stem.split("_")
        timestamp = int(seconds) + int(nanoseconds) / 1e9
        sample_indices = (np.random.default_rng(index).choice(
            len(points), base.TRAIN_MAX_POINTS, replace=False
        ) if len(points) > base.TRAIN_MAX_POINTS else np.arange(len(points)))
        sampled = points[sample_indices]
        active_tracks, people, clouds = [], [], []
        for person_index, person_gt in enumerate(record["gt"]):
            track = tracks.setdefault(person_index, base.PoseTrack(person_index))
            bbox = np.concatenate((person_gt.min(axis=0) - base.GT_BBOX_MARGIN,
                                   person_gt.max(axis=0) + base.GT_BBOX_MARGIN))
            inside = ((points[:, :3] >= bbox[:3]) & (points[:, :3] <= bbox[3:])).all(1)
            if (skip_reason is not None or not inside.any() or
                    timestamp - timestamps.get(person_index, timestamp) > 0.5):
                track.history.clear()
                track.static_history.clear()
            if skip_reason is not None or not inside.any():
                timestamps.pop(person_index, None)
                continue
            timestamps[person_index] = timestamp
            track.history.append(None)
            cloud, _, _ = base.accumulate_track_static_points(
                track, sampled, inside[sample_indices], (bbox[:3] + bbox[3:]) / 2
            )
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            crop, _ = base.crop_and_pad(
                cloud, point_mask=np.ones(len(cloud), dtype=bool),
                max_points=base.TRAIN_MAX_POINTS,
            )
            if crop is not None:
                track.history[-1] = crop
                active_tracks.append(track)
                people.append(person_index)
                clouds.append(cloud)
        for person_index in set(tracks) - set(range(len(record["gt"]))):
            tracks[person_index].history.clear()
            tracks[person_index].static_history.clear()
            timestamps.pop(person_index, None)
        result = {
            "accumulated_point_cloud": {
                "points_by_person": clouds,
                "person_indices": np.asarray(people, dtype=int),
            },
            "accumulated_inference": {
                "poses": np.empty((0, 17, 3), dtype=np.float32),
                "person_indices": np.empty(0, dtype=int),
            },
        }
        results.append(result)
        if index >= base.T - 1 and active_tracks:
            batch = {key: value.to(app.device)
                     for key, value in base.build_pose_input(active_tracks).items()}
            position = base.POSE_OUTPUT_POSITION
            valid = batch["mask"][:, position].any(dim=1).cpu().numpy()
            poses = app.pose_model(batch)["pose"][:, position, 0].cpu().numpy()[valid]
            results[index - base.POSE_LOOKAHEAD]["accumulated_inference"] = {
                "poses": poses, "person_indices": np.asarray(people, dtype=int)[valid],
            }
    return results


def build_results(app):
    """保存原始及静点积累后的点云与推理结果；统计由 analysis.py 计算。"""
    app.infer_gt_raw_as_training(collect_quality=False)
    accumulated_results = infer_accumulated_results(app)
    results = []
    for record, accumulated in zip(app.records, accumulated_results, strict=True):
        points, _ = base.load_point_cloud(record["path"])
        action_gt = load_action_gt(record.get("gt_pose_path"))
        if len(action_gt) != len(record["gt"]):
            raise ValueError(
                f"姿态与行为真值人数不一致：{record['path']}，"
                f"pose={len(record['gt'])}, action={len(action_gt)}"
            )
        results.append({
            "point_cloud": points,
            "pose_gt": record["gt"],
            "action_gt": action_gt,
            "raw_inference": {
                "poses": record["gt_crop_poses_raw"],
                "person_indices": record["gt_crop_pose_indices"],
            },
            **accumulated,
        })
    return results


def save_results(results, path):
    """沿用原脚本的最高协议及临时文件原子替换保存方式。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        pickle.dump(results, destination, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f"results 已保存：{path} ({path.stat().st_size / 1024**2:.1f} MiB)")
    return path


def self_test():
    from unittest.mock import patch

    training_data = configure_pose_model(base.MODEL_POSE_PATH)
    # 第一人有点云，第二人有 GT 但整窗没有框内点，不应送入模型。
    gt = np.stack((np.zeros((17, 3)), np.full((17, 3), 10))).astype(np.float32)
    frames = []
    for index in range(base.T + 2):
        points = np.zeros((30, 6), dtype=np.float32)
        points[:, 3] = index
        frames.append(points)
    inputs = []

    def predict(model_input):
        points, mask = model_input["input"], model_input["mask"]
        assert points.shape == (1, base.T, base.TRAIN_MAX_POINTS, 6)
        assert mask.all()
        # 原始分支遵循所选实验的训练积累配置。
        end = len(inputs) + base.T - 1
        acc_frame = training_data.get("acc_frame", 0)
        assert set(points[0, -1, :, 3].tolist()) == set(range(end - acc_frame, end + 1))
        inputs.append(points.clone())
        return {"pose": torch.ones((1, base.T, 1, 17, 3))}

    app = object.__new__(base.SelectedGroupVisualizer)
    app.records = [{"path": str(i), "gt": gt, "has_gt": True} for i in range(len(frames))]
    app.raw_training_ready, app.device, app.pose_model = False, "cpu", predict
    with patch.object(
        base, "load_point_cloud", side_effect=lambda path: (frames[int(path)], None)
    ):
        app.infer_gt_raw_as_training(collect_quality=False)
        assert len(inputs) == 3
        assert all(len(record["gt_crop_poses_raw"]) == 0 for record in app.records[:base.T - 1])
        for record in app.records[base.T - 1:]:
            np.testing.assert_array_equal(record["gt_crop_pose_indices"], [0])
            assert record["gt_crop_poses_raw"].shape == (1, 17, 3)
        app.infer_gt_raw_as_training(collect_quality=False)
        assert len(inputs) == 3  # 已完成的结果不重复推理。
    print("self-test passed")

    # 静点补充、中心位移对齐，以及空关联/时间中断时清空历史。
    accumulated_frames = []
    records = []
    for index in range(base.T + 2):
        cloud = np.zeros((3, 6), dtype=np.float32)
        cloud[:, 0], cloud[:, 3], cloud[:, -1] = index * .01, index, 2
        accumulated_frames.append(cloud)
        records.append({"path": f"0_{index * 100000000:09d}.npz",
                        "gt": gt + np.array([index * .01, 0, 0])})
    accumulated_frames[4] = accumulated_frames[4][:0]
    records[-1]["path"] = "10_000000000.npz"
    calls = []

    def accumulated_predict(batch):
        assert batch["input"].shape == (1, base.T, base.TRAIN_MAX_POINTS, 6)
        calls.append(batch)
        return {"pose": torch.full((1, base.T, 1, 17, 3), 2.0)}

    app.records, app.pose_model = records, accumulated_predict
    clouds_by_path = {record["path"]: cloud
                      for record, cloud in zip(records, accumulated_frames)}
    with patch.object(base, "load_point_cloud", side_effect=lambda path: (clouds_by_path[path], None)):
        accumulated = infer_accumulated_results(app)
    assert len(calls) == 3
    for index, expected in ((3, 12), (4, 0), (5, 3), (8, 12), (len(records) - 1, 3)):
        cloud_result = accumulated[index]["accumulated_point_cloud"]
        if expected:
            np.testing.assert_array_equal(cloud_result["person_indices"], [0])
            assert len(cloud_result["points_by_person"][0]) == expected
            np.testing.assert_allclose(cloud_result["points_by_person"][0][:, 0], index * .01)
        else:
            assert not cloud_result["points_by_person"]
    assert accumulated[base.T - 1]["accumulated_inference"]["poses"].shape == (1, 17, 3)
    assert not len(accumulated[0]["accumulated_inference"]["poses"])
    print("accumulation self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=GROUPS,
                        help="待处理的 date/group；默认处理脚本内置的全部组")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-pose-path", type=Path, default=MODEL_POSE_PATH,
                        help="实验目录；加载其中的 best.pth 和训练配置")
    parser.add_argument("--output-dir", type=Path, default=base.PROJECT_ROOT / "temp")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    configure_pose_model(args.model_pose_path)
    if args.self_test:
        self_test()
        return

    print(f"姿态模型实验：{base.MODEL_POSE_PATH}")
    for value in args.groups:
        try:
            date, group = value.split("/", 1)
        except ValueError as error:
            raise ValueError(f"组名必须为 date/group 格式：{value}") from error
        base.DATE, base.GROUP = date, group
        output = args.output_dir / f"Inference_selected_group_{date}_{group}_results.pkl"
        print(f"开始处理 {date}/{group}")
        app = base.SelectedGroupVisualizer(args.device, show_gt=True, visualize=False)
        for path in tqdm(app.paths, desc=f"{date}/{group} GT"):
            record = {"path": str(path)}
            app.load_gt(record)
            app.records.append(record)
        save_results(build_results(app), output)


if __name__ == "__main__":
    main()
