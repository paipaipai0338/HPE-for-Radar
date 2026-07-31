"""读取一个未参与训练/验证的组，模拟部署时的数据输入。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from data2datasets.dataset_for_all_task import HPE_Dataset, collate_fn
from metrics.detection import (
    get_acc,
    get_bbox_iou,
    get_bbox_l1,
    get_objectness,
    pairwise_axis_aligned_iou_3d,
    get_precision,
    get_recall,
    paired_axis_aligned_iou_3d,
)
from metrics.pose import get_bone_length, get_mpjpe, get_pampjpe
from preprocess.actionprocess import LABEL_NAMES, classify_actions
from run.utils.build_model import build_model
from run.utils.checkpoint import load_model_checkpoint
from run.utils.plot_fig import plt_fig


ROOT_PATH = Path("/mnt/huawei")
DATE = "20260721"
GROUP = "group_012"
STAGE = "analysis"  # inference / analysis
T = 8
BATCH_SIZE = 8
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
POINT_CLOUD_RANGE = [0.0, -3.0, -2.0, 6.0, 3.0, 2.0]
RESULT_PATH = Path("/home/pai/Huawei/deploy/result_selected_group.pth")
VIDEO_PATH = Path("/home/pai/Huawei/deploy/selected_group.mp4")
VIDEO_FPS = 20

model_detect_path = Path(
    "/home/pai/Huawei/experiments/VoxelNeXt/20260724_231418"
)
model_pose_path = Path(
    "/home/pai/Huawei/experiments/P4Transformer/20260723_174845"
)

def load_models(device):
    if device.type != "cuda":
        raise RuntimeError("VoxelNeXt 依赖 CUDA/spconv，请在可用的 GPU 环境运行")

    detect_model = build_model("VoxelNeXt").to(device)
    pose_model = build_model("P4Transformer").to(device)

    load_model_checkpoint(
        model_detect_path / "checkpoint" / "best.pth",
        detect_model,
        device,
    )
    load_model_checkpoint(
        model_pose_path / "checkpoint" / "best.pth",
        pose_model,
        device,
    )

    detect_model.eval()
    pose_model.eval()
    return detect_model, pose_model


class SelectedGroupDataset(HPE_Dataset):
    """只读取一个指定组，不经过训练集/验证集划分。"""

    def __init__(self, root_path: Path, date: str, group: str, T: int):
        # 不调用 HPE_Dataset.__init__，避免读取和划分整个数据集。
        self.root_path = root_path
        self.date = date
        self.T = T
        self.base_source = "radar_high_pc"
        self.sensor_config = {
            "radar_high_bin": False,
            "radar_high_pc": True,
            "gt": True,
        }
        self.suffix_map = {
            "radar_high_bin": ".bin",
            "radar_high_pc": ".npy",
            "gt": ".pkl",
        }
        self.cached_sensor_names = {"radar_high_pc", "gt"}
        self.calib_cache = {}
        self.pointcloud_cache = {}
        self.gt_cache = {}
        self.action_cache = {}

        group_path = root_path / date / "data_collection" / group
        sensor_paths = {
            "radar_high_pc": group_path / "dpct高位机" / "PC",
            "gt": group_path / "camera results" / "smoothed 3D",
        }
        aligned = self._align_multi_sensor_files(
            sources=sensor_paths,
            base_source=self.base_source,
        )

        frame_count = len(aligned[self.base_source])
        if frame_count < T:
            raise ValueError(f"对齐后只有 {frame_count} 帧，小于 T={T}")

        # 滑窗后的每个 item 是长度为 T 的序列。
        self.data_path_list = {
            sensor: [
                paths[start : start + T]
                for start in range(frame_count - T + 1)
            ]
            for sensor, paths in aligned.items()
        }

    def __getitem__(self, idx):
        radar = self._get_sensor_data_from_path(
            "radar_high_pc",
            self.data_path_list["radar_high_pc"][idx],
        )
        gt = self._get_sensor_data_from_path(
            "gt",
            self.data_path_list["gt"][idx],
        )
        calib = self._load_calib_T(self.date)
        gt_for_high = self._transform_gt_sequence(
            gt, calib["gt_to_high"]["R"], calib["gt_to_high"]["t"]
        )
        gt_for_low = self._transform_gt_sequence(
            gt, calib["gt_to_low"]["R"], calib["gt_to_low"]["t"]
        )

        # 此组没有 action label；占位值不参与检测和姿态评估。
        action = [
            np.zeros((len(frame_gt), 4), dtype=np.float32)
            for frame_gt in gt_for_high
        ]
        return {
            "radar_high_pc": radar,
            "gt": gt,
            "action": action,
            "gt_for_high": gt_for_high,
            "gt_for_low": gt_for_low,
            "high_to_low_R": [calib["high_to_low"]["R"].copy() for _ in range(self.T)],
            "high_to_low_t": [calib["high_to_low"]["t"].copy() for _ in range(self.T)],
        }


def build_dataloader():
    dataset = SelectedGroupDataset(ROOT_PATH, DATE, GROUP, T)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_points=300,
            max_people=6,
        ),
    )


def build_pose_input(points, mask, bbox, detection_mask):
    """为每个过阈值的检测框构造一条姿态模型输入序列。"""
    B, T, N, C = points.shape
    K = bbox.shape[2]
    center = (bbox[..., :3] + bbox[..., 3:]) / 2
    points = points[:, :, None].expand(-1, -1, K, -1, -1)
    mask = mask[:, :, None].expand(-1, -1, K, -1)
    xyz = points[..., :3]
    inside = (
        (xyz >= bbox[..., None, :3])
        & (xyz <= bbox[..., None, 3:])
    ).all(dim=-1)
    pose_mask = mask & inside & detection_mask[..., None]

    pose_points = points.clone()
    pose_points[..., :3] -= center[..., None, :]
    pose_points *= pose_mask[..., None]
    pose_points = pose_points.permute(0, 2, 1, 3, 4).reshape(B * K, T, N, C)
    pose_mask = pose_mask.permute(0, 2, 1, 3).reshape(B * K, T, N)
    return {"input": pose_points, "mask": pose_mask}, center


def transform_xyz(xyz, R, t):
    return xyz @ R.transpose(-1, -2) + t


def transform_bbox(bbox, R, t):
    """变换 bbox 的 8 个角点，再生成目标坐标系下的轴对齐框。"""
    bbox_min, bbox_max = bbox[..., :3], bbox[..., 3:]
    corners = torch.stack(
        [
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_min[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_min[..., 0], bbox_max[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_min[..., 1], bbox_max[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_min[..., 2]), -1),
            torch.stack((bbox_max[..., 0], bbox_max[..., 1], bbox_max[..., 2]), -1),
        ],
        dim=-2,
    )
    corners = transform_xyz(corners, R, t)
    return torch.cat((corners.amin(dim=-2), corners.amax(dim=-2)), dim=-1)


def get_partial_hungarian_matches(
    pred_bbox,
    gt_bbox,
    pred_mask,
    gt_mask,
    bbox_l1_weight=5.0,
    bbox_iou_weight=2.0,
):
    """只匹配有效 query；允许 query 数少于 GT 人数。"""
    pc_min = pred_bbox.new_tensor(POINT_CLOUD_RANGE[:3])
    extent = (
        pred_bbox.new_tensor(POINT_CLOUD_RANGE[3:]) - pc_min
    ).clamp_min(1e-6)
    matches = []

    for batch_idx in range(pred_bbox.shape[0]):
        for time_idx in range(pred_bbox.shape[1]):
            pred_idx = pred_mask[batch_idx, time_idx].nonzero().flatten()
            gt_idx = gt_mask[batch_idx, time_idx].nonzero().flatten()
            if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                empty = torch.empty(0, dtype=torch.long)
                matches.append((empty, empty))
                continue

            pred = pred_bbox[batch_idx, time_idx, pred_idx]
            gt = gt_bbox[batch_idx, time_idx, gt_idx]
            pred_norm = torch.cat(
                ((pred[:, :3] - pc_min) / extent, (pred[:, 3:] - pc_min) / extent),
                dim=-1,
            )
            gt_norm = torch.cat(
                ((gt[:, :3] - pc_min) / extent, (gt[:, 3:] - pc_min) / extent),
                dim=-1,
            )
            l1_cost = (
                pred_norm[:, None] - gt_norm[None]
            ).abs().sum(dim=-1)
            iou_cost = 1 - pairwise_axis_aligned_iou_3d(pred, gt)
            cost = bbox_l1_weight * l1_cost + bbox_iou_weight * iou_cost
            pred_local, gt_local = linear_sum_assignment(cost.cpu().numpy())
            matches.append(
                (
                    pred_idx[torch.as_tensor(pred_local)],
                    gt_idx[torch.as_tensor(gt_local)],
                )
            )

    return matches


def analyze_results():
    loaded = torch.load(RESULT_PATH, map_location="cpu")

    # 滑窗步长为 1，同一物理帧会重复出现。按时间顺序只保留首次出现。
    unique_indices = []
    unique_paths = []
    seen = set()
    for window_idx, window_paths in enumerate(loaded["radar_paths"]):
        for time_idx, path in enumerate(window_paths):
            path = str(path)
            if path not in seen:
                seen.add(path)
                unique_indices.append(window_idx * T + time_idx)
                unique_paths.append(path)

    indices = torch.tensor(unique_indices, dtype=torch.long)

    def unique_frames(key):
        value = loaded[key]
        return value.flatten(0, 1).index_select(0, indices).unsqueeze(0)

    bbox_pre = unique_frames("bbox_pre")
    logits = unique_frames("objectness_logits")
    detection_mask = unique_frames("detection_mask").bool()
    pose_pre = unique_frames("pose_pre")
    pose_gt = unique_frames("pose_gt")
    bbox_gt = unique_frames("bbox_gt")
    gt_mask = unique_frames("gt_valid").bool()
    pc = unique_frames("pc")
    pc_valid = unique_frames("pc_valid")
    high_to_low_R = unique_frames("high_to_low_R")
    high_to_low_t = unique_frames("high_to_low_t")
    pose_gt_gravity = torch.empty_like(pose_gt)
    pose_pre_gravity = torch.empty_like(pose_pre)
    for frame_idx in range(pose_gt.shape[1]):
        R = high_to_low_R[0, frame_idx]
        t = high_to_low_t[0, frame_idx]
        pose_gt_gravity[:, frame_idx] = transform_xyz(
            pose_gt[:, frame_idx], R, t
        )
        pose_pre_gravity[:, frame_idx] = transform_xyz(
            pose_pre[:, frame_idx], R, t
        )

    pose_gt_for_action = pose_gt_gravity.numpy().copy()
    pose_pre_for_action = pose_pre_gravity.numpy().copy()
    pose_gt_for_action[~gt_mask.numpy()] = np.nan
    pose_pre_for_action[~detection_mask.numpy()] = np.nan
    action_gt_result = classify_actions(pose_gt_for_action)
    action_pre_result = classify_actions(pose_pre_for_action)
    action_gt = torch.from_numpy(action_gt_result.labels)
    action_pre = torch.from_numpy(action_pre_result.labels)

    matches = get_partial_hungarian_matches(
        bbox_pre,
        bbox_gt,
        detection_mask,
        gt_mask,
    )

    objectness = get_objectness(logits, gt_mask, matches).mean()
    bbox_l1 = get_bbox_l1(
        bbox_pre, bbox_gt, matches, POINT_CLOUD_RANGE
    )
    bbox_iou_loss = get_bbox_iou(bbox_pre, bbox_gt, matches)
    gt_num = int(gt_mask.sum())
    matched_gt_mask = torch.zeros_like(gt_mask)
    for frame_idx, (_, gt_idx) in enumerate(matches):
        matched_gt_mask[0, frame_idx, gt_idx] = True
    matched_num = int(matched_gt_mask.sum())

    tp = 0
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() == 0:
            continue
        iou = paired_axis_aligned_iou_3d(
            bbox_pre[0, frame_idx, pred_idx],
            bbox_gt[0, frame_idx, gt_idx],
        )
        tp += int((iou >= IOU_THRESHOLD).sum())

    detected_num = int(detection_mask.sum())
    fn = gt_num - tp
    fp = detected_num - matched_num
    tn = (
        detection_mask.numel()
        - detected_num
        - (gt_num - matched_num)
    )
    tp = torch.tensor(tp)
    fp = torch.tensor(fp)
    fn = torch.tensor(fn)
    tn = torch.tensor(tn)
    precision = get_precision(tp, fp, fn, tn)
    recall = get_recall(tp, fp, fn, tn)
    accuracy = get_acc(tp, fp, fn, tn)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"\n去重前帧数: {loaded['pose_gt'].shape[0] * loaded['pose_gt'].shape[1]}")
    print(f"去重后帧数: {len(unique_paths)}")
    print(f"GT 人数: {gt_num}")
    print(f"有效 query 数: {int(detection_mask.sum())}")
    print(f"匹配人数: {matched_num}")
    print(f"未匹配 GT 数: {gt_num - matched_num}")
    print(f"未匹配 query 数: {int(detection_mask.sum()) - matched_num}")
    print("\nBBox metrics")
    print(f"参与框误差计算人数: {matched_num}/{gt_num}")
    print(f"objectness: {objectness.item():.6f}")
    if matched_num:
        print(f"bbox_l1: {bbox_l1[matched_gt_mask].mean().item():.6f}")
        print(
            "bbox_iou: "
            f"{(1 - bbox_iou_loss[matched_gt_mask]).mean().item():.6f}"
        )
    print(f"TP: {int(tp)}, TN: {int(tn)}, FP: {int(fp)}, FN: {int(fn)}")
    print(f"precision: {precision.item():.6f}")
    print(f"recall: {recall.item():.6f}")
    print(f"accuracy: {accuracy.item():.6f}")
    print(f"f1: {f1.item():.6f}")

    # 姿态误差在 confidence 筛选后的匈牙利匹配人体上计算。
    pred_people = []
    gt_people = []
    flat_pose_pre = pose_pre.flatten(0, 1)
    flat_pose_gt = pose_gt.flatten(0, 1)
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        if pred_idx.numel() == 0:
            continue
        pred_people.append(flat_pose_pre[frame_idx, pred_idx])
        gt_people.append(flat_pose_gt[frame_idx, gt_idx])

    valid_pose_num = sum(value.shape[0] for value in pred_people)
    print("\nPose metrics")
    print(f"参与姿态评估人数: {valid_pose_num}/{gt_num}")
    if valid_pose_num:
        pred_people = torch.cat(pred_people)
        gt_people = torch.cat(gt_people)
        print(f"mpjpe: {get_mpjpe(pred_people, gt_people).mean().item():.6f}")
        pred_root = (pred_people[:, 11] + pred_people[:, 12]) / 2
        gt_root = (gt_people[:, 11] + gt_people[:, 12]) / 2
        root_relative_mpjpe = get_mpjpe(
            pred_people - pred_root[:, None],
            gt_people - gt_root[:, None],
        ).mean()
        print(f"root_relative_mpjpe: {root_relative_mpjpe.item():.6f}")
        print(f"pampjpe: {get_pampjpe(pred_people, gt_people).mean().item():.6f}")
        print(
            "bone_length: "
            f"{get_bone_length(pred_people, gt_people, type='coco').mean().item():.6f}"
        )

    matched_action_pre = []
    matched_action_gt = []
    for frame_idx, (pred_idx, gt_idx) in enumerate(matches):
        matched_action_pre.append(action_pre[0, frame_idx, pred_idx])
        matched_action_gt.append(action_gt[0, frame_idx, gt_idx])

    print("\nAction metrics")
    if matched_num:
        matched_action_pre = torch.cat(matched_action_pre).argmax(dim=-1)
        matched_action_gt = torch.cat(matched_action_gt).argmax(dim=-1)
        action_accuracy = (
            matched_action_pre == matched_action_gt
        ).float().mean()
        confusion = torch.zeros(4, 4, dtype=torch.long)
        for gt_label, pred_label in zip(
            matched_action_gt,
            matched_action_pre,
        ):
            confusion[gt_label, pred_label] += 1
        print(f"accuracy: {action_accuracy.item():.6f}")
        print(f"labels: {LABEL_NAMES}")
        print("confusion_matrix (row=GT, col=Pred):")
        print(confusion)

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    video_writer = None
    video_size = None
    with TemporaryDirectory(dir="/tmp") as temp_dir:
        try:
            for frame_idx in tqdm(
                range(len(unique_paths)),
                desc="Rendering video",
            ):
                pred_idx, gt_idx = matches[frame_idx]
                aligned_pose = torch.zeros_like(
                    pose_gt[:, frame_idx : frame_idx + 1]
                )
                aligned_pose[0, 0, gt_idx] = pose_pre[
                    0, frame_idx, pred_idx
                ]
                aligned_action = torch.zeros_like(
                    action_gt[:, frame_idx : frame_idx + 1]
                )
                aligned_action[..., 3] = 1
                aligned_action[0, 0, gt_idx] = action_pre[
                    0, frame_idx, pred_idx
                ]

                R = high_to_low_R[0, frame_idx]
                t = high_to_low_t[0, frame_idx]
                plot_pc = pc[:, frame_idx : frame_idx + 1].clone()
                plot_pc[..., :3] = transform_xyz(plot_pc[..., :3], R, t)
                plot_pose_pre = transform_xyz(aligned_pose, R, t)
                plot_pose_gt = transform_xyz(
                    pose_gt[:, frame_idx : frame_idx + 1], R, t
                )
                plot_bbox_pre = transform_bbox(
                    bbox_pre[:, frame_idx : frame_idx + 1], R, t
                )
                plot_bbox_gt = transform_bbox(
                    bbox_gt[:, frame_idx : frame_idx + 1], R, t
                )

                frame_path = Path(temp_dir) / f"{frame_idx:06d}.png"
                plt_fig(
                    frame_path,
                    pre={
                        "pose": plot_pose_pre,
                        "bbox": plot_bbox_pre,
                        "objectness_logits": logits[
                            :, frame_idx : frame_idx + 1
                        ],
                        "action_logits": aligned_action,
                    },
                    gt={
                        "padded": plot_pose_gt,
                        "bbox": plot_bbox_gt,
                        "mask": gt_mask[:, frame_idx : frame_idx + 1],
                        "action": action_gt[:, frame_idx : frame_idx + 1],
                        "action_label": LABEL_NAMES,
                    },
                    model_input={
                        "input": plot_pc,
                        "mask": pc_valid[:, frame_idx : frame_idx + 1],
                    },
                    matches=[(pred_idx, gt_idx)],
                    horizontal=True,
                    dpi=100,
                )

                image = cv2.imread(str(frame_path))
                if video_writer is None:
                    video_size = (image.shape[1], image.shape[0])
                    video_writer = cv2.VideoWriter(
                        str(VIDEO_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        VIDEO_FPS,
                        video_size,
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError("无法创建 MP4 视频")
                elif (image.shape[1], image.shape[0]) != video_size:
                    image = cv2.resize(image, video_size)
                video_writer.write(image)
        finally:
            if video_writer is not None:
                video_writer.release()

    print(f"\n可视化视频已保存到 {VIDEO_PATH}")


def run_inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detect_model, pose_model = load_models(device)
    dataloader = build_dataloader()
    print(f"模型已加载到 {device}")
    print(f"共 {len(dataloader.dataset)} 个滑窗，{len(dataloader)} 个 batch")
    results = {
        key: []
        for key in (
            "pc",
            "pc_valid",
            "bbox_pre",
            "objectness_logits",
            "detection_mask",
            "pose_pre",
            "pose_pre_local",
            "pose_input",
            "pose_input_mask",
            "pose_gt",
            "bbox_gt",
            "gt_valid",
            "high_to_low_R",
            "high_to_low_t",
        )
    }

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            points = batch["radar_high_pc"]["padded"].to(device)
            mask = batch["radar_high_pc"]["mask"].to(device)

            detection = detect_model({"input": points, "mask": mask})
            scores = detection["objectness_logits"].sigmoid()
            detection_mask = scores >= SCORE_THRESHOLD
            pose_input, center = build_pose_input(
                points,
                mask,
                detection["bbox"],
                detection_mask,
            )
            pose_output = pose_model(pose_input)
            B, T, K = scores.shape
            pose_local = pose_output["pose"][:, :, 0].reshape(
                B, K, T, 17, 3
            ).permute(0, 2, 1, 3, 4)
            pose = pose_local + center[..., None, :]
            pose *= detection_mask[..., None, None]
            pose_points = pose_input["input"].reshape(
                B, K, T, points.shape[2], points.shape[3]
            ).permute(0, 2, 1, 3, 4)
            pose_mask = pose_input["mask"].reshape(
                B, K, T, points.shape[2]
            ).permute(0, 2, 1, 3)

            batch_results = {
                "pc": points,
                "pc_valid": mask,
                "bbox_pre": detection["bbox"],
                "objectness_logits": detection["objectness_logits"],
                "detection_mask": detection_mask,
                "pose_pre": pose,
                "pose_pre_local": pose_local,
                "pose_input": pose_points,
                "pose_input_mask": pose_mask,
                "pose_gt": batch["gt_for_high"]["padded"],
                "bbox_gt": batch["gt_for_high"]["bbox"],
                "gt_valid": batch["gt_for_high"]["mask"],
                "high_to_low_R": batch["high_to_low_R"],
                "high_to_low_t": batch["high_to_low_t"],
            }
            for key, value in batch_results.items():
                results[key].append(value.detach().cpu())

    results = {
        key: torch.cat(value, dim=0)
        for key, value in results.items()
    }
    results["input_key"] = "radar_high_pc"
    results["target_key"] = "gt_for_high"
    results["radar_paths"] = dataloader.dataset.data_path_list["radar_high_pc"]
    torch.save(results, RESULT_PATH)
    print(f"结果已保存到 {RESULT_PATH}")


if __name__ == "__main__":
    if STAGE == "inference":
        run_inference()
    elif STAGE == "analysis":
        analyze_results()
    else:
        raise ValueError(f"不支持的 STAGE: {STAGE}")
