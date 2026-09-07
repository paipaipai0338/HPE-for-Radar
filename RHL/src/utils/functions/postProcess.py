import numpy as np
import torch
from scipy import ndimage
from typing import List, Dict


def split_fragmented_masks_3d(
    results: List[Dict[str, torch.Tensor]],
    connectivity: int = 26,
) -> List[Dict[str, torch.Tensor]]:
    """
    处理 3D 掩码分裂情况：
    对每个掩码做 3D 连通域分解，将互不连通的子块拆分为独立的实例掩码，
    每个拆分出来的掩码继承原实例的 score 和 box。

    Args:
        results: 模型输出列表，
                 每个元素包含 {"scores": Tensor, "boxes": Tensor, "masks": Tensor(num_keep, X, Y, Z)}
        connectivity (int): 3D 连通域邻域类型，支持 6、18 或 26 邻域（默认 26）。

    Returns:
        List[Dict[str, torch.Tensor]]: 拆分后的检测结果列表。
    """
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(rank=3, connectivity=2)
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(rank=3, connectivity=3)
    else:
        raise ValueError("3D connectivity 必须为 6、18 或 26")

    processed_results = []

    for item in results:
        scores = item["scores"]
        boxes = item["boxes"]
        masks = item["masks"]  # (num_keep, X, Y, Z), dtype=torch.long

        if masks.shape[0] == 0:
            processed_results.append(item)
            continue

        device = masks.device
        num_instances, X, Y, Z = masks.shape
        masks_np = masks.cpu().numpy().astype(np.uint8)

        new_scores = []
        new_boxes = []
        new_masks = []

        for i in range(num_instances):
            mask = masks_np[i]
            if np.count_nonzero(mask) == 0:
                continue

            labeled_mask, num_labels = ndimage.label(mask, structure=struct)
            if num_labels == 0:
                continue

            # 将每个独立的连通块分别拆出作为独立的新实例
            for lbl in range(1, num_labels + 1):
                sub_mask = (labeled_mask == lbl).astype(np.int64)
                new_masks.append(sub_mask)
                new_scores.append(scores[i])
                new_boxes.append(boxes[i])

        if len(new_masks) > 0:
            out_scores = torch.stack(new_scores, dim=0)
            out_boxes = torch.stack(new_boxes, dim=0)
            out_masks = torch.from_numpy(np.stack(new_masks, axis=0)).to(
                device=device, dtype=torch.long
            )
        else:
            out_scores = scores[:0]
            out_boxes = boxes[:0]
            out_masks = torch.empty((0, X, Y, Z), dtype=torch.long, device=device)

        processed_results.append({
            "scores": out_scores,
            "boxes": out_boxes,
            "masks": out_masks,
        })

    return processed_results

def filter_outlier_masks_3d(
    results: List[Dict[str, torch.Tensor]],
    min_volume_abs: int = 10,
) -> List[Dict[str, torch.Tensor]]:
    """
    对所有掩码进行统一的离群噪点过滤：
    计算每个掩码的有效体素体积，直接剔除小于 min_volume_abs 的掩码。

    Args:
        results: 检测结果列表，每个元素包含 {"scores": Tensor, "boxes": Tensor, "masks": Tensor(num_keep, X, Y, Z)}
        min_volume_abs (int): 掩码最小体素体积阈值，小于该值的掩码被直接丢弃。

    Returns:
        List[Dict[str, torch.Tensor]]: 过滤离群噪点后的结果列表。
    """
    processed_results = []

    for item in results:
        scores = item["scores"]
        boxes = item["boxes"]
        masks = item["masks"]  # (num_keep, X, Y, Z)

        if masks.shape[0] == 0:
            processed_results.append(item)
            continue

        # 在 GPU/CPU 上直接向量化计算每个 mask 的有效体素总数
        volumes = masks.sum(dim=(1, 2, 3))  # shape: (num_keep,)
        keep = volumes >= min_volume_abs

        processed_results.append({
            "scores": scores[keep],
            "boxes": boxes[keep],
            "masks": masks[keep],
        })

    return processed_results

def masks_to_bounding_cubes_from_outputs(
    outputs: dict, voxel_config: dict
) -> np.ndarray:
    """从后处理 outputs 字典中提取体素掩码，并转换为真实物理坐标下的轴对齐最小外接 3D Bounding Cube。

    参数:
        outputs (dict): 包含 'masks' 字段的检测后处理结果字典，
                        'masks' 形状为 (N, X, Y, Z)。
        voxel_config (dict): 体素配置字典，需包含:
            - "region": {"XLIM": [x_min, x_max], "YLIM": [y_min, y_max],
            "ZLIM": [z_min, z_max]}
            - "resolution": [dx, dy, dz]

    返回:
        np.ndarray: 形状为 (N, 6) 的外接盒数组，格式为 [x_min, y_min, z_min, x_max,
        y_max, z_max]。
                    若某实例掩码全为 0，则该行填充为 NaN。
    """
    region = voxel_config["region"]
    xlim = region["XLIM"]
    ylim = region["YLIM"]
    zlim = region["ZLIM"]
    voxel_size = voxel_config["resolution"]  # [dx, dy, dz]

    masks = outputs["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    num_instances = masks.shape[0]
    bounding_cubes = np.full((num_instances, 6), np.nan, dtype=np.float32)

    for i in range(num_instances):
        mask = masks[i]
        voxel_bools = mask > 0

        if not np.any(voxel_bools):
            continue

        # 计算有效体素在三轴上的最小/最大索引
        valid_x, valid_y, valid_z = np.where(voxel_bools)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
        min_z, max_z = valid_z.min(), valid_z.max()

        # 映射至物理坐标系
        x_min = xlim[0] + min_x * voxel_size[0]
        x_max = xlim[0] + (max_x + 1) * voxel_size[0]
        y_min = ylim[0] + min_y * voxel_size[1]
        y_max = ylim[0] + (max_y + 1) * voxel_size[1]
        z_min = zlim[0] + min_z * voxel_size[2]
        z_max = zlim[0] + (max_z + 1) * voxel_size[2]

        bounding_cubes[i] = [x_min, y_min, z_min, x_max, y_max, z_max]

    return bounding_cubes

def add_bounding_cubes_to_outputs(
    outputs: dict, voxel_config: dict
) -> dict:
    """计算体素掩码的 3D 外接盒并直接写入 outputs 字典中。

    参数:
        outputs (dict): 包含 'scores', 'boxes', 'masks' 的字典。
        voxel_config (dict): 体素空间配置。

    返回:
        dict: 更新后包含 'bounding_cubes' (N, 6) 的 outputs 字典。
    """
    region = voxel_config["region"]
    xlim = region["XLIM"]
    ylim = region["YLIM"]
    zlim = region["ZLIM"]
    voxel_size = voxel_config["resolution"]

    masks = outputs["masks"]
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = masks

    num_instances = masks_np.shape[0]
    bounding_cubes = np.full((num_instances, 6), np.nan, dtype=np.float32)

    for i in range(num_instances):
        mask = masks_np[i]
        voxel_bools = mask > 0

        if not np.any(voxel_bools):
            continue

        valid_x, valid_y, valid_z = np.where(voxel_bools)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
        min_z, max_z = valid_z.min(), valid_z.max()

        x_min = xlim[0] + min_x * voxel_size[0]
        x_max = xlim[0] + (max_x + 1) * voxel_size[0]
        y_min = ylim[0] + min_y * voxel_size[1]
        y_max = ylim[0] + (max_y + 1) * voxel_size[1]
        z_min = zlim[0] + min_z * voxel_size[2]
        z_max = zlim[0] + (max_z + 1) * voxel_size[2]

        bounding_cubes[i] = [x_min, y_min, z_min, x_max, y_max, z_max]

    outputs["bounding_cubes"] = bounding_cubes
    return outputs

def compute_3d_iou_and_containment(box1: np.ndarray, box2: np.ndarray):
    """计算两个 3D 轴对齐框 (AABB) 的 IoU 和 包含率(IoMin)。

    参数格式假设为 [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    # 1. 计算重叠区域尺寸
    min_coords = np.maximum(box1[:3], box2[:3])
    max_coords = np.minimum(box1[3:], box2[3:])
    inter_dims = np.maximum(0.0, max_coords - min_coords)
    inter_vol = np.prod(inter_dims)

    if inter_vol == 0:
        return 0.0, 0.0

    # 2. 计算各自体积
    vol1 = np.prod(np.maximum(0.0, box1[3:] - box1[:3]))
    vol2 = np.prod(np.maximum(0.0, box2[3:] - box2[:3]))

    union_vol = vol1 + vol2 - inter_vol
    iou = inter_vol / union_vol if union_vol > 0 else 0.0

    # 3. 计算 IoMin (重叠体积占较小框体积的比例)
    min_vol = min(vol1, vol2)
    io_min = inter_vol / min_vol if min_vol > 0 else 0.0

    return iou, io_min

def merge_overlapping_bounding_cubes(
    outputs: dict,
    iou_threshold: float = 0.5,
    io_min_threshold: float = 0.85,
) -> dict:
    """基于连通图合并高重合度及相互包含的 3D Bounding Cube，并取组内最大置信度。

    参数:
        outputs (dict): 包含 'bounding_cubes' 和 'scores' 的字典。
        iou_threshold (float): 标准 3D IoU 阈值。
        io_min_threshold (float): 包含率阈值（小框被覆盖的体积占比，默认 >= 85% 即判定合并）。

    返回:
        dict: 仅包含合并后的 {'bounding_cubes': ..., 'scores': ...} 的新字典。
    """
    cubes = outputs["bounding_cubes"]
    scores = outputs["scores"]

    is_tensor = isinstance(scores, torch.Tensor)
    if is_tensor:
        device = scores.device
        scores_np = scores.detach().cpu().numpy()
        cubes_np = (
            cubes.detach().cpu().numpy()
            if isinstance(cubes, torch.Tensor)
            else cubes
        )
    else:
        scores_np = np.array(scores)
        cubes_np = (
            cubes.detach().cpu().numpy()
            if isinstance(cubes, torch.Tensor)
            else np.array(cubes)
        )

    # 过滤无效 NaN 框
    valid_idx = np.where(~np.isnan(cubes_np).any(axis=1))[0]
    num_valid = len(valid_idx)

    if num_valid == 0:
        empty_cubes = np.empty((0, 6), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float32)
        return {
            "bounding_cubes": torch.from_numpy(empty_cubes).to(device)
            if is_tensor
            else empty_cubes,
            "scores": torch.from_numpy(empty_scores).to(device)
            if is_tensor
            else empty_scores,
        }

    valid_cubes = cubes_np[valid_idx]

    # 构建重叠/包含连通图的邻接矩阵
    adj_matrix = np.zeros((num_valid, num_valid), dtype=bool)
    for i in range(num_valid):
        adj_matrix[i, i] = True
        for j in range(i + 1, num_valid):
            iou, io_min = compute_3d_iou_and_containment(
                valid_cubes[i], valid_cubes[j]
            )
            # 满足常规 IoU 阈值 或 满足高包含率 (大包小/大部分覆盖)
            if iou > iou_threshold or io_min > io_min_threshold:
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

    # 遍历连通分支并合并
    visited = np.zeros(num_valid, dtype=bool)
    merged_cubes = []
    merged_scores = []

    for i in range(num_valid):
        if visited[i]:
            continue

        component = []
        queue = [i]
        visited[i] = True

        while queue:
            node = queue.pop(0)
            component.append(node)
            neighbors = np.where(adj_matrix[node] & ~visited)[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(neighbor)

        comp_orig_idx = valid_idx[component]

        # 1. 计算合并后外接立方体的最小/最大包围范围
        comp_cubes = cubes_np[comp_orig_idx]
        merged_cube = np.empty(6, dtype=np.float32)
        merged_cube[:3] = np.min(comp_cubes[:, :3], axis=0)
        merged_cube[3:] = np.max(comp_cubes[:, 3:], axis=0)
        merged_cubes.append(merged_cube)

        # 2. 取该连通分支内的最高置信度
        merged_scores.append(np.max(scores_np[comp_orig_idx]))

    merged_cubes = np.array(merged_cubes, dtype=np.float32)
    merged_scores = np.array(merged_scores, dtype=np.float32)

    if is_tensor:
        return {
            "bounding_cubes": torch.from_numpy(merged_cubes).to(device),
            "scores": torch.from_numpy(merged_scores).to(device),
        }

    return {"bounding_cubes": merged_cubes, "scores": merged_scores}


#