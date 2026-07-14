from pathlib import Path
import numpy as np
import pickle
import torch

def get_gt_data(path: Path|str) -> np.ndarray:
    with open(path, 'rb') as ff:
        gt = pickle.load(ff)
    has_nan = np.isnan(gt).any(axis=(1, 2))  # 形状 (a,)
    gt = gt[~has_nan]  # 形状 (a-1, b, c)
    return gt

def get_gt_boxes(gt: torch.Tensor, gt_mask: torch.Tensor, threshold: float=0.1):
    """
    gt: [B, T, K, J, 3]
    gt_mask: [B, T, K] - 布尔值，True表示有效
    
    Returns: bbox_3d [B, T, K, 6]
    """
    mask_expanded = gt_mask.unsqueeze(-1).unsqueeze(-1).expand_as(gt)

    # 使用 +/-inf 忽略无效人，避免依赖 torch.nanmin/nanmax。
    min_xyz = gt.masked_fill(~mask_expanded, torch.inf).amin(dim=-2) - threshold
    max_xyz = gt.masked_fill(~mask_expanded, -torch.inf).amax(dim=-2) + threshold
    
    bbox_3d = torch.cat([min_xyz, max_xyz], dim=-1)  # [B, T, K, 6]
    return bbox_3d

def get_gt_detection_targets(
    gt: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap_shape: tuple = (64, 64),
    xy_limits: tuple = ((0.0, 5.0), (-2.0, 2.0)),
    box_margin: float = 0.1,
    sigma: float = 1.0,
    clip_boxes: bool = True,
    require_full_box_inside: bool = False,
):
    """
    Returns:
        center_heatmap: [B, T, 1, H, W]
            GT heatmap for Center Heatmap Head.
        center_indices: [B, T, K, 2]
            Integer [row, col] index of GT center on heatmap.
        center_offsets: [B, T, K, 2]
            Fractional offset [d_col, d_row] for Center Offset Head.
        box_sizes: [B, T, K, 4]
            GT [left, top, right, bottom] distances in heatmap coordinates.
        inside: [B, T, K]
            Valid mask for offset and size losses.
    """
    B, T, K, J, _ = gt.shape
    H, W = heatmap_shape
    device = gt.device
    dtype = gt.dtype

    valid = gt_mask.to(device=device, dtype=torch.bool)

    # ----------------------------
    # 1. Build GT xy boxes from joints
    # ----------------------------
    xy = gt[..., :2]  # [B, T, K, J, 2]
    joint_mask = valid[..., None, None].expand_as(xy)

    min_xy = xy.masked_fill(~joint_mask, torch.inf).amin(dim=-2)   # [B, T, K, 2]
    max_xy = xy.masked_fill(~joint_mask, -torch.inf).amax(dim=-2)  # [B, T, K, 2]

    min_xy = min_xy - box_margin
    max_xy = max_xy + box_margin

    x1 = min_xy[..., 0]
    y1 = min_xy[..., 1]
    x2 = max_xy[..., 0]
    y2 = max_xy[..., 1]

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    (x_min, x_max), (y_min, y_max) = xy_limits

    # ----------------------------
    # 2. Validity mask
    # ----------------------------
    center_inside = (
        valid
        & (center_x >= x_min) & (center_x <= x_max)
        & (center_y >= y_min) & (center_y <= y_max)
    )

    full_box_inside = (
        valid
        & (x1 >= x_min) & (x2 <= x_max)
        & (y1 >= y_min) & (y2 <= y_max)
    )

    inside = full_box_inside if require_full_box_inside else center_inside

    # ----------------------------
    # 3. World xy -> heatmap continuous coordinates
    #    Standard image/ROIAlign convention:
    #       row corresponds to y
    #       col corresponds to x
    # ----------------------------
    x1_col = (x1 - x_min) / (x_max - x_min) * (W - 1)
    x2_col = (x2 - x_min) / (x_max - x_min) * (W - 1)
    y1_row = (y1 - y_min) / (y_max - y_min) * (H - 1)
    y2_row = (y2 - y_min) / (y_max - y_min) * (H - 1)

    center_col = (center_x - x_min) / (x_max - x_min) * (W - 1)
    center_row = (center_y - y_min) / (y_max - y_min) * (H - 1)

    if clip_boxes:
        x1_col = x1_col.clamp(0, W - 1)
        x2_col = x2_col.clamp(0, W - 1)
        y1_row = y1_row.clamp(0, H - 1)
        y2_row = y2_row.clamp(0, H - 1)

    # ----------------------------
    # 4. Center index and offset
    # ----------------------------
    center_col_int = center_col.floor().clamp(0, W - 1)
    center_row_int = center_row.floor().clamp(0, H - 1)

    offset_col = center_col - center_col_int
    offset_row = center_row - center_row_int

    center_indices = torch.stack(
        [center_row_int.long(), center_col_int.long()],
        dim=-1,
    )  # [B, T, K, 2], [row, col]

    center_offsets = torch.stack(
        [offset_col, offset_row],
        dim=-1,
    )  # [B, T, K, 2], [dx/col, dy/row]

    # ----------------------------
    # 5. Box size target: [left, top, right, bottom]
    #    All in heatmap coordinates.
    # ----------------------------
    left = center_col - x1_col
    top = center_row - y1_row
    right = x2_col - center_col
    bottom = y2_row - center_row

    box_sizes = torch.stack(
        [left, top, right, bottom],
        dim=-1,
    )  # [B, T, K, 4]

    # Avoid invalid targets polluting later operations.
    center_offsets = center_offsets.masked_fill(~inside[..., None], 0.0)
    box_sizes = box_sizes.masked_fill(~inside[..., None], 0.0)

    # ----------------------------
    # 6. Center heatmap target
    # ----------------------------
    row_for_hm = center_row_int.masked_fill(~inside, 0).reshape(B, T, K, 1, 1)
    col_for_hm = center_col_int.masked_fill(~inside, 0).reshape(B, T, K, 1, 1)

    rows = torch.arange(H, device=device, dtype=dtype).reshape(1, 1, 1, H, 1)
    cols = torch.arange(W, device=device, dtype=dtype).reshape(1, 1, 1, 1, W)

    dist_sq = (rows - row_for_hm).square() + (cols - col_for_hm).square()

    heatmaps_per_person = torch.exp(-dist_sq / (sigma ** 2))
    heatmaps_per_person = heatmaps_per_person * inside[..., None, None].to(dtype)

    center_heatmap = heatmaps_per_person.sum(dim=2, keepdim=True)

    return center_heatmap, center_indices, center_offsets, box_sizes, inside
