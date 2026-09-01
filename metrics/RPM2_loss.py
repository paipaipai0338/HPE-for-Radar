import torch
import torch.nn.functional as F


def gather_at_center_indices(pred_map: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred_map: [B, T, V, C, H, W]
            V = 2 views.
            C = 2 for offset, 4 for box.
        indices: [B, T, V, K, 2]
            Your convention: [h_idx, w_idx].
            horizontal: [x_idx, y_idx]
            vertical:   [x_idx, z_idx]

    Returns:
        gathered: [B, T, V, K, C]
    """
    B, T, V, C, H, W = pred_map.shape
    K = indices.shape[3]

    h_idx = indices[..., 0].long().clamp(0, H - 1)
    w_idx = indices[..., 1].long().clamp(0, W - 1)

    linear_idx = h_idx * W + w_idx  # [B,T,V,K]

    pred_flat = pred_map.flatten(-2)  # [B,T,V,C,H*W]

    gather_idx = linear_idx.unsqueeze(3).expand(B, T, V, C, K)
    gathered = pred_flat.gather(dim=-1, index=gather_idx)  # [B,T,V,C,K]

    gathered = gathered.permute(0, 1, 2, 4, 3).contiguous()  # [B,T,V,K,C]
    return gathered


def get_center_heatmap_loss(pre: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pre: [B,T,2,1,H,W]
    gt:  [B,T,2,1,H,W]
    """
    assert pre.shape == gt.shape, f"shape mismatch: pre={pre.shape}, gt={gt.shape}"
    return ((pre - gt) ** 2).mean()


def get_center_offset_loss(
    pre: torch.Tensor,
    gt: torch.Tensor,
    indices: torch.Tensor,
    inside: torch.Tensor,
) -> torch.Tensor:
    """
    pre:     [B,T,2,2,H,W]
    gt:      [B,T,2,K,2]
    indices: [B,T,2,K,2]
    inside:  [B,T,2,K]
    """
    pred_sparse = gather_at_center_indices(pre, indices)  # [B,T,2,K,2]

    valid = inside.to(device=pre.device, dtype=pre.dtype)  # [B,T,2,K]

    # paper-style L1: sum over offset channels, average over valid centers
    per_instance_l1 = (pred_sparse - gt).abs().sum(dim=-1)  # [B,T,2,K]

    denom = valid.sum().clamp_min(1.0)
    loss = (per_instance_l1 * valid).sum() / denom
    return loss


def get_box_size_loss(
    pre: torch.Tensor,
    gt: torch.Tensor,
    indices: torch.Tensor,
    inside: torch.Tensor,
) -> torch.Tensor:
    """
    pre:     [B,T,2,4,H,W]
    gt:      [B,T,2,K,4]
    indices: [B,T,2,K,2]
    inside:  [B,T,2,K]
    """
    pred_sparse = gather_at_center_indices(pre, indices)  # [B,T,2,K,4]

    valid = inside.to(device=pre.device, dtype=pre.dtype)  # [B,T,2,K]

    # paper-style L1: sum over 4 box channels, average over valid centers
    per_instance_l1 = (pred_sparse - gt).abs().sum(dim=-1)  # [B,T,2,K]

    denom = valid.sum().clamp_min(1.0)
    loss = (per_instance_l1 * valid).sum() / denom
    return loss


def get_pose_loss(
    pre: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    pre:   [B,T,K,J,3]
    gt:    [B,T,K,J,3]
    valid: [B,T,K]
    """
    assert pre.shape == gt.shape, f"shape mismatch: pre={pre.shape}, gt={gt.shape}"

    B, T, K, J, D = pre.shape
    assert D == 3

    valid = valid.to(device=pre.device, dtype=pre.dtype)  # [B,T,K]

    dist = torch.linalg.norm(pre - gt, dim=-1)  # [B,T,K,J]

    denom = (valid.sum() * J).clamp_min(1.0)
    loss = (dist * valid[..., None]).sum() / denom
    return loss