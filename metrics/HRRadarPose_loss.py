import torch
import torch.nn.functional as F

def get_body_center_loss(C_pre, C_gt, alpha=2, beta=4):
    # C_pre     B, T, 1, X, Y, Z
    # C_gt      B, T, 1, X, Y, Z
    # loss      B, T
    C_pre = C_pre.clamp(min=1e-6, max=1 - 1e-6)
    pos_mask = C_gt.eq(1)
    neg_mask = C_gt.lt(1)

    # positive: Y == 1
    pos_loss = (1 - C_pre) ** alpha * torch.log(C_pre) * pos_mask

    # negative: Y < 1
    neg_loss = (1 - C_gt) ** beta * C_pre ** alpha * torch.log(1 - C_pre) * neg_mask

    spatial_dims = tuple(range(2, C_pre.ndim))
    num_pos = pos_mask.sum(dim=spatial_dims).clamp(min=1)
    loss = -(pos_loss.sum(dim=spatial_dims) + neg_loss.sum(dim=spatial_dims)) / num_pos
    return loss

def get_keypoint_offset_loss(K_pre, indices_gt, keypoint_offset_gt, valid):
    # K_pre                 B, T, J*3, X, Y, Z
    # indices_gt            B, T, K
    # keypoint_offset_gt    B, T, K, J, 3
    # valid                 B, T, K
    # loss                  B, T, K

    B, T, C, X, Y, Z = K_pre.shape
    K = indices_gt.shape[-1]
    J = C // 3

    # B,T,J*3,X,Y,Z -> B,T,X*Y*Z,J*3
    pred = K_pre.flatten(start_dim=-3).permute(0, 1, 3, 2)

    # [B,T,K] -> [B,T,K,J*3]
    gather_idx = indices_gt.unsqueeze(-1).expand(-1, -1, -1, C)

    pred = torch.gather(pred, dim=2, index=gather_idx)

    # [B,T,K,J,3]
    pred = pred.reshape(B, T, K, J, 3)

    # 每个 person 的所有关节坐标 L1；无效 person 不参与后续聚合。
    loss = torch.norm(pred - keypoint_offset_gt, dim=-1).mean(-1)
    return loss.masked_fill(~valid, 0.0)

