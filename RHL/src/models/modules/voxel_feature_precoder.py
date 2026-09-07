import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from src.models import VFPS


@VFPS.register("MLP_Voxel_Feature_Precoder")
class VoxelizationPrecoder(nn.Module):
    """
    体素特征预编码模块 (VoxelizationPrecoder - 终极完备版)
    完全适配混合精度 (FP16/FP32)、极值安全初始化及自动 dtype 对齐。
    """
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        self.config = config
        in_dim = config.in_dim
        hidden_dim1 = config.hidden_dim1
        hidden_dim2 = config.hidden_dim2
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU()
        )
        
        self.hidden_dim2 = hidden_dim2
        self.out_channels = hidden_dim2 * 2 + 1 + 1  # 66 维

    def forward(self, outputs: Any) -> torch.Tensor:
        points = outputs.raw_point_cloud          # [B, P, 4] -> [x, y, z, v]
        valid_mask = outputs.point_valid_mask     # [B, P] (bool)
        
        # 获取空间边界与分辨率 Tensor
        xlim = outputs.xlim                       # [xmin, xmax]
        ylim = outputs.ylim                       # [ymin, ymax]
        zlim = outputs.zlim                       # [zmin, zmax]
        voxel_size = outputs.voxel_size           # [vx, vy, vz]
        grid_shape = outputs.grid_shape           # [X, Y, Z]
        
        B, P, _ = points.shape
        device = points.device
        
        X, Y, Z = grid_shape[0].item(), grid_shape[1].item(), grid_shape[2].item()

        # 空间参数对齐点云数据类型
        min_lim = torch.stack([xlim[0], ylim[0], zlim[0]]).to(points.dtype)  # [3]
        voxel_size = voxel_size.to(points.dtype)                            # [3]

        # 1. 三轴独立的体素索引计算
        coords = torch.floor((points[..., :3] - min_lim) / voxel_size).long()  # [B, P, 3]
        
        # 3. 过滤越界点、无效填充点、以及防 NaN/Inf 崩溃的有限性检查
        in_bounds = (
            (coords[..., 0] >= 0) & (coords[..., 0] < X) &
            (coords[..., 1] >= 0) & (coords[..., 1] < Y) &
            (coords[..., 2] >= 0) & (coords[..., 2] < Z) &
            valid_mask &
            torch.isfinite(points).all(dim=-1)
        )  # [B, P]

        # 2. 三轴独立的体素中心与相对偏置 (dx, dy, dz) 计算
        coords_float = coords.to(points.dtype)
        voxel_centers = (coords_float + 0.5) * voxel_size + min_lim  # [B, P, 3]
        dx_dy_dz = points[..., :3] - voxel_centers                      # [B, P, 3]
        
        point_inputs = torch.cat([dx_dy_dz, points[..., 3:4]], dim=-1)  # [B, P, 4]

        # 【鲁棒性修缮】自动将输入类型对齐到模型线性层权重的 dtype（解决混合精度下 mat1/mat2 dtype 不一致错误）
        target_dtype = self.point_mlp[0].weight.dtype
        if point_inputs.dtype != target_dtype:
            point_inputs = point_inputs.to(target_dtype)

        # 5. 通过 PointNet MLP 升维
        point_features = self.point_mlp(point_inputs)  # [B, P, hidden_dim2]
        C_feat = self.hidden_dim2

        # 6. 全局唯一体素展平索引
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(-1, P)
        voxel_flat_coords = (
            batch_indices * (X * Y * Z) +
            coords[..., 0] * (Y * Z) +
            coords[..., 1] * Z +
            coords[..., 2]
        )  # [B, P]

        flat_in_bounds = in_bounds.view(-1)
        flat_voxel_indices = voxel_flat_coords.view(-1)[flat_in_bounds]
        flat_point_feats = point_features.view(-1, C_feat)[flat_in_bounds]  # 动态决定后续特征计算的 dtype

        total_voxels = B * X * Y * Z
        feat_dtype = flat_point_feats.dtype  

        # 7. 体素内聚合（使用 feat_dtype 统一初始化）
        ones = torch.ones(flat_voxel_indices.shape[0], 1, dtype=feat_dtype, device=device)
        voxel_counts = torch.zeros(total_voxels, 1, dtype=feat_dtype, device=device)
        voxel_counts.scatter_add_(0, flat_voxel_indices.unsqueeze(1).expand(-1, 1), ones)

        valid_voxel_mask = (voxel_counts.squeeze(-1) > 0)
        safe_counts = torch.clamp(voxel_counts, min=1.0)

        # (1) Masked Mean
        sum_feats = torch.zeros(total_voxels, C_feat, dtype=feat_dtype, device=device)
        sum_feats.scatter_add_(0, flat_voxel_indices.unsqueeze(1).expand(-1, C_feat), flat_point_feats)
        voxel_mean = sum_feats / safe_counts

        # (2) Masked Max (使用 torch.finfo 兼容 FP16/FP32 的极小值初始化)
        min_val = torch.finfo(feat_dtype).min if feat_dtype.is_floating_point else -1e9
        max_feats = torch.full((total_voxels, C_feat), min_val, dtype=feat_dtype, device=device)
        
        if flat_voxel_indices.numel() > 0:
            if hasattr(torch, 'scatter_reduce'):
                max_feats = max_feats.scatter_reduce(
                    0, 
                    flat_voxel_indices.unsqueeze(1).expand(-1, C_feat), 
                    flat_point_feats, 
                    reduce='amax', 
                    include_self=False
                )
            else:
                for i in range(C_feat):
                    max_feats[:, i].index_reduce_(0, flat_voxel_indices, flat_point_feats[:, i], reduce='amax', include_self=False)
        
        # 将无点体素重置为 0.0
        max_feats = torch.where(
            valid_voxel_mask.unsqueeze(1).expand(-1, C_feat),
            max_feats,
            torch.zeros_like(max_feats)
        )

        # (3) 密度特征: log(1 + count)
        voxel_density = torch.log(1.0 + voxel_counts)

        # 8. 拼接稀疏特征
        sparse_features_base = torch.cat([voxel_mean, max_feats, voxel_density], dim=-1)
        C_base = sparse_features_base.shape[-1]

        # 9. 转换为稠密网格并追加 Occupancy 通道
        dense_grid_base = sparse_features_base.view(B, X, Y, Z, C_base).permute(0, 4, 1, 2, 3).contiguous()
        occupancy_grid = valid_voxel_mask.view(B, X, Y, Z).to(feat_dtype).unsqueeze(1)

        # 10. 输出 [B, 66, X, Y, Z]
        dense_voxel_features = torch.cat([dense_grid_base, occupancy_grid], dim=1)

        return dense_voxel_features