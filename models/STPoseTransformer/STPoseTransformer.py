"""HoloSense spatial_cross_temporal_parallel/20260427_1523 pose architecture."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _select_point_subset(features, key_padding_mask, velocity, xyz, max_points):
    count = min(max_points, features.shape[1])
    if count == features.shape[1]:
        return features, key_padding_mask, xyz
    scores = velocity.abs().squeeze(-1).masked_fill(key_padding_mask, float('-inf'))
    indices = scores.topk(count, dim=1, sorted=False).indices
    return (features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1])),
            key_padding_mask.gather(1, indices),
            xyz.gather(1, indices[..., None].expand(-1, -1, 3)))


def _voxel_local_pooling(features, key_padding_mask, xyz, voxel_size, xyz_ranges):
    if not features.numel():
        return features
    bounds = xyz.new_tensor(xyz_ranges)
    active = ~key_padding_mask & ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(-1)
    grid_size = [max(1, math.ceil((high - low) / voxel_size)) for low, high in xyz_ranges]
    coordinates = torch.minimum(torch.maximum(xyz, bounds[:, 0]), bounds[:, 1] - 1e-5)
    indices = ((coordinates - bounds[:, 0]) / voxel_size).floor().long()
    indices = torch.minimum(indices.clamp_min(0), indices.new_tensor(grid_size) - 1)
    local_ids = (indices * indices.new_tensor([1, grid_size[0], grid_size[0] * grid_size[1]])).sum(-1)
    frame_indices = torch.arange(features.shape[0], device=xyz.device)[:, None]
    voxel_ids = local_ids + frame_indices * math.prod(grid_size)
    # 无效点各占一个临时组，避免集中到同一组造成原子操作争用。
    unused_ids = -1 - torch.arange(voxel_ids.numel(), device=xyz.device).reshape_as(voxel_ids)
    voxel_ids = torch.where(active, voxel_ids, unused_ids).flatten()
    # 固定长度排序分组，避免布尔索引和 unique 的动态输出触发 CUDA 同步。
    sorted_ids, order = voxel_ids.sort()
    starts = torch.cat((sorted_ids.new_ones(1), (sorted_ids[1:] != sorted_ids[:-1]).long()))
    inverse = torch.empty_like(order).scatter_(0, order, starts.cumsum(0) - 1)
    flat_features = features.flatten(0, 1)
    pooled = torch.full_like(flat_features, float('-inf'))
    indices = inverse[:, None].expand_as(flat_features)
    pooled.scatter_reduce_(0, indices, flat_features, reduce='amax', include_self=False)
    broadcast = pooled.gather(0, indices).reshape_as(features).masked_fill(~active[..., None], 0.)
    return features + broadcast


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer('pe', encoding, persistent=False)

    def forward(self, inputs):
        return inputs + self.pe[:inputs.shape[1]].unsqueeze(0).to(inputs.dtype)


class STPoseTransformer(nn.Module):
    """XYZV point sequences [B,T,N,C] -> single-person poses [B,T,1,J,3].

    Input coordinates and output poses use the same coordinate frame.
    """

    def __init__(self, point_in_dim=4, num_joints=17, embed_dim=64,
                 temporal_joint_dim=12, vel_mask_floor=0.1, spatial_max_points=92,
                 spatial_voxel_sizes=(0.1, 0.2, 0.4),
                 spatial_voxel_x_range=(-1.2, 1.2), spatial_voxel_y_range=(-1.2, 1.2),
                 spatial_voxel_z_range=(-0.2, 2.4), num_heads=4, temporal_depth=2,
                 dropout=0.2, max_len=1024):
        super().__init__()
        counts = (point_in_dim, num_joints, embed_dim, temporal_joint_dim,
                  spatial_max_points, num_heads, temporal_depth, max_len)
        if any(type(value) is not int or value < 1 for value in counts) or point_in_dim < 4:
            raise ValueError('Model dimensions/counts must be positive integers; point_in_dim must include XYZV')
        temporal_dim = num_joints * temporal_joint_dim
        if embed_dim % num_heads or temporal_dim % num_heads or temporal_dim % 2:
            raise ValueError('Attention dimensions must be divisible by num_heads; temporal dimension must be even')
        if not 0 <= dropout < 1 or not 0 <= vel_mask_floor <= 1:
            raise ValueError('dropout must be in [0,1) and vel_mask_floor in [0,1]')
        if not spatial_voxel_sizes or any(not math.isfinite(size) or size <= 0 for size in spatial_voxel_sizes):
            raise ValueError('spatial_voxel_sizes must contain finite positive values')
        ranges = (spatial_voxel_x_range, spatial_voxel_y_range, spatial_voxel_z_range)
        if any(len(bounds) != 2 or not all(math.isfinite(value) for value in bounds)
               or bounds[0] >= bounds[1] for bounds in ranges):
            raise ValueError('Spatial voxel ranges must be finite increasing [min,max] pairs')
        self.point_in_dim, self.num_joints = point_in_dim, num_joints
        self.temporal_joint_dim = temporal_joint_dim
        self.vel_mask_floor, self.spatial_max_points = vel_mask_floor, spatial_max_points
        self.spatial_voxel_sizes, self.xyz_ranges = tuple(spatial_voxel_sizes), ranges
        self.max_len = max_len

        self.point_mlp = nn.Sequential(
            nn.Linear(point_in_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.point_feat_norm = nn.LayerNorm(embed_dim)
        self.velocity_scale_raw = nn.Parameter(torch.tensor(2.0))
        self.spatial_pos_enc = nn.Sequential(
            nn.Linear(3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.spatial_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_joint_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, temporal_joint_dim), nn.GELU(),
            nn.Linear(temporal_joint_dim, temporal_joint_dim),
        )
        self.temporal_joint_skip = nn.Linear(embed_dim, temporal_joint_dim, bias=False)
        self.temporal_pos = SinusoidalPositionalEncoding(temporal_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim, nhead=num_heads, dim_feedforward=temporal_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_depth)
        hidden_dim = max(temporal_dim // 2, num_joints * 3)
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, model_input):
        points, mask = model_input['input'], model_input['mask']
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError('input must be a real floating-point tensor')
        if points.ndim != 4 or min(points.shape[:3]) < 1 or points.shape[-1] < self.point_in_dim:
            raise ValueError('input must be nonempty [B,T,N,C] with C >= point_in_dim')
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != points.shape[:3] or mask.device != points.device:
            raise ValueError('mask must be bool [B,T,N] on the input device')
        if points.shape[1] > self.max_len:
            raise ValueError(f'Sequence length exceeds max_len={self.max_len}')
        points = points[..., :self.point_in_dim].masked_fill(~mask[..., None], 0.)
        if not torch.isfinite(points).all():
            raise ValueError('Valid input points must be finite')
        batch, frames, count, _ = points.shape
        frame_mask = mask.any(-1)
        points = points.reshape(batch * frames, count, self.point_in_dim)
        xyz, velocity = points[..., :3], points[..., 3:4]
        features = self.point_feat_norm(self.point_mlp(points))
        scale = F.softplus(self.velocity_scale_raw) + 1e-6
        gate = self.vel_mask_floor + (1. - self.vel_mask_floor) * torch.tanh(scale * velocity.abs())
        features = features * gate + self.spatial_pos_enc(xyz)

        valid = mask.reshape(batch * frames, count).clone()
        empty = ~valid.any(-1)
        valid[empty, 0] = True
        features = features.clone()
        features[empty, 0] = 0.
        features, key_padding_mask, xyz = _select_point_subset(
            features, ~valid, velocity, xyz, self.spatial_max_points)
        for size in self.spatial_voxel_sizes:
            features = _voxel_local_pooling(features, key_padding_mask, xyz, size, self.xyz_ranges)
        queries = self.query_norm(self.joint_queries.expand(batch * frames, -1, -1))
        joint_features, _ = self.spatial_cross_attn(
            queries, features, features, key_padding_mask=key_padding_mask, need_weights=False)
        joint_features = self.spatial_norm(joint_features + queries)
        temporal = self.temporal_joint_proj(joint_features) + self.temporal_joint_skip(joint_features)
        temporal = self.temporal_pos(temporal.reshape(batch, frames, self.num_joints * self.temporal_joint_dim))
        valid_frames = frame_mask.clone()
        empty_sequences = ~valid_frames.any(-1)
        valid_frames[empty_sequences, 0] = True
        temporal = temporal.clone()
        temporal[empty_sequences, 0] = 0.
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=~valid_frames)
        pose = self.head(temporal).reshape(batch, frames, 1, self.num_joints, 3)
        return {'pose': pose.masked_fill(~frame_mask[..., None, None, None], 0.)}
