"""PyTorch pose adaptation of MenghaoGuo/PCT Point_Transformer2.

Architecture reference (Jittor):
https://github.com/MenghaoGuo/PCT/blob/main/networks/cls/pct.py
Uses P4Transformer's pointnet2_ops CUDA sampling/gather/group operations.
"""

import torch
from torch import nn

from models.PointNet.PointNet import _BatchNorm, _mlp, _pointwise, _pool
from models.P4Transformer import pointnet2_utils


class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, npoint, nsample):
        super().__init__()
        self.npoint, self.nsample = npoint, nsample
        self.mlp = _mlp((2 * in_channels, out_channels, out_channels))

    def forward(self, xyz, features, mask):
        # CUDA FPS has no mask: replace padding with the first valid point.
        first_valid = mask.long().argmax(-1)
        batch = torch.arange(xyz.shape[0], device=xyz.device)
        xyz = torch.where(mask[..., None], xyz, xyz[batch, first_valid][:, None]).contiguous()
        features = torch.where(mask[..., None], features,
                               features[batch, first_valid][:, None])
        feature_channels = features.transpose(1, 2).float().contiguous()
        with torch.no_grad():
            centers_idx = pointnet2_utils.furthest_point_sample(xyz, min(self.npoint, xyz.shape[1]))
            if centers_idx.shape[1] < self.npoint:
                centers_idx = torch.cat((centers_idx, centers_idx[:, :1].expand(
                    -1, self.npoint - centers_idx.shape[1])), 1).contiguous()
            centers = pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), centers_idx).transpose(1, 2).contiguous()
            # shortcut: dense KNN suits 200-point crops; use optimized KNN for large clouds.
            distances = (centers[:, :, None] - xyz[:, None]).square().sum(-1)
            distances = distances.masked_fill(~mask[:, None], float('inf'))
            values, indices = distances.topk(min(self.nsample, xyz.shape[1]), largest=False)
            first = indices[..., :1]
            indices = torch.where(torch.isfinite(values), indices, first)
            if indices.shape[-1] < self.nsample:
                indices = torch.cat((indices, first.expand(
                    *first.shape[:-1], self.nsample - indices.shape[-1])), -1)
        center_features = pointnet2_utils.gather_operation(
            feature_channels, centers_idx).transpose(1, 2)[:, :, None]
        neighbors = pointnet2_utils.grouping_operation(
            feature_channels, indices.int().contiguous()).permute(0, 2, 3, 1)
        grouped = torch.cat((neighbors - center_features, center_features.expand_as(neighbors)), -1)
        encoded = self.mlp(grouped.flatten(0, 2))
        encoded = encoded.reshape(*grouped.shape[:3], -1).amax(2)
        return centers, encoded, torch.ones(centers.shape[:2], dtype=torch.bool, device=xyz.device)


class OffsetAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qk = nn.Linear(channels, channels // 4, bias=False)
        self.value = nn.Linear(channels, channels)
        self.transform = nn.Linear(channels, channels)
        self.norm = _BatchNorm(channels)

    def forward(self, features, position):
        x = features + position
        qk, value = self.qk(x), self.value(x)
        # FP32 attention avoids overflow in unscaled dot products under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            qk = qk.float()
            attention = (qk @ qk.transpose(1, 2)).softmax(-1)
            attention = attention / (attention.sum(1, keepdim=True) + 1e-9)
            aggregated = attention.transpose(1, 2) @ value.float()
        offset = self.transform(x - aggregated.to(x.dtype))
        offset = self.norm(offset.flatten(0, 1)).reshape_as(x)
        return x + torch.relu(offset)


class PCT(nn.Module):
    """Per-frame pose [B,T,1,J,3] from input [B,T,N,3+F] and bool mask."""

    def __init__(self, num_joints=17, feat_dim=2, stem_channels=64,
                 local_channels=(128, 256), npoints=(64, 16), nsample=32,
                 attention_layers=4, fusion_channels=1024,
                 head_channels=(512, 256), dropout=0.5):
        super().__init__()
        for name, value in (('num_joints', num_joints), ('stem_channels', stem_channels),
                            ('nsample', nsample), ('attention_layers', attention_layers),
                            ('fusion_channels', fusion_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(feat_dim) is not int or feat_dim < 0:
            raise ValueError('feat_dim must be a nonnegative integer')
        for name, values in (('local_channels', local_channels), ('npoints', npoints),
                             ('head_channels', head_channels)):
            if (not isinstance(values, (list, tuple)) or not values
                    or any(type(v) is not int or v < 1 for v in values)):
                raise ValueError(f'{name} must be a nonempty list of positive integers')
        if len(local_channels) != 2 or len(npoints) != 2 or npoints[1] > npoints[0]:
            raise ValueError('use two local stages with nonincreasing npoints')
        if local_channels[-1] < 4:
            raise ValueError('last local width must be at least 4 for Q/K projection')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.num_joints, self.feat_dim = num_joints, feat_dim
        self.stem = _mlp((3 + feat_dim, stem_channels, stem_channels))
        self.local = nn.ModuleList([
            LocalAggregation(stem_channels, local_channels[0], npoints[0], nsample),
            LocalAggregation(local_channels[0], local_channels[1], npoints[1], nsample),
        ])
        channels = local_channels[-1]
        self.pre_attention = _mlp((channels, channels))
        self.position = nn.Linear(3, channels, bias=False)
        self.attention = nn.ModuleList([OffsetAttention(channels) for _ in range(attention_layers)])
        self.fusion = nn.Sequential(
            nn.Linear(channels * (attention_layers + 1), fusion_channels, bias=False),
            _BatchNorm(fusion_channels), nn.LeakyReLU(0.2))
        self.mlp_head_pose = nn.Sequential(
            _mlp((fusion_channels, *head_channels), dropout),
            nn.Linear(head_channels[-1], num_joints * 3))

    def forward(self, model_input):
        points = model_input['input']
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError('input must be a floating-point tensor')
        if not points.is_cuda:
            raise ValueError('PCT requires CUDA input for pointnet2_ops, like P4Transformer')
        if points.ndim != 4 or min(points.shape[:3]) < 1 or points.shape[-1] < 3 + self.feat_dim:
            raise ValueError('input must be nonempty [B,T,N,D], D >= 3 + feat_dim')
        mask = model_input.get('mask')
        if mask is None:
            mask = torch.ones(points.shape[:3], dtype=torch.bool, device=points.device)
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != points.shape[:3] or mask.device != points.device):
            raise ValueError('mask must be bool [B,T,N] on the input device')
        b, t, n, _ = points.shape
        points = points[..., :3 + self.feat_dim].masked_fill(~mask[..., None], 0)
        if not torch.isfinite(points).all():
            raise ValueError('valid input points must be finite')
        points, mask = points.reshape(b * t, n, -1), mask.reshape(b * t, n)
        frame_valid = mask.any(-1)
        if not frame_valid.any():
            return {'pose': points.new_zeros(b, t, 1, self.num_joints, 3)
                    + self.mlp_head_pose[-1].weight.sum() * 0}
        points, mask = points[frame_valid], mask[frame_valid]
        xyz = points[..., :3].float()
        features = _pointwise(self.stem, points, mask)
        for layer in self.local:
            xyz, features, mask = layer(xyz, features, mask)
        local_features = features
        features = _pointwise(self.pre_attention, features, mask)
        position = self.position(xyz.to(features.dtype))
        outputs = []
        for layer in self.attention:
            features = layer(features, position)
            outputs.append(features)
        features = _pointwise(self.fusion, torch.cat((*outputs, local_features), -1), mask)
        valid_pose = self.mlp_head_pose(_pool(features, mask))
        pose = valid_pose.new_zeros(b * t, self.num_joints * 3)
        pose[frame_valid] = valid_pose
        return {'pose': pose.reshape(b, t, 1, self.num_joints, 3)}


if __name__ == '__main__':
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('PCT').to(device)
    profile_model('PCT', model, {
        'input': torch.zeros(1, 16, 200, 3 + model.feat_dim, device=device),
        'mask': torch.ones(1, 16, 200, dtype=torch.bool, device=device),
    })
