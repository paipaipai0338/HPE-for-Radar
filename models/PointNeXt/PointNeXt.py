"""PointNeXt encoder adapted to this repository's per-person pose interface.

Architecture reference: guochengqian/openpoints/models/backbone/pointnext.py.
Copyright (c) Guocheng Qian; MIT license, see LICENSE in this directory.
The point operations use PyTorch; OpenPoints/CUDA extensions are not required.
"""

import math

import torch
from torch import nn


def _gather(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch.reshape(-1, *([1] * (indices.ndim - 1))), indices]


@torch.no_grad()
def _fps(xyz, mask, count):
    """FPS over valid points; repeat a valid point when there are too few."""
    distances = xyz.new_full(xyz.shape[:2], float('inf'))
    current = mask.long().argmax(-1)
    indices = torch.empty((xyz.shape[0], count), dtype=torch.long, device=xyz.device)
    for i in range(count):
        indices[:, i] = current
        center = _gather(xyz, current[:, None])
        distances = torch.minimum(distances, (xyz - center).square().sum(-1))
        current = distances.masked_fill(~mask, -1).argmax(-1)
    return indices


def _group(xyz, features, mask, centers, radius, nsamples):
    # shortcut: dense distances suit the current 200-point crops; use CUDA ball query for large clouds.
    with torch.no_grad():
        distances = (centers[:, :, None] - xyz[:, None]).square().sum(-1)
        valid = (distances < radius * radius) & mask[:, None]
        # Ball query follows input order, repeating the first neighbor if needed.
        candidates = torch.arange(xyz.shape[1], device=xyz.device).expand_as(distances)
        indices = candidates.masked_fill(~valid, xyz.shape[1]).topk(
            min(nsamples, xyz.shape[1]), dim=-1, largest=False, sorted=True
        ).values
        first = indices[..., :1].clamp_max(xyz.shape[1] - 1)
        indices = torch.where(indices == xyz.shape[1], first, indices)
        if indices.shape[-1] < nsamples:
            indices = torch.cat((indices, first.expand(*first.shape[:-1], nsamples - indices.shape[-1])), -1)
    dp = (_gather(xyz, indices) - centers[:, :, None]) / radius
    neighbors = _gather(features.transpose(1, 2), indices)
    return torch.cat((dp.to(neighbors.dtype), neighbors), -1).permute(0, 3, 1, 2)


def _conv(in_channels, out_channels, dim, activation=True):
    conv = nn.Conv2d if dim == 2 else nn.Conv1d
    norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d
    layers = [conv(in_channels, out_channels, 1, bias=False), norm(out_channels)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, stride, radius, nsamples, layers, use_res):
        super().__init__()
        self.stride, self.radius, self.nsamples = stride, radius, nsamples
        self.use_res = use_res
        channels = [in_channels + 3] + [out_channels // 2] * (layers - 1) + [out_channels]
        self.convs = nn.Sequential(*[
            _conv(a, b, 2, activation=not (use_res and i == layers - 1))
            for i, (a, b) in enumerate(zip(channels, channels[1:]))
        ])
        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if use_res and in_channels != out_channels else nn.Identity())

    def forward(self, xyz, features, mask):
        indices = _fps(xyz, mask, max(1, xyz.shape[1] // self.stride))
        centers = _gather(xyz, indices)
        next_mask = _gather(mask, indices)
        grouped = _group(xyz, features, mask, centers, self.radius, self.nsamples)
        output = self.convs(grouped).amax(-1)
        if self.use_res:
            identity = _gather(features.transpose(1, 2), indices).transpose(1, 2)
            output = torch.relu(output + self.skip(identity))
        return centers, output.masked_fill(~next_mask[:, None], 0), next_mask


class InvResMLP(nn.Module):
    def __init__(self, channels, radius, nsamples, expansion):
        super().__init__()
        self.radius, self.nsamples = radius, nsamples
        self.local = _conv(channels + 3, channels, 2)
        self.pointwise = nn.Sequential(
            _conv(channels, channels * expansion, 1),
            _conv(channels * expansion, channels, 1, activation=False),
        )

    def forward(self, xyz, features, mask):
        grouped = _group(xyz, features, mask, xyz, self.radius, self.nsamples)
        output = torch.relu(features + self.pointwise(self.local(grouped).amax(-1)))
        return xyz, output.masked_fill(~mask[:, None], 0), mask


class PointNeXt(nn.Module):
    """Input: input [B,T,N,3+F], optional bool mask [B,T,N].

    Output: {'pose': [B,T,1,num_joints,3]}, matching P4Transformer.
    XYZ and the first feat_dim extra channels form the initial features.
    Spatial encoding is followed by centered temporal convolution before regression.
    """

    def __init__(self, num_joints=17, feat_dim=1, width=32,
                 blocks=(1, 1, 1, 1, 1), strides=(1, 4, 4, 4, 4),
                 radius=0.1, radius_scaling=2.0, nsamples=32,
                 sa_layers=2, sa_use_res=True, expansion=4,
                 mlp_dim=256, dropout=0.0, temporal_kernel_size=3):
        super().__init__()
        integer_args = dict(num_joints=num_joints, width=width, nsamples=nsamples,
                            sa_layers=sa_layers, expansion=expansion, mlp_dim=mlp_dim)
        if any(type(v) is not int or v < 1 for v in integer_args.values()):
            raise ValueError('joint/channel counts, nsamples, layers and expansion must be positive integers')
        if type(feat_dim) is not int or feat_dim < 0:
            raise ValueError('feat_dim must be a nonnegative integer')
        if not blocks or len(blocks) != len(strides):
            raise ValueError('blocks and strides must have the same nonzero length')
        if any(type(v) is not int or v < 1 for v in (*blocks, *strides)):
            raise ValueError('blocks and strides must contain positive integers')
        if strides[0] != 1 or any(s == 1 for s in strides[1:]):
            raise ValueError('use a stride-1 stem followed by downsampling stages (stride > 1)')
        if not all(math.isfinite(v) and v > 0 for v in (radius, radius_scaling)):
            raise ValueError('radius and radius_scaling must be finite and positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if type(temporal_kernel_size) is not int or temporal_kernel_size < 1 or temporal_kernel_size % 2 != 1:
            raise ValueError('temporal_kernel_size must be a positive odd integer')
        self.num_joints, self.feat_dim = num_joints, feat_dim
        self.stem = nn.Conv1d(3 + feat_dim, width, 1)
        self.encoder = nn.ModuleList()
        channels = width
        for stage, (depth, stride) in enumerate(zip(blocks, strides)):
            if stage:
                self.encoder.append(SetAbstraction(
                    channels, channels * 2, stride, radius, nsamples, sa_layers, sa_use_res
                ))
                channels *= 2
                radius *= radius_scaling
            for _ in range(depth - 1):
                self.encoder.append(InvResMLP(channels, radius, nsamples, expansion))
        self.temporal_conv = nn.Conv1d(
            channels, channels, temporal_kernel_size, padding=temporal_kernel_size // 2
        )
        self.mlp_head_pose = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, mlp_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, num_joints * 3),
        )

    def forward(self, model_input):
        points = model_input['input']
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError("input must be a real floating-point tensor")
        if points.ndim != 4 or min(points.shape[:3]) < 1 or points.shape[-1] < 3 + self.feat_dim:
            raise ValueError('input must be nonempty [B,T,N,D], with D >= 3 + feat_dim')
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
        points = points.reshape(b * t, n, -1)
        mask = mask.reshape(b * t, n)
        frame_valid = mask.any(-1)
        # Duplicate a valid point so padding cannot introduce artificial origin features.
        first = _gather(points, mask.long().argmax(-1)[:, None])
        points = torch.where(mask[..., None], points, first)
        xyz = points[..., :3].float().contiguous()
        features = self.stem(points.transpose(1, 2))
        for layer in self.encoder:
            xyz, features, mask = layer(xyz, features, mask)
        pooled = features.masked_fill(~mask[:, None], torch.finfo(features.dtype).min).amax(-1)
        pooled = pooled.masked_fill(~frame_valid[:, None], 0)
        pooled = pooled.reshape(b, t, -1).transpose(1, 2)
        pooled = (pooled + torch.relu(self.temporal_conv(pooled))).transpose(1, 2)
        pose = self.mlp_head_pose(pooled).reshape(b, t, 1, self.num_joints, 3)
        return {'pose': pose.masked_fill(~frame_valid.reshape(b, t, 1, 1, 1), 0)}


if __name__ == '__main__':
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('PointNeXt').to(device)
    x = {
        'input': torch.zeros((1, 64, 200, 5), device=device),
        'mask': torch.ones((1, 64, 200), dtype=torch.bool, device=device),
    }
    profile_model("PointNeXt", model, x)
