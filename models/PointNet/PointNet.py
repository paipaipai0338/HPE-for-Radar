"""PointNet architecture adapted to per-person radar pose regression.

Independent PyTorch implementation; architecture reference:
https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls.py
Shared point MLPs operate on valid points only, excluding padding from BN.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _BatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        # A single valid point/frame has no batch variance to estimate.
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


def _mlp(channels, dropout=0.0):
    layers = []
    for a, b in zip(channels, channels[1:]):
        layers.extend((nn.Linear(a, b, bias=False), _BatchNorm(b), nn.ReLU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _pointwise(mlp, points, mask):
    valid = mlp(points[mask])
    output = valid.new_zeros(*mask.shape, valid.shape[-1])
    output[mask] = valid
    return output


def _pool(features, mask):
    return features.masked_fill(~mask[..., None], float('-inf')).amax(1)


class TNet(nn.Module):
    def __init__(self, channels, mlp_channels, fc_channels):
        super().__init__()
        self.channels = channels
        self.local = _mlp((channels, *mlp_channels))
        self.global_mlp = _mlp((mlp_channels[-1], *fc_channels))
        # Explicit parameters preserve identity initialization through model_init.
        self.transform_weight = nn.Parameter(torch.zeros(channels * channels, fc_channels[-1]))
        self.transform_bias = nn.Parameter(torch.zeros(channels * channels))

    def forward(self, points, mask):
        features = _pool(_pointwise(self.local, points, mask), mask)
        features = self.global_mlp(features)
        delta = F.linear(features, self.transform_weight, self.transform_bias)
        identity = torch.eye(self.channels, device=delta.device, dtype=delta.dtype)
        return delta.reshape(-1, self.channels, self.channels) + identity


class PointNet(nn.Module):
    """Input [B,T,N,3+F] with optional bool mask; pose [B,T,1,J,3]."""

    def __init__(self, num_joints=17, feat_dim=2, dropout=0.3,
                 transform_reg_weight=0.001, local_channels=(64, 64),
                 encoder_channels=(64, 128, 1024), head_channels=(512, 256),
                 tnet_mlp_channels=(64, 128, 1024), tnet_fc_channels=(512, 256)):
        super().__init__()
        if type(num_joints) is not int or num_joints < 1:
            raise ValueError('num_joints must be a positive integer')
        if type(feat_dim) is not int or feat_dim < 0:
            raise ValueError('feat_dim must be a nonnegative integer')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not math.isfinite(transform_reg_weight) or transform_reg_weight < 0:
            raise ValueError('transform_reg_weight must be finite and nonnegative')
        for name, channels in (
                ('local_channels', local_channels), ('encoder_channels', encoder_channels),
                ('head_channels', head_channels), ('tnet_mlp_channels', tnet_mlp_channels),
                ('tnet_fc_channels', tnet_fc_channels)):
            if (not isinstance(channels, (list, tuple)) or not channels
                    or any(type(c) is not int or c < 1 for c in channels)):
                raise ValueError(f'{name} must be a nonempty list of positive integers')
        self.num_joints, self.feat_dim = num_joints, feat_dim
        self.transform_reg_weight = transform_reg_weight
        self.input_transform = TNet(3, tnet_mlp_channels, tnet_fc_channels)
        self.local = _mlp((3 + feat_dim, *local_channels))
        self.feature_transform = TNet(local_channels[-1], tnet_mlp_channels, tnet_fc_channels)
        self.encoder = _mlp((local_channels[-1], *encoder_channels))
        self.mlp_head_pose = nn.Sequential(
            _mlp((encoder_channels[-1], *head_channels), dropout),
            nn.Linear(head_channels[-1], num_joints * 3))

    def forward(self, model_input):
        points = model_input['input']
        if not isinstance(points, torch.Tensor) or not points.is_floating_point():
            raise TypeError('input must be a floating-point tensor')
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
            zero = self.mlp_head_pose[-1].weight.sum() * 0
            return {'pose': points.new_zeros(b, t, 1, self.num_joints, 3) + zero,
                    'transform_regularization': zero}
        points, mask = points[frame_valid], mask[frame_valid]
        xyz = points[..., :3] @ self.input_transform(points[..., :3], mask)
        features = _pointwise(self.local, torch.cat((xyz, points[..., 3:]), -1), mask)
        transform = self.feature_transform(features, mask)
        features = _pointwise(self.encoder, features @ transform, mask)
        valid_pose = self.mlp_head_pose(_pool(features, mask))
        pose = valid_pose.new_zeros(b * t, self.num_joints * 3)
        pose[frame_valid] = valid_pose
        output = {'pose': pose.reshape(b, t, 1, self.num_joints, 3)}
        if self.training:
            # Keep the orthogonality penalty in FP32 under mixed precision.
            with torch.autocast(device_type=points.device.type, enabled=False):
                transform = transform.float()
                diff = transform @ transform.transpose(1, 2) - torch.eye(transform.shape[-1], device=points.device)
                output['transform_regularization'] = (
                    0.5 * diff.square().sum((1, 2)).mean() * self.transform_reg_weight)
        return output


if __name__ == '__main__':
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('PointNet').to(device)
    profile_model('PointNet', model, {
        'input': torch.zeros(1, 16, 200, 3 + model.feat_dim, device=device),
        'mask': torch.ones(1, 16, 200, dtype=torch.bool, device=device),
    })
