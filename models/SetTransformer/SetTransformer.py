"""Set Transformer adapted to per-person radar pose regression.

Reference: https://proceedings.mlr.press/v97/lee19d.html
ISAB inducing vectors are learned features, not sampled spatial anchors.
"""
import torch
from torch import nn


class MAB(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, mask=None):
        attended = self.attention(query, memory, memory,
                                  key_padding_mask=None if mask is None else ~mask,
                                  need_weights=False)[0]
        features = self.norm1(query + self.dropout(attended))
        return self.norm2(features + self.dropout(self.ffn(features)))


class ISAB(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout, num_inducing_points):
        super().__init__()
        self.inducing = nn.Embedding(num_inducing_points, dim)
        self.summarize = MAB(dim, num_heads, ffn_dim, dropout)
        self.broadcast = MAB(dim, num_heads, ffn_dim, dropout)

    def forward(self, features, mask):
        queries = self.inducing.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        summary = self.summarize(queries, features, mask)
        return self.broadcast(features, summary).masked_fill(~mask[..., None], 0)


class SetTransformer(nn.Module):
    """Input [B,T,N,3+F] with bool mask; per-frame pose [B,T,1,J,3], no FPS."""

    def __init__(self, num_joints=17, feat_dim=2, dim=128, num_heads=4,
                 num_inducing_points=32, encoder_layers=2, decoder_layers=1,
                 ffn_dim=256, dropout=0.0):
        super().__init__()
        for name, value in (('num_joints', num_joints), ('dim', dim), ('num_heads', num_heads),
                            ('num_inducing_points', num_inducing_points),
                            ('encoder_layers', encoder_layers), ('decoder_layers', decoder_layers),
                            ('ffn_dim', ffn_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(feat_dim) is not int or feat_dim < 0:
            raise ValueError('feat_dim must be a nonnegative integer')
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.num_joints, self.feat_dim = num_joints, feat_dim
        self.stem = nn.Linear(3 + feat_dim, dim)
        self.encoder = nn.ModuleList([
            ISAB(dim, num_heads, ffn_dim, dropout, num_inducing_points)
            for _ in range(encoder_layers)])
        self.pool_features = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dim))
        self.joint_queries = nn.Embedding(num_joints, dim)
        self.pool = MAB(dim, num_heads, ffn_dim, dropout)
        self.decoder = nn.ModuleList([MAB(dim, num_heads, ffn_dim, dropout)
                                      for _ in range(decoder_layers)])
        self.mlp_head_pose = nn.Linear(dim, 3)

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
            return {'pose': points.new_zeros(b, t, 1, self.num_joints, 3)
                    + self.mlp_head_pose.weight.sum() * 0}
        points, mask = points[frame_valid], mask[frame_valid]
        features = self.stem(points)
        for layer in self.encoder:
            features = layer(features, mask)
        queries = self.joint_queries.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        joints = self.pool(queries, self.pool_features(features), mask)
        for layer in self.decoder:
            joints = layer(joints, joints)
        valid_pose = self.mlp_head_pose(joints)
        pose = valid_pose.new_zeros(b * t, self.num_joints, 3)
        pose[frame_valid] = valid_pose
        return {'pose': pose.reshape(b, t, 1, self.num_joints, 3)}


if __name__ == '__main__':
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('SetTransformer').to(device)
    profile_model('SetTransformer', model, {
        'input': torch.zeros(1, 16, 200, 3 + model.feat_dim, device=device),
        'mask': torch.ones(1, 16, 200, dtype=torch.bool, device=device),
    })
