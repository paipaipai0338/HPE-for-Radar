from __future__ import annotations

import torch
from torch import nn


class MaskedPointEncoder(nn.Module):
    """Encode each frame of a padded point set with shared point-wise layers."""

    def __init__(self, input_dim: int, point_feature_dim: int, frame_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, point_feature_dim),
            nn.LayerNorm(point_feature_dim),
            nn.SiLU(),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(point_feature_dim * 3, frame_feature_dim),
            nn.LayerNorm(frame_feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, point_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """Return frame features with shape [B, T, frame_feature_dim]."""
        encoded = self.point_mlp(point_features)
        mask = point_mask.bool().unsqueeze(-1)
        counts = mask.sum(dim=2).clamp_min(1)
        masked_mean = encoded.masked_fill(~mask, 0.0).sum(dim=2) / counts
        centered = encoded - masked_mean.unsqueeze(2)
        variance = centered.square().masked_fill(~mask, 0.0).sum(dim=2) / counts
        masked_std = variance.clamp_min(torch.finfo(variance.dtype).eps).sqrt()
        masked_max = encoded.masked_fill(~mask, -torch.inf).amax(dim=2)
        masked_max = torch.where(point_mask.any(dim=2, keepdim=True), masked_max, torch.zeros_like(masked_max))
        return self.frame_mlp(torch.cat((masked_max, masked_mean, masked_std), dim=-1))


class ShapeEncoder(nn.Module):
    """Encode compact physical shape statistics computed before point sampling."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, output_dim),
            nn.SiLU(),
        )

    def forward(self, statistics: torch.Tensor) -> torch.Tensor:
        return self.mlp(statistics)
