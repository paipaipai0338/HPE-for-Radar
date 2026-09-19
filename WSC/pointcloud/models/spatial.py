from __future__ import annotations

import torch
from torch import nn

from WSC.pointcloud.models.edgeconv import MaskedEdgeConvEncoder
from WSC.pointcloud.models.pointnet import MaskedPointEncoder


class PointNetEdgeConvEncoder(nn.Module):
    """Fuse global PointNet and local EdgeConv frame representations."""

    def __init__(
        self,
        input_dim: int,
        point_feature_dim: int,
        frame_feature_dim: int,
        dropout: float,
        neighbor_count: int,
    ) -> None:
        super().__init__()
        self.pointnet = MaskedPointEncoder(
            input_dim,
            point_feature_dim,
            frame_feature_dim,
            dropout,
        )
        self.edgeconv = MaskedEdgeConvEncoder(
            input_dim,
            point_feature_dim,
            frame_feature_dim,
            dropout,
            neighbor_count,
        )
        self.fusion = nn.Sequential(
            nn.Linear(frame_feature_dim * 2, frame_feature_dim),
            nn.LayerNorm(frame_feature_dim),
            nn.SiLU(),
        )

    def forward(self, point_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        global_features = self.pointnet(point_features, point_mask)
        local_features = self.edgeconv(point_features, point_mask)
        return self.fusion(torch.cat((global_features, local_features), dim=-1))
