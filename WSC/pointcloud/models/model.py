from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from WSC.pointcloud.models.edgeconv import MaskedEdgeConvEncoder
from WSC.pointcloud.models.pointnet import MaskedPointEncoder, ShapeEncoder
from WSC.pointcloud.models.spatial import PointNetEdgeConvEncoder
from WSC.temporal.tcn import CausalTemporalEncoder


@dataclass(frozen=True, slots=True)
class BehaviorModelConfig:
    sequence_length: int = 4
    coordinate_scale: float = 1.0
    doppler_scale: float = 2.5
    use_doppler: bool = False
    point_feature_dim: int = 64
    frame_feature_dim: int = 128
    shape_feature_dim: int = 32
    shape_statistic_dim: int = 15
    use_shape_statistics: bool = True
    use_compact_statistics: bool = False
    use_point_count: bool = False
    spatial_encoder: str = "pointnet"
    edge_neighbor_count: int = 8
    temporal_layer_count: int = 3
    temporal_kernel_size: int = 2
    dropout: float = 0.1
    class_count: int = 4


class FrameEncoder(nn.Module):
    """Fuse learned point features with physical per-frame statistics."""

    def __init__(self, config: BehaviorModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = (
            3
            + int(config.use_doppler)
            + int(config.use_point_count)
            + 4 * int(config.use_compact_statistics)
        )
        if config.spatial_encoder == "pointnet":
            self.point_encoder = MaskedPointEncoder(
                input_dim,
                config.point_feature_dim,
                config.frame_feature_dim,
                config.dropout,
            )
        elif config.spatial_encoder == "edgeconv":
            self.point_encoder = MaskedEdgeConvEncoder(
                input_dim,
                config.point_feature_dim,
                config.frame_feature_dim,
                config.dropout,
                config.edge_neighbor_count,
            )
        elif config.spatial_encoder == "pointnet_edgeconv":
            self.point_encoder = PointNetEdgeConvEncoder(
                input_dim,
                config.point_feature_dim,
                config.frame_feature_dim,
                config.dropout,
                config.edge_neighbor_count,
            )
        else:
            raise ValueError(f"Unsupported spatial_encoder: {config.spatial_encoder}")
        self.shape_encoder = None
        self.fusion = None
        if config.use_shape_statistics:
            self.shape_encoder = ShapeEncoder(config.shape_statistic_dim, config.shape_feature_dim)
            self.fusion = nn.Sequential(
                nn.Linear(config.frame_feature_dim + config.shape_feature_dim, config.frame_feature_dim),
                nn.LayerNorm(config.frame_feature_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )

    def _point_features(self, points: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3] / self.config.coordinate_scale
        features = [xyz]
        if self.config.use_doppler:
            features.append(points[..., 3:4] / self.config.doppler_scale)
        if self.config.use_point_count:
            point_count = statistics.unsqueeze(2).expand(*points.shape[:3], 1)
            features.append(point_count)
        if self.config.use_compact_statistics:
            scaled_statistics = statistics.clone()
            scaled_statistics[..., :2] /= self.config.coordinate_scale
            compact_statistics = scaled_statistics.unsqueeze(2).expand(*points.shape[:3], 4)
            features.append(compact_statistics)
        return torch.cat(features, dim=-1)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        shape_statistics: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused frame features with shape [B, T, frame_feature_dim]."""
        if points.ndim != 4 or points.shape[-1] != 4:
            raise ValueError(f"points must have shape [B, T, N, 4], got {tuple(points.shape)}")
        if point_mask.shape != points.shape[:3]:
            raise ValueError("point_mask must have shape [B, T, N]")
        statistic_dim = 4 if self.config.use_compact_statistics else int(self.config.use_point_count)
        if statistic_dim and shape_statistics.shape != (*points.shape[:2], statistic_dim):
            raise ValueError(f"shape_statistics must have shape [B, T, {statistic_dim}]")
        learned_features = self.point_encoder(self._point_features(points, shape_statistics), point_mask)
        if not self.config.use_shape_statistics:
            return learned_features

        expected_shape = (*points.shape[:2], self.config.shape_statistic_dim)
        if shape_statistics.shape != expected_shape:
            raise ValueError(f"shape_statistics must have shape [B, T, {self.config.shape_statistic_dim}]")
        scaled_statistics = shape_statistics.clone()
        scaled_statistics[..., :6] /= self.config.coordinate_scale
        if self.shape_encoder is None or self.fusion is None:
            raise RuntimeError("Shape statistics branch was not initialized")
        shape_features = self.shape_encoder(scaled_statistics)
        return self.fusion(torch.cat((learned_features, shape_features), dim=-1))


class BehaviorModel(nn.Module):
    """Stable entry point for frame encoding, temporal modeling, and classification."""

    def __init__(self, config: BehaviorModelConfig = BehaviorModelConfig()) -> None:
        super().__init__()
        if config.sequence_length <= 0 or config.coordinate_scale <= 0 or config.doppler_scale <= 0:
            raise ValueError("sequence_length, coordinate_scale, and doppler_scale must be positive")
        if config.edge_neighbor_count <= 0:
            raise ValueError("edge_neighbor_count must be positive")
        statistic_modes = (config.use_shape_statistics, config.use_compact_statistics, config.use_point_count)
        if sum(statistic_modes) > 1:
            raise ValueError("Only one statistics mode can be enabled")
        feature_dimensions = (
            config.point_feature_dim,
            config.frame_feature_dim,
            config.shape_feature_dim,
            config.shape_statistic_dim,
            config.class_count,
        )
        if min(feature_dimensions) <= 0:
            raise ValueError("feature dimensions and class_count must be positive")
        self.config = config
        self.frame_encoder = FrameEncoder(config)
        self.temporal_encoder = CausalTemporalEncoder(
            config.frame_feature_dim,
            config.temporal_layer_count,
            config.temporal_kernel_size,
            config.dropout,
        )
        if self.temporal_encoder.receptive_field < config.sequence_length:
            raise ValueError("Temporal receptive field must cover the complete input sequence")
        self.classifier = nn.Sequential(
            nn.Linear(config.frame_feature_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.class_count),
        )

    def forward_sequence(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        shape_statistics: torch.Tensor,
    ) -> torch.Tensor:
        """Return causal classification logits for every frame."""
        if points.shape[1] != self.config.sequence_length:
            raise ValueError("Input sequence length differs from model configuration")
        frame_features = self.frame_encoder(points, point_mask, shape_statistics)
        temporal_features = self.temporal_encoder(frame_features)
        return self.classifier(temporal_features)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        shape_statistics: torch.Tensor,
    ) -> torch.Tensor:
        """Return the classification logits of the final frame."""
        return self.forward_sequence(points, point_mask, shape_statistics)[:, -1]
