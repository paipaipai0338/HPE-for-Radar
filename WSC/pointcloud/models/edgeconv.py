from __future__ import annotations

import torch
from torch import nn


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch_indices, indices]


class _EdgeConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbors = _gather_neighbors(features, neighbor_indices)
        neighbor_xyz = _gather_neighbors(xyz, neighbor_indices)
        centers = features.unsqueeze(2).expand_as(neighbors)
        edge_features = torch.cat(
            (centers, neighbors - centers, neighbor_xyz - xyz.unsqueeze(2)),
            dim=-1,
        )
        edge_features = edge_features.masked_fill(~neighbor_mask.unsqueeze(-1), 0.0)
        encoded = self.edge_mlp(edge_features).masked_fill(~neighbor_mask.unsqueeze(-1), -torch.inf)
        aggregated = encoded.amax(dim=2)

        # A self edge keeps the encoder defined for a frame containing only one valid point.
        self_edges = torch.cat(
            (features, torch.zeros_like(features), torch.zeros_like(xyz)),
            dim=-1,
        )
        self_encoded = self.edge_mlp(self_edges)
        aggregated = torch.where(neighbor_mask.any(dim=2, keepdim=True), aggregated, self_encoded)
        return torch.where(point_mask.unsqueeze(-1), aggregated, torch.zeros_like(aggregated))


class MaskedEdgeConvEncoder(nn.Module):
    """Encode padded point sets with two local spatial aggregation layers."""

    def __init__(
        self,
        input_dim: int,
        point_feature_dim: int,
        frame_feature_dim: int,
        dropout: float,
        neighbor_count: int,
    ) -> None:
        super().__init__()
        if neighbor_count <= 0:
            raise ValueError("neighbor_count must be positive")
        self.neighbor_count = neighbor_count
        self.edge_layers = nn.ModuleList(
            (
                _EdgeConv(input_dim, point_feature_dim),
                _EdgeConv(point_feature_dim, point_feature_dim),
            )
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(point_feature_dim * 3, frame_feature_dim),
            nn.LayerNorm(frame_feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def _neighbors(
        self,
        xyz: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_count = xyz.shape[1]
        if point_count == 0:
            raise ValueError("point sets must contain at least one padded position")
        neighbor_count = min(self.neighbor_count, max(point_count - 1, 1))
        with torch.no_grad():
            distances = torch.cdist(xyz, xyz)
            distances = distances.masked_fill(~point_mask.unsqueeze(1), torch.inf)
            diagonal = torch.eye(point_count, dtype=torch.bool, device=xyz.device).unsqueeze(0)
            distances = distances.masked_fill(diagonal, torch.inf)
            neighbor_distances, neighbor_indices = distances.topk(
                neighbor_count,
                dim=-1,
                largest=False,
                sorted=False,
            )
            neighbor_mask = point_mask.unsqueeze(-1) & torch.isfinite(neighbor_distances)
        return neighbor_indices, neighbor_mask

    def forward(self, point_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """Return frame features with shape [B, T, frame_feature_dim]."""
        batch_size, sequence_length, point_count, feature_dim = point_features.shape
        features = point_features.reshape(-1, point_count, feature_dim)
        xyz = features[..., :3]
        mask = point_mask.reshape(-1, point_count).bool()
        neighbor_indices, neighbor_mask = self._neighbors(xyz, mask)
        for edge_layer in self.edge_layers:
            features = edge_layer(features, xyz, neighbor_indices, neighbor_mask, mask)

        expanded_mask = mask.unsqueeze(-1)
        counts = expanded_mask.sum(dim=1).clamp_min(1)
        masked_mean = features.masked_fill(~expanded_mask, 0.0).sum(dim=1) / counts
        centered = features - masked_mean.unsqueeze(1)
        variance = centered.square().masked_fill(~expanded_mask, 0.0).sum(dim=1) / counts
        masked_std = variance.clamp_min(torch.finfo(variance.dtype).eps).sqrt()
        masked_max = features.masked_fill(~expanded_mask, -torch.inf).amax(dim=1)
        masked_max = torch.where(mask.any(dim=1, keepdim=True), masked_max, torch.zeros_like(masked_max))
        frame_features = self.frame_mlp(torch.cat((masked_max, masked_mean, masked_std), dim=-1))
        return frame_features.reshape(batch_size, sequence_length, -1)
