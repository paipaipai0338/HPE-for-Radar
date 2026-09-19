from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class CausalTemporalBlock(nn.Module):
    """Residual dilated convolution with left-only temporal padding."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        features = functional.pad(sequence.transpose(1, 2), (self.left_padding, 0))
        features = self.convolution(features).transpose(1, 2)
        features = self.dropout(self.activation(self.norm(features)))
        return sequence + features


class CausalTemporalEncoder(nn.Module):
    """Stack residual causal convolutions with exponentially growing dilation."""

    def __init__(self, channels: int, layer_count: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if min(channels, layer_count, kernel_size) <= 0:
            raise ValueError("channels, layer_count, and kernel_size must be positive")
        self.dilations = tuple(2**index for index in range(layer_count))
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.blocks = nn.ModuleList(
            CausalTemporalBlock(channels, kernel_size, dilation, dropout)
            for dilation in self.dilations
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return sequence
