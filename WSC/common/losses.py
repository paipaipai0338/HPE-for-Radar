"""Loss functions shared by behavior-recognition routes."""

from __future__ import annotations

import torch
from torch import nn


class MulticlassFocalLoss(nn.Module):
    """Weighted focal loss for mutually exclusive classes."""

    def __init__(self, gamma: float, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probability = logits.log_softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        losses = -(1.0 - log_probability.exp()).pow(self.gamma) * log_probability
        if self.weight is None:
            return losses.mean()
        sample_weights = self.weight[labels]
        return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(
            torch.finfo(losses.dtype).eps
        )


def sequence_supervision_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    auxiliary_weight: float,
) -> torch.Tensor:
    """Combine final-frame classification with supervision on earlier frames."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels must have shapes [B, T, C] and [B, T]")
    if auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be non-negative")

    final_loss = criterion(logits[:, -1], labels[:, -1])
    if logits.shape[1] == 1 or auxiliary_weight == 0:
        return final_loss
    auxiliary_loss = criterion(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, :-1].reshape(-1),
    )
    return (final_loss + auxiliary_weight * auxiliary_loss) / (1.0 + auxiliary_weight)
