from __future__ import annotations

import torch
import torch.nn.functional as F


def affinity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    context_mask: torch.Tensor,
    chunk_mask: torch.Tensor,
    *,
    negative_weight: float = 5.0,
) -> torch.Tensor:
    valid = context_mask.bool().unsqueeze(2) & chunk_mask.bool().unsqueeze(1)
    positive = valid & (targets == 1)
    negative = valid & (targets == 0)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_loss = losses[positive].mean() if positive.any() else logits.sum() * 0
    negative_loss = losses[negative].mean() if negative.any() else logits.sum() * 0
    return positive_loss + negative_weight * negative_loss
