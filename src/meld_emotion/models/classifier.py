"""Utterance-level classifier head for the dialogue RNN model."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class EmotionClassifierHead(nn.Module):
    """Classify each utterance from fused, context, and memory vectors."""

    def __init__(
        self,
        fusion_dim: int = 256,
        context_dim: int = 256,
        memory_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = fusion_dim + context_dim + memory_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        fused: torch.Tensor,
        context_h: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.net(torch.cat([fused, context_h, memory], dim=-1)))
