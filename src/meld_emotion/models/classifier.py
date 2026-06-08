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
        classifier_head_type: str = "concat",
        use_context: bool = True,
        use_memory: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if classifier_head_type not in {"concat", "gated_residual"}:
            raise ValueError("classifier_head_type must be 'concat' or 'gated_residual'")
        self.classifier_head_type = classifier_head_type
        self.use_context = use_context
        self.use_memory = use_memory
        self.last_alpha_context: torch.Tensor | None = None
        self.last_alpha_memory: torch.Tensor | None = None
        input_dim = fusion_dim + context_dim + memory_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.utterance_head = _head(fusion_dim, hidden_dim, num_classes, dropout)
        self.context_head = _head(context_dim, hidden_dim, num_classes, dropout)
        self.memory_head = _head(memory_dim, hidden_dim, num_classes, dropout)
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fused: torch.Tensor,
        context_h: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([fused, context_h, memory], dim=-1)
        if self.classifier_head_type == "concat":
            self.last_alpha_context = None
            self.last_alpha_memory = None
            return cast(torch.Tensor, self.net(inputs))

        utterance_logits = self.utterance_head(fused)
        context_logits = self.context_head(context_h)
        memory_logits = self.memory_head(memory)
        alpha = self.residual_gate(inputs)
        alpha_context = alpha[..., 0:1] if self.use_context else torch.zeros_like(alpha[..., 0:1])
        alpha_memory = alpha[..., 1:2] if self.use_memory else torch.zeros_like(alpha[..., 1:2])
        self.last_alpha_context = alpha_context
        self.last_alpha_memory = alpha_memory
        return cast(
            torch.Tensor,
            utterance_logits + alpha_context * context_logits + alpha_memory * memory_logits,
        )


def _head(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )
