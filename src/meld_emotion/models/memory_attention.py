"""Causal memory attention over dialogue context states."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from meld_emotion.models.rope import apply_rope


class MemoryAttention(nn.Module):
    """Let each utterance attend to itself and previous utterances."""

    def __init__(
        self,
        hidden_dim: int = 256,
        attn_dim: int = 256,
        enabled: bool = True,
        use_rope: bool = False,
        use_relative_distance_bias: bool = True,
        use_same_speaker_bias: bool = True,
        max_relative_distance: int = 32,
    ) -> None:
        super().__init__()
        if use_rope and attn_dim % 2 != 0:
            raise ValueError("RoPE requires an even attn_dim")
        self.enabled = enabled
        self.use_rope = use_rope
        self.use_relative_distance_bias = use_relative_distance_bias
        self.use_same_speaker_bias = use_same_speaker_bias
        self.max_relative_distance = max_relative_distance
        self.q = nn.Linear(hidden_dim, attn_dim)
        self.k = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.distance_bias = nn.Embedding(max_relative_distance + 1, 1)
        self.same_speaker_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        context_h: torch.Tensor,
        utterance_mask: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            n_utt = context_h.shape[1]
            eye = torch.eye(n_utt, device=context_h.device, dtype=context_h.dtype)
            attn = eye.unsqueeze(0).expand(context_h.shape[0], -1, -1)
            return context_h, attn

        q = self.q(context_h)
        k = self.k(context_h)
        v = self.v(context_h)
        n_utt = context_h.shape[1]
        if self.use_rope:
            positions = torch.arange(n_utt, device=context_h.device)
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(q.shape[-1]))
        if self.use_relative_distance_bias:
            score = score + self._distance_bias(n_utt, context_h.device).unsqueeze(0)
        if self.use_same_speaker_bias:
            same = speaker_id.unsqueeze(2) == speaker_id.unsqueeze(1)
            score = score + same.to(dtype=score.dtype) * self.same_speaker_bias

        causal = torch.tril(torch.ones(n_utt, n_utt, device=context_h.device, dtype=torch.bool))
        key_mask = utterance_mask.to(dtype=torch.bool).unsqueeze(1)
        score = score.masked_fill(~causal.unsqueeze(0), -1.0e9)
        score = score.masked_fill(~key_mask, -1.0e9)
        attn = torch.softmax(score, dim=-1)
        attn = attn * utterance_mask.to(dtype=attn.dtype).unsqueeze(-1)
        memory = torch.matmul(attn, v)
        memory = memory * utterance_mask.to(dtype=memory.dtype).unsqueeze(-1)
        return memory, attn

    def _distance_bias(self, n_utt: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n_utt, device=device)
        distance = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(
            min=0,
            max=self.max_relative_distance,
        )
        return cast(torch.Tensor, self.distance_bias(distance).squeeze(-1))
