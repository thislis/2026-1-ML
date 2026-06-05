"""Modality-level RNN encoder with safe masked attention pooling."""

from __future__ import annotations

import torch
from torch import nn


class AttentiveRnnEncoder(nn.Module):
    """Encode one modality sequence into a fixed-size utterance embedding."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        rnn_type: str = "gru",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.proj = nn.Linear(input_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)
        rnn_name = rnn_type.lower()
        if rnn_name == "gru":
            self.rnn: nn.Module = nn.GRU(proj_dim, hidden_dim, batch_first=True)
        elif rnn_name == "lstm":
            self.rnn = nn.LSTM(proj_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled embeddings ``[B,N,H]`` and attention ``[B,N,L]``."""

        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,N,L,D], got {tuple(x.shape)}")
        if mask.shape != x.shape[:3]:
            raise ValueError(f"mask shape must be [B,N,L], got {tuple(mask.shape)}")

        bsz, n_utt, seq_len, feat_dim = x.shape
        flat_x = x.reshape(bsz * n_utt, seq_len, feat_dim)
        flat_mask = mask.reshape(bsz * n_utt, seq_len).to(dtype=torch.bool, device=x.device)

        projected = self.dropout(self.norm(self.proj(flat_x)))
        hidden, _ = self.rnn(projected)
        scores = self.attn_score(torch.tanh(self.attn(hidden))).squeeze(-1)

        safe_mask = flat_mask
        all_missing = ~safe_mask.any(dim=1)
        if bool(all_missing.any()):
            safe_mask = safe_mask.clone()
            safe_mask[all_missing, 0] = True

        scores = scores.masked_fill(~safe_mask, -1.0e9)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~safe_mask, 0.0)
        denom = alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        alpha = alpha / denom
        pooled = torch.sum(hidden * alpha.unsqueeze(-1), dim=1)
        pooled = pooled.masked_fill(all_missing.unsqueeze(-1), 0.0)
        return pooled.reshape(bsz, n_utt, -1), alpha.reshape(bsz, n_utt, seq_len)
