"""Modality-level Conformer encoder with safe masked pooling."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from meld_emotion.core.status import real


@real
class ConformerEncoder(nn.Module):
    """Encode one modality sequence into an utterance embedding.

    The interface mirrors ``AttentiveRnnEncoder``: input ``[B,N,L,D]`` plus a
    ``[B,N,L]`` mask, output pooled embeddings ``[B,N,H]`` and pooling weights
    ``[B,N,L]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        pooling_type: str = "attentive",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if pooling_type not in {"attentive", "mean"}:
            raise ValueError("pooling_type must be 'attentive' or 'mean'")

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _ConformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pooling_type = pooling_type
        self.pool_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,N,L,D], got {tuple(x.shape)}")
        if mask.shape != x.shape[:3]:
            raise ValueError(
                f"mask shape must be [B,N,L] matching x[:3], got {tuple(mask.shape)}"
            )

        bsz, n_utt, seq_len, feat_dim = x.shape
        flat_x = x.reshape(bsz * n_utt, seq_len, feat_dim)
        flat_mask = mask.reshape(bsz * n_utt, seq_len).to(dtype=torch.bool, device=x.device)

        safe_mask = flat_mask
        all_missing = ~safe_mask.any(dim=1)
        if bool(all_missing.any()):
            safe_mask = safe_mask.clone()
            safe_mask[all_missing, 0] = True

        hidden = self.dropout(self.input_norm(self.input_projection(flat_x)))
        hidden = hidden.masked_fill(~safe_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, safe_mask)
            hidden = hidden.masked_fill(~safe_mask.unsqueeze(-1), 0.0)

        weights = self._pooling_weights(hidden, safe_mask)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        pooled = pooled.masked_fill(all_missing.unsqueeze(-1), 0.0)
        weights = weights.masked_fill(all_missing.unsqueeze(-1), 0.0)
        return pooled.reshape(bsz, n_utt, -1), weights.reshape(bsz, n_utt, seq_len)

    def _pooling_weights(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            weights = mask.to(dtype=hidden.dtype)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        scores = self.pool_score(torch.tanh(hidden)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


class _ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        conv_kernel_size: int,
        ffn_multiplier: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        ffn_dim = max(hidden_dim, round(hidden_dim * ffn_multiplier))
        self.ffn1 = _FeedForward(hidden_dim, ffn_dim, dropout)
        self.ffn2 = _FeedForward(hidden_dim, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = _DepthwiseTemporalConv(hidden_dim, conv_kernel_size, dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + 0.5 * self.ffn1(hidden)
        attn_input = self.attn_norm(hidden)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=~mask,
            need_weights=False,
        )
        hidden = hidden + self.attn_dropout(attn_out)
        hidden = hidden + self.conv(hidden, mask)
        hidden = hidden + 0.5 * self.ffn2(hidden)
        return cast(torch.Tensor, self.final_norm(hidden))


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(hidden))


class _DepthwiseTemporalConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden).masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        if x.shape[-1] != hidden.shape[1]:
            x = x[..., : hidden.shape[1]]
        x = self.activation(x)
        x = self.pointwise(x).transpose(1, 2)
        return cast(torch.Tensor, self.dropout(x).masked_fill(~mask.unsqueeze(-1), 0.0))
