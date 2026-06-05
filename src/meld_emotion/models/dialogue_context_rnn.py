"""Dialogue-level unidirectional RNN context encoder."""

from __future__ import annotations

import torch
from torch import nn


class DialogueContextRnn(nn.Module):
    """Encode a sequence of fused utterance vectors with speaker embeddings."""

    def __init__(
        self,
        fusion_dim: int = 256,
        speaker_vocab_size: int = 1,
        speaker_emb_dim: int = 32,
        hidden_dim: int = 256,
        rnn_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.speaker_embedding = nn.Embedding(max(speaker_vocab_size, 1), speaker_emb_dim)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        input_dim = fusion_dim + speaker_emb_dim
        rnn_name = rnn_type.lower()
        if rnn_name == "gru":
            self.rnn: nn.Module = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif rnn_name == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

    def forward(
        self,
        fused: torch.Tensor,
        speaker_id: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> torch.Tensor:
        speaker_emb = self.speaker_embedding(speaker_id.clamp_min(0))
        ctx_input = torch.cat([fused, speaker_emb], dim=-1)
        lengths = utterance_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            ctx_input,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=fused.shape[1],
        )
        return out * utterance_mask.to(dtype=out.dtype).unsqueeze(-1)
