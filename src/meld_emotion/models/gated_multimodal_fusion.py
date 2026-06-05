"""Gated multimodal fusion for text/audio/video utterance embeddings."""

from __future__ import annotations

import torch
from torch import nn


class GatedMultimodalFusion(nn.Module):
    """Fuse modality embeddings with missing-modality-aware gates."""

    def __init__(
        self,
        modality_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        use_gated_fusion: bool = True,
        use_interaction_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        self.use_interaction_features = use_interaction_features
        self.text_proj = nn.Linear(modality_dim, modality_dim)
        self.audio_proj = nn.Linear(modality_dim, modality_dim)
        self.video_proj = nn.Linear(modality_dim, modality_dim)
        self.gate = nn.Linear(modality_dim * 3 + 3, 3)
        n_parts = 4 if not use_interaction_features else 7
        self.output = nn.Sequential(
            nn.LayerNorm(modality_dim * n_parts),
            nn.Linear(modality_dim * n_parts, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused vector ``[B,N,F]`` and modality gates ``[B,N,3]``."""

        mask = modality_mask.to(dtype=text.dtype, device=text.device)
        text = text * mask[..., 0:1]
        audio = audio * mask[..., 1:2]
        video = video * mask[..., 2:3]

        combined = torch.cat([text, audio, video, mask], dim=-1)
        logits = self.gate(combined)
        valid = mask > 0.0
        safe_valid = valid
        all_missing = ~safe_valid.any(dim=-1)
        if bool(all_missing.any()):
            safe_valid = safe_valid.clone()
            safe_valid[..., 0] = safe_valid[..., 0] | all_missing

        if self.use_gated_fusion:
            raw_gate = torch.softmax(logits.masked_fill(~safe_valid, -1.0e9), dim=-1)
            gate = raw_gate * mask
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        else:
            gate = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        gated_sum = (
            gate[..., 0:1] * self.text_proj(text)
            + gate[..., 1:2] * self.audio_proj(audio)
            + gate[..., 2:3] * self.video_proj(video)
        )

        parts = [text, audio, video]
        if self.use_interaction_features:
            parts.extend([text * audio, text * video, audio * video])
        parts.append(gated_sum)
        return self.output(torch.cat(parts, dim=-1)), gate
