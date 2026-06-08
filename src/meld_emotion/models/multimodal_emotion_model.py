"""PyTorch multimodal dialogue emotion model."""

from __future__ import annotations

import torch
from torch import nn

from meld_emotion.models.attentive_rnn_encoder import AttentiveRnnEncoder
from meld_emotion.models.classifier import EmotionClassifierHead
from meld_emotion.models.dialogue_context_rnn import DialogueContextRnn
from meld_emotion.models.gated_multimodal_fusion import GatedMultimodalFusion
from meld_emotion.models.memory_attention import MemoryAttention


class MultimodalEmotionModel(nn.Module):
    """LSTM/GRU + gated fusion + memory attention classifier."""

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        speaker_vocab_size: int,
        num_classes: int = 7,
        rnn_type: str = "gru",
        modality_proj_dim: int = 128,
        modality_hidden_dim: int = 128,
        modality_dropout: float = 0.2,
        fusion_dim: int = 256,
        fusion_dropout: float = 0.3,
        use_gated_fusion: bool = True,
        use_interaction_features: bool = True,
        speaker_emb_dim: int = 32,
        context_hidden_dim: int = 256,
        context_num_layers: int = 1,
        context_dropout: float = 0.3,
        memory_enabled: bool = True,
        memory_attn_dim: int = 256,
        use_rope: bool = False,
        use_relative_distance_bias: bool = True,
        use_same_speaker_bias: bool = True,
        max_relative_distance: int = 32,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.text_encoder = AttentiveRnnEncoder(
            text_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            dropout=modality_dropout,
        )
        self.audio_encoder = AttentiveRnnEncoder(
            audio_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            dropout=modality_dropout,
        )
        self.video_encoder = AttentiveRnnEncoder(
            video_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            dropout=modality_dropout,
        )
        self.fusion = GatedMultimodalFusion(
            modality_dim=modality_hidden_dim,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
            use_gated_fusion=use_gated_fusion,
            use_interaction_features=use_interaction_features,
        )
        self.context = DialogueContextRnn(
            fusion_dim=fusion_dim,
            speaker_vocab_size=speaker_vocab_size,
            speaker_emb_dim=speaker_emb_dim,
            hidden_dim=context_hidden_dim,
            rnn_type=rnn_type,
            num_layers=context_num_layers,
            dropout=context_dropout,
        )
        self.memory = MemoryAttention(
            hidden_dim=context_hidden_dim,
            attn_dim=memory_attn_dim,
            enabled=memory_enabled,
            use_rope=use_rope,
            use_relative_distance_bias=use_relative_distance_bias,
            use_same_speaker_bias=use_same_speaker_bias,
            max_relative_distance=max_relative_distance,
        )
        self.classifier = EmotionClassifierHead(
            fusion_dim=fusion_dim,
            context_dim=context_hidden_dim,
            memory_dim=context_hidden_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        text_x: torch.Tensor,
        audio_x: torch.Tensor,
        video_x: torch.Tensor,
        speaker_id: torch.Tensor,
        utterance_mask: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
        modality_mask: torch.Tensor,
        return_xai: bool = False,
        ablate_classifier_block: str | None = None,
    ) -> dict[str, torch.Tensor]:
        u_t, text_attn = self.text_encoder(text_x, text_mask)
        u_a, audio_attn = self.audio_encoder(audio_x, audio_mask)
        u_v, video_attn = self.video_encoder(video_x, video_mask)

        mask = modality_mask.to(dtype=u_t.dtype, device=u_t.device)
        u_t = u_t * mask[..., 0:1]
        u_a = u_a * mask[..., 1:2]
        u_v = u_v * mask[..., 2:3]

        fused, gate = self.fusion(u_t, u_a, u_v, mask)
        context_h = self.context(fused, speaker_id, utterance_mask)
        memory, memory_attn = self.memory(context_h, utterance_mask, speaker_id)
        classifier_fused = torch.zeros_like(fused) if ablate_classifier_block == "fused" else fused
        classifier_context = (
            torch.zeros_like(context_h) if ablate_classifier_block == "context" else context_h
        )
        classifier_memory = torch.zeros_like(memory) if ablate_classifier_block == "memory" else memory
        logits = self.classifier(classifier_fused, classifier_context, classifier_memory)
        logits = logits * utterance_mask.to(dtype=logits.dtype).unsqueeze(-1)

        output = {
            "logits": logits,
            "modality_gate": gate,
            "text_attention": text_attn,
            "audio_attention": audio_attn,
            "video_attention": video_attn,
            "memory_attention": memory_attn,
        }
        if return_xai:
            output.update(
                {
                    "u_text": u_t,
                    "u_audio": u_a,
                    "u_video": u_v,
                    "fused": fused,
                    "context_h": context_h,
                    "memory": memory,
                }
            )
        return output
