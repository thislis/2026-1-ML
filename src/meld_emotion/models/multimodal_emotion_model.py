"""PyTorch multimodal dialogue emotion model."""

from __future__ import annotations

import torch
from torch import nn

from meld_emotion.models.attentive_rnn_encoder import AttentiveRnnEncoder
from meld_emotion.models.classifier import EmotionClassifierHead
from meld_emotion.models.conformer_encoder import ConformerEncoder
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
        modality_encoder_type: str = "rnn",
        modality_proj_dim: int = 128,
        modality_hidden_dim: int = 128,
        modality_num_layers: int = 1,
        modality_num_heads: int = 4,
        modality_conv_kernel_size: int = 15,
        modality_ffn_multiplier: float = 4.0,
        modality_dropout: float = 0.2,
        modality_attention_dropout: float = 0.1,
        modality_pooling_type: str = "attentive",
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
        classifier_head_type: str = "concat",
        classifier_use_context: bool = True,
        classifier_use_memory: bool = True,
        classifier_gate_hidden_dim: int = 128,
        classifier_gate_dropout: float = 0.1,
        context_state_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= context_state_dropout <= 1.0:
            raise ValueError("context_state_dropout must be in [0, 1]")
        self.context_state_dropout = context_state_dropout
        self.text_encoder = _build_modality_encoder(
            modality_encoder_type,
            text_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            num_layers=modality_num_layers,
            num_heads=modality_num_heads,
            conv_kernel_size=modality_conv_kernel_size,
            ffn_multiplier=modality_ffn_multiplier,
            dropout=modality_dropout,
            attention_dropout=modality_attention_dropout,
            pooling_type=modality_pooling_type,
        )
        self.audio_encoder = _build_modality_encoder(
            modality_encoder_type,
            audio_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            num_layers=modality_num_layers,
            num_heads=modality_num_heads,
            conv_kernel_size=modality_conv_kernel_size,
            ffn_multiplier=modality_ffn_multiplier,
            dropout=modality_dropout,
            attention_dropout=modality_attention_dropout,
            pooling_type=modality_pooling_type,
        )
        self.video_encoder = _build_modality_encoder(
            modality_encoder_type,
            video_input_dim,
            proj_dim=modality_proj_dim,
            hidden_dim=modality_hidden_dim,
            rnn_type=rnn_type,
            num_layers=modality_num_layers,
            num_heads=modality_num_heads,
            conv_kernel_size=modality_conv_kernel_size,
            ffn_multiplier=modality_ffn_multiplier,
            dropout=modality_dropout,
            attention_dropout=modality_attention_dropout,
            pooling_type=modality_pooling_type,
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
            classifier_head_type=classifier_head_type,
            use_context=classifier_use_context,
            use_memory=classifier_use_memory,
            gate_hidden_dim=classifier_gate_hidden_dim,
            gate_dropout=classifier_gate_dropout,
        )
        self.text_aux_head = nn.Linear(modality_hidden_dim, num_classes)
        self.audio_aux_head = nn.Linear(modality_hidden_dim, num_classes)
        self.video_aux_head = nn.Linear(modality_hidden_dim, num_classes)

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
        context_h, memory = self._apply_context_dropout(context_h, memory, utterance_mask)
        classifier_fused = torch.zeros_like(fused) if ablate_classifier_block == "fused" else fused
        classifier_context = (
            torch.zeros_like(context_h) if ablate_classifier_block == "context" else context_h
        )
        classifier_memory = (
            torch.zeros_like(memory) if ablate_classifier_block == "memory" else memory
        )
        logits = self.classifier(classifier_fused, classifier_context, classifier_memory)
        logits = logits * utterance_mask.to(dtype=logits.dtype).unsqueeze(-1)

        output = {
            "logits": logits,
            "modality_gate": gate,
            "text_attention": text_attn,
            "audio_attention": audio_attn,
            "video_attention": video_attn,
            "memory_attention": memory_attn,
            "aux_text_logits": self.text_aux_head(u_t)
            * utterance_mask.to(dtype=u_t.dtype).unsqueeze(-1),
            "aux_audio_logits": self.audio_aux_head(u_a)
            * utterance_mask.to(dtype=u_a.dtype).unsqueeze(-1),
            "aux_video_logits": self.video_aux_head(u_v)
            * utterance_mask.to(dtype=u_v.dtype).unsqueeze(-1),
        }
        if self.classifier.last_alpha_context is not None:
            output["alpha_context"] = self.classifier.last_alpha_context
        if self.classifier.last_alpha_memory is not None:
            output["alpha_memory"] = self.classifier.last_alpha_memory
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

    def _apply_context_dropout(
        self,
        context_h: torch.Tensor,
        memory: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.context_state_dropout <= 0.0:
            return context_h, memory
        valid = utterance_mask > 0.0
        drop = (torch.rand_like(utterance_mask) < self.context_state_dropout) & valid
        if not bool(drop.any()):
            return context_h, memory
        keep = (~drop).to(dtype=context_h.dtype).unsqueeze(-1)
        return context_h * keep, memory * keep


def _build_modality_encoder(
    encoder_type: str,
    input_dim: int,
    *,
    proj_dim: int,
    hidden_dim: int,
    rnn_type: str,
    num_layers: int,
    num_heads: int,
    conv_kernel_size: int,
    ffn_multiplier: float,
    dropout: float,
    attention_dropout: float,
    pooling_type: str,
) -> nn.Module:
    name = encoder_type.lower()
    if name == "rnn":
        return AttentiveRnnEncoder(
            input_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            dropout=dropout,
        )
    if name == "conformer":
        return ConformerEncoder(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            attention_dropout=attention_dropout,
            pooling_type=pooling_type,
        )
    raise ValueError("modality_encoder.encoder_type must be 'rnn' or 'conformer'")
