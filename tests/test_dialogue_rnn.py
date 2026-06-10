"""Dialogue-level PyTorch RNN model integration."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import (
    DialogueRnnConfig,
    DialogueTrainingSettings,
    ExperimentConfig,
    ModalityEncoderSettings,
)
from meld_emotion.fusion.masking import get_scenario, mask_bundle

_ROOT = Path(__file__).resolve().parents[1]


def _run_torch_snippet(code: str) -> None:
    if importlib.util.find_spec("torch") is None:
        return
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_torch_dialogue_components_in_subprocess() -> None:
    _run_torch_snippet(
        """
import numpy as np
import torch
from meld_emotion.models.attentive_rnn_encoder import AttentiveRnnEncoder
from meld_emotion.models.conformer_encoder import ConformerEncoder
from meld_emotion.models.gated_multimodal_fusion import GatedMultimodalFusion
from meld_emotion.models.memory_attention import MemoryAttention
from meld_emotion.models.multimodal_emotion_model import MultimodalEmotionModel
from meld_emotion.models.rope import apply_rope
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier
from meld_emotion.config.schema import DialogueRnnConfig

x = torch.randn(2, 4, 8)
assert apply_rope(x, torch.arange(4)).shape == x.shape
try:
    apply_rope(torch.randn(2, 4, 7), torch.arange(4))
except ValueError:
    pass
else:
    raise AssertionError("odd RoPE dim should fail")

encoder = AttentiveRnnEncoder(input_dim=3, hidden_dim=5, proj_dim=4)
pooled, attn = encoder(torch.zeros(2, 3, 1, 3), torch.zeros(2, 3, 1))
assert pooled.shape == (2, 3, 5)
assert attn.shape == (2, 3, 1)
assert not torch.isnan(pooled).any()
assert torch.allclose(pooled, torch.zeros_like(pooled))

conformer = ConformerEncoder(
    input_dim=3,
    hidden_dim=8,
    num_layers=1,
    num_heads=2,
    conv_kernel_size=3,
    dropout=0.0,
    attention_dropout=0.0,
)
pooled, weights = conformer(torch.randn(2, 3, 4, 3), torch.ones(2, 3, 4))
assert pooled.shape == (2, 3, 8)
assert weights.shape == (2, 3, 4)
empty_pooled, empty_weights = conformer(torch.randn(1, 2, 4, 3), torch.zeros(1, 2, 4))
assert torch.allclose(empty_pooled, torch.zeros_like(empty_pooled))
assert torch.allclose(empty_weights, torch.zeros_like(empty_weights))
try:
    conformer(torch.randn(2, 3, 4, 3), torch.ones(2, 3, 5))
except ValueError:
    pass
else:
    raise AssertionError("mask shape mismatch should fail")

fusion = GatedMultimodalFusion(modality_dim=4, fusion_dim=6)
text = torch.randn(2, 3, 4)
audio = torch.randn(2, 3, 4)
video = torch.randn(2, 3, 4)
mask = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
fused, gate = fusion(text, audio, video, mask)
assert fused.shape == (2, 3, 6)
assert torch.allclose(gate[..., 1], torch.zeros_like(gate[..., 1]))
assert torch.allclose(gate[:, 2], torch.zeros_like(gate[:, 2]))

memory = MemoryAttention(hidden_dim=6, attn_dim=6, max_relative_distance=4)
context = torch.randn(1, 4, 6)
utterance_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
speaker_id = torch.tensor([[1, 1, 2, 0]])
_, attn = memory(context, utterance_mask, speaker_id)
assert torch.allclose(attn[0].triu(1), torch.zeros_like(attn[0].triu(1)))
assert torch.allclose(attn[0, :, 3], torch.zeros_like(attn[0, :, 3]))
assert torch.allclose(attn[0, 3], torch.zeros_like(attn[0, 3]))

model = MultimodalEmotionModel(
    text_input_dim=3,
    audio_input_dim=3,
    video_input_dim=3,
    speaker_vocab_size=3,
    modality_hidden_dim=4,
    modality_proj_dim=4,
    fusion_dim=5,
    context_hidden_dim=6,
    memory_attn_dim=6,
    classifier_hidden_dim=7,
    classifier_head_type="gated_residual",
    context_state_dropout=1.0,
)
args = (
    torch.randn(1, 2, 3, 3),
    torch.randn(1, 2, 3, 3),
    torch.randn(1, 2, 3, 3),
    torch.tensor([[1, 2]]),
    torch.ones(1, 2),
    torch.ones(1, 2, 3),
    torch.ones(1, 2, 3),
    torch.ones(1, 2, 3),
    torch.ones(1, 2, 3),
)
model.train()
train_out = model(*args, return_xai=True)
assert torch.allclose(train_out["context_h"], torch.zeros_like(train_out["context_h"]))
assert torch.allclose(train_out["memory"], torch.zeros_like(train_out["memory"]))
model.eval()
eval_out = model(*args, return_xai=True)
assert eval_out["context_h"].abs().sum() > 0

classifier = TorchDialogueEmotionClassifier(DialogueRnnConfig())
distill = classifier._distillation_loss(
    torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
    torch.tensor([[[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
    torch.ones(1, 1),
    temperature=2.0,
    weight=0.3,
)
assert distill.item() > 0.0
assert classifier._distillation_loss(
    torch.zeros(1, 1, 7),
    torch.ones(1, 1, 7) / 7.0,
    torch.zeros(1, 1),
    temperature=2.0,
    weight=0.3,
).item() == 0.0
"""
    )


def test_dialogue_rnn_pipeline_in_subprocess() -> None:
    _run_torch_snippet(
        """
import tempfile
from pathlib import Path

import torch

from meld_emotion.config.schema import (
    AudioConceptConfig,
    DialogueRnnConfig,
    DialogueTrainingSettings,
    EvaluationConfig,
    ExperimentConfig,
    SyntheticConfig,
    TextConceptConfig,
    VideoConceptConfig,
)
from meld_emotion.core.protocols import Classifier
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier
from meld_emotion.pipeline.builder import build_classifier, build_experiment

classifier = build_classifier(DialogueRnnConfig(), tuple())
assert isinstance(classifier, TorchDialogueEmotionClassifier)
assert isinstance(classifier, Classifier)

checkpoint_dir = tempfile.TemporaryDirectory()
checkpoint_path = Path(checkpoint_dir.name) / "best_model.pt"
config = ExperimentConfig(
    name="dialogue_smoke",
    dataset=SyntheticConfig(n_train=42, n_dev=0, n_test=14, seed=3),
    extractors=(TextConceptConfig(), AudioConceptConfig(), VideoConceptConfig()),
    model=DialogueRnnConfig(
        training=DialogueTrainingSettings(
            max_epochs=2,
            batch_size=2,
            validation_fraction=0.0,
            modality_dropout=0.0,
            seed=0,
            best_checkpoint_path=str(checkpoint_path),
        )
    ),
    evaluation=EvaluationConfig(metrics=("accuracy",), confusion=False, scenarios=("full",)),
    reporters=(),
)
result = build_experiment(config).run()
assert result.evaluation.metric("accuracy") is not None
assert checkpoint_path.exists()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
assert checkpoint["epoch"] >= 1
assert checkpoint["score_name"] == "weighted_f1"
assert checkpoint["score_split"] == "train"
assert "model_state_dict" in checkpoint
assert "false_positive_counts" in checkpoint
assert "gate_stats" in checkpoint
assert checkpoint["config"]["training"]["best_checkpoint_path"] == str(checkpoint_path)

from meld_emotion.core.types import Split
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.features.audio import AudioConceptExtractor
from meld_emotion.features.text import TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

restore_source = SyntheticDatasetSource(n_train=42, n_dev=0, n_test=14, seed=3)
restore_samples = list(restore_source.load(Split.TRAIN))
restore_bundle = FeaturePipeline(
    [TextConceptExtractor(), AudioConceptExtractor(), VideoConceptExtractor()]
).fit_transform(restore_samples, Split.TRAIN)
restored = TorchDialogueEmotionClassifier.from_checkpoint(checkpoint_path, device="cpu")
proba = restored.predict_proba(restore_bundle)
assert proba.shape == (restore_bundle.n_samples, 7)
assert (abs(proba.sum(axis=1) - 1.0) < 1e-6).all()
checkpoint_dir.cleanup()
"""
    )


def test_dialogue_rnn_accepts_sequence_bundle_in_subprocess() -> None:
    _run_torch_snippet(
        """
import numpy as np

from meld_emotion.config.schema import (
    ClassifierHeadSettings,
    DialogueContextSettings,
    DialogueRnnConfig,
    DialogueTrainingSettings,
    FusionSettings,
    MemoryAttentionSettings,
    ModalityEncoderSettings,
)
from meld_emotion.core.features import FeatureBundle, FeatureUnit, SequenceFeatureMatrix, UtteranceSpec
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier

def seq(modality, base):
    values = np.full((4, 3, 2), base, dtype=np.float64)
    mask = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool)
    units = tuple(
        tuple(FeatureUnit(f"{modality.value}_{row}_{i}", i) for i in range(int(mask[row].sum())))
        for row in range(4)
    )
    return SequenceFeatureMatrix(
        values=values,
        mask=mask,
        units=units,
        names=("d0", "d1"),
        modality=modality,
        kind=FeatureKind.EMBEDDING,
    )

bundle = FeatureBundle(
    uids=("d0/u0", "d0/u1", "d1/u0", "d1/u1"),
    matrices=(),
    sequence_matrices=(
        seq(Modality.TEXT, 1.0),
        seq(Modality.AUDIO, 2.0),
        seq(Modality.VIDEO, 3.0),
    ),
    availability={
        Modality.TEXT: np.ones(4, dtype=bool),
        Modality.AUDIO: np.ones(4, dtype=bool),
        Modality.VIDEO: np.ones(4, dtype=bool),
    },
    utterances=(
        UtteranceSpec("d0/u0", 0, 0, "A"),
        UtteranceSpec("d0/u1", 0, 1, "B"),
        UtteranceSpec("d1/u0", 1, 0, "A"),
        UtteranceSpec("d1/u1", 1, 1, "B"),
    ),
)
config = DialogueRnnConfig(
    modality_encoder=ModalityEncoderSettings(proj_dim=4, hidden_dim=5, dropout=0.0),
    fusion=FusionSettings(fusion_dim=6, dropout=0.0, gate_entropy_weight=0.01),
    dialogue_context=DialogueContextSettings(hidden_dim=7, dropout=0.0),
    memory_attention=MemoryAttentionSettings(attn_dim=7),
    classifier=ClassifierHeadSettings(
        classifier_head_type="gated_residual",
        hidden_dim=8,
        dropout=0.0,
        aux_text_loss_weight=0.1,
        aux_audio_loss_weight=0.1,
        aux_video_loss_weight=0.1,
    ),
    training=DialogueTrainingSettings(
        max_epochs=1,
        batch_size=2,
        validation_fraction=0.0,
        modality_dropout=0.0,
        text_dropout=0.1,
        seed=0,
    ),
)
classifier = TorchDialogueEmotionClassifier(config)
classifier.fit(bundle, np.array([0, 1, 2, 3], dtype=np.int64))
assert classifier.last_gate_stats["gate_entropy_mean"] >= 0.0
arrays = classifier.xai_arrays(bundle)
assert arrays.text_x.shape == (2, 2, 3, 2)
assert arrays.audio_x.shape == (2, 2, 3, 2)
assert arrays.video_x.shape == (2, 2, 3, 2)
proba = classifier.predict_proba(bundle)
assert proba.shape == (4, 7)
batch = classifier.xai_tensor_batch(arrays, (0,))
assert str(batch["text_x"].dtype) == "torch.float32"
assert str(batch["speaker_id"].dtype) == "torch.int64"
output = classifier.xai_model()(
    batch["text_x"],
    batch["audio_x"],
    batch["video_x"],
    batch["speaker_id"],
    batch["utterance_mask"],
    batch["text_mask"],
    batch["audio_mask"],
    batch["video_mask"],
    batch["modality_mask"],
)
assert output["aux_text_logits"].shape[-1] == 7
assert output["aux_audio_logits"].shape[-1] == 7
assert output["aux_video_logits"].shape[-1] == 7
assert output["alpha_context"].min() >= 0.0
assert output["alpha_context"].max() <= 1.0
assert output["alpha_memory"].min() >= 0.0
assert output["alpha_memory"].max() <= 1.0
"""
    )


def test_dialogue_rnn_config_roundtrip() -> None:
    config = ExperimentConfig(
        name="dialogue",
        model=DialogueRnnConfig(
            modality_encoder=ModalityEncoderSettings(
                encoder_type="conformer",
                sequence_fallback_policy="error",
                text_input_dim=768,
                audio_input_dim=1024,
                video_input_dim=768,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                conv_kernel_size=15,
                ffn_multiplier=4.0,
                attention_dropout=0.1,
                pooling_type="attentive",
            ),
            training=DialogueTrainingSettings(
                max_epochs=2,
                validation_fraction=0.0,
                best_checkpoint_path="outputs/best_model.pt",
            )
        ),
    )
    assert from_dict(to_dict(config)) == config


def test_mask_bundle_preserves_utterance_metadata(train_bundle) -> None:
    masked = mask_bundle(train_bundle, get_scenario("no_audio"))
    assert masked.utterances == train_bundle.utterances


def test_dialogue_rnn_requires_sequence_when_policy_is_error(train_bundle) -> None:
    if importlib.util.find_spec("torch") is None:
        return
    from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier

    config = DialogueRnnConfig(
        modality_encoder=ModalityEncoderSettings(sequence_fallback_policy="error")
    )
    classifier = TorchDialogueEmotionClassifier(config)
    try:
        classifier.fit(train_bundle, np.zeros(train_bundle.n_samples, dtype=np.int64))
    except ValueError as exc:
        assert "sequence feature is required" in str(exc)
    else:
        raise AssertionError("sequence_fallback_policy='error' should reject pooled bundles")
