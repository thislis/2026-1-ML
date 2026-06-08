"""Dialogue-level PyTorch RNN model integration."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import (
    DialogueRnnConfig,
    DialogueTrainingSettings,
    ExperimentConfig,
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
from meld_emotion.models.gated_multimodal_fusion import GatedMultimodalFusion
from meld_emotion.models.memory_attention import MemoryAttention
from meld_emotion.models.rope import apply_rope

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


def test_dialogue_rnn_config_roundtrip() -> None:
    config = ExperimentConfig(
        name="dialogue",
        model=DialogueRnnConfig(
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
