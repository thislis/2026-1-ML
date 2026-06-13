"""Dialogue RNN single-stream multimodal mode tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

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


def test_dialogue_rnn_accepts_multimodal_bundle_in_subprocess() -> None:
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
from meld_emotion.core.features import FeatureBundle, FeatureMatrix, UtteranceSpec
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier

bundle = FeatureBundle(
    uids=("d0/u0", "d0/u1", "d1/u0", "d1/u1"),
    matrices=(
        FeatureMatrix(
            values=np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0,
            names=("m0", "m1", "m2", "m3"),
            modality=Modality.MULTIMODAL,
            kind=FeatureKind.EMBEDDING,
        ),
    ),
    availability={Modality.MULTIMODAL: np.ones(4, dtype=bool)},
    utterances=(
        UtteranceSpec("d0/u0", 0, 0, "A"),
        UtteranceSpec("d0/u1", 0, 1, "B"),
        UtteranceSpec("d1/u0", 1, 0, "A"),
        UtteranceSpec("d1/u1", 1, 1, "B"),
    ),
)
config = DialogueRnnConfig(
    input_mode="multimodal",
    modality_encoder=ModalityEncoderSettings(
        encoder_type="rnn",
        text_input_dim=4,
        proj_dim=4,
        hidden_dim=5,
        dropout=0.0,
    ),
    fusion=FusionSettings(fusion_dim=6, dropout=0.0),
    dialogue_context=DialogueContextSettings(hidden_dim=7, dropout=0.0),
    memory_attention=MemoryAttentionSettings(attn_dim=7),
    classifier=ClassifierHeadSettings(hidden_dim=8, dropout=0.0),
    training=DialogueTrainingSettings(
        max_epochs=1,
        batch_size=2,
        validation_fraction=0.0,
        modality_dropout=0.0,
        seed=0,
    ),
)
classifier = TorchDialogueEmotionClassifier(config)
classifier.fit(bundle, np.array([0, 1, 2, 3], dtype=np.int64))
arrays = classifier.xai_arrays(bundle)
assert arrays.text_x.shape == (2, 2, 1, 4)
assert arrays.modality_mask.shape == (2, 2, 1)
proba = classifier.predict_proba(bundle)
assert proba.shape == (4, 7)
assert np.allclose(proba.sum(axis=1), 1.0)
assert "gate_multimodal_mean" in classifier.last_gate_stats
"""
    )
