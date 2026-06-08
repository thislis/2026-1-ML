"""Fine-grained dialogue XAI explainer smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.config.schema import (
    ClassifierHeadSettings,
    DialogueContextSettings,
    DialogueRnnConfig,
    DialogueTrainingSettings,
    FusionSettings,
    MemoryAttentionSettings,
    ModalityEncoderSettings,
)
from meld_emotion.core.features import (
    FeatureBundle,
    FeatureUnit,
    SequenceFeatureMatrix,
    UtteranceSpec,
)
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier


def _seq(modality: Modality, base: float) -> SequenceFeatureMatrix:
    values = np.full((3, 2, 2), base, dtype=np.float64)
    mask = np.ones((3, 2), dtype=bool)
    units = tuple(
        (
            FeatureUnit(f"{modality.value}_{row}_0", 0),
            FeatureUnit(f"{modality.value}_{row}_1", 1),
        )
        for row in range(3)
    )
    return SequenceFeatureMatrix(
        values=values,
        mask=mask,
        units=units,
        names=("d0", "d1"),
        modality=modality,
        kind=FeatureKind.EMBEDDING,
    )


def _bundle() -> FeatureBundle:
    return FeatureBundle(
        uids=("d0/u0", "d0/u1", "d0/u2"),
        matrices=(),
        sequence_matrices=(
            _seq(Modality.TEXT, 1.0),
            _seq(Modality.AUDIO, 2.0),
            _seq(Modality.VIDEO, 3.0),
        ),
        availability={
            Modality.TEXT: np.ones(3, dtype=bool),
            Modality.AUDIO: np.ones(3, dtype=bool),
            Modality.VIDEO: np.ones(3, dtype=bool),
        },
        utterances=(
            UtteranceSpec("d0/u0", 0, 0, "A"),
            UtteranceSpec("d0/u1", 0, 1, "B"),
            UtteranceSpec("d0/u2", 0, 2, "A"),
        ),
    )


def test_dialogue_finegrained_xai_outputs_serializable_result() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("captum")
    bundle = _bundle()
    config = DialogueRnnConfig(
        modality_encoder=ModalityEncoderSettings(proj_dim=4, hidden_dim=5, dropout=0.0),
        fusion=FusionSettings(fusion_dim=6, dropout=0.0),
        dialogue_context=DialogueContextSettings(hidden_dim=7, dropout=0.0),
        memory_attention=MemoryAttentionSettings(attn_dim=7),
        classifier=ClassifierHeadSettings(hidden_dim=8, dropout=0.0),
        training=DialogueTrainingSettings(
            max_epochs=1,
            batch_size=1,
            validation_fraction=0.0,
            modality_dropout=0.0,
            seed=0,
        ),
    )
    classifier = TorchDialogueEmotionClassifier(config)
    y = np.array([0, 1, 2], dtype=np.int64)
    classifier.fit(bundle, y)
    report = DialogueFineGrainedXaiExplainer(n_steps=2, top_k=2, max_targets=1).explain(
        classifier, bundle, y
    )
    assert len(report.dialogue_xai) == 1
    item = report.dialogue_xai[0]
    assert item.modality
    assert item.utterances
    assert set(item.classifier_blocks) == {"fused", "context", "memory"}
    assert item.text_dimension_attribution or item.audio_dimension_attribution
