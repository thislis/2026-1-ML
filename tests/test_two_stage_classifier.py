"""Two-stage Neutral/Non-Neutral classifier wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.config.loader import load_config, to_dict
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    NearestCentroidConfig,
    TwoStageConfig,
)
from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.types import (
    EMOTION_ORDER,
    Emotion,
    FeatureKind,
    FloatArray,
    IntArray,
    Modality,
)
from meld_emotion.models.two_stage import TwoStageEmotionClassifier
from meld_emotion.pipeline.builder import build_classifier, build_experiment


class _FixedClassifier:
    def __init__(self, proba: FloatArray) -> None:
        self._proba = proba

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return EMOTION_ORDER

    def fit(self, bundle: FeatureBundle, y: IntArray) -> _FixedClassifier:
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._proba[: bundle.n_samples]

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        return PredictionSet(
            uids=bundle.uids,
            y_pred=np.argmax(proba, axis=1).astype(np.int64),
            proba=proba,
            classes=self.classes,
        )


def _bundle(n: int) -> FeatureBundle:
    return FeatureBundle(
        uids=tuple(f"u{i}" for i in range(n)),
        matrices=(
            FeatureMatrix(
                values=np.ones((n, 2), dtype=np.float64),
                names=("a", "b"),
                modality=Modality.TEXT,
                kind=FeatureKind.EMBEDDING,
            ),
        ),
    )


def test_two_stage_threshold_controls_final_prediction() -> None:
    proba = np.asarray(
        [
            [0.42, 0.40, 0.04, 0.04, 0.04, 0.03, 0.03],
            [0.70, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03],
        ],
        dtype=np.float64,
    )
    clf = TwoStageEmotionClassifier(_FixedClassifier(proba), neutral_threshold=0.6)
    prediction = clf.predict(_bundle(2))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.NEUTRAL
    assert prediction.classes[prediction.y_pred[1]] == Emotion.NEUTRAL
    decisions = clf.last_two_stage_decisions
    assert decisions[0].stage1_label == "neutral"
    assert decisions[0].stage2_label == Emotion.JOY
    assert decisions[0].non_neutral_probability == pytest.approx(0.58)


def test_two_stage_non_neutral_uses_model2_top_emotion() -> None:
    proba = np.asarray([[0.20, 0.25, 0.10, 0.30, 0.05, 0.05, 0.05]], dtype=np.float64)
    clf = TwoStageEmotionClassifier(_FixedClassifier(proba), neutral_threshold=0.5)
    prediction = clf.predict(_bundle(1))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.ANGER
    decision = clf.last_two_stage_decisions[0]
    assert decision.stage1_label == "non_neutral"
    assert decision.stage2_label == Emotion.ANGER
    assert "Model 1" in decision.rationale


def test_two_stage_config_loads_and_builds() -> None:
    config = load_config("configs/default.yaml")
    assert isinstance(config.model, TwoStageConfig)
    assert isinstance(config.model.base, EarlyFusionConfig)
    assert to_dict(config)["model"]["type"] == "two_stage"

    classifier = build_classifier(
        TwoStageConfig(
            base=EarlyFusionConfig(base=NearestCentroidConfig()),
            neutral_threshold=0.5,
        ),
        EMOTION_ORDER,
    )
    assert isinstance(classifier, TwoStageEmotionClassifier)


def test_two_stage_end_to_end_smoke() -> None:
    result = build_experiment(load_config("configs/default.yaml")).run()
    assert result.evaluation.metric("accuracy") is not None
