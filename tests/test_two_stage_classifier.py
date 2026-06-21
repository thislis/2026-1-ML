"""Two-stage Neutral/Non-Neutral classifier wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.config.loader import from_dict, load_config, to_dict
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    ExperimentConfig,
    NearestCentroidConfig,
    SvmConfig,
    SvmFourStageConfig,
    SvmMarginTwoStageConfig,
    SvmTwoStageConfig,
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
from meld_emotion.models.two_stage import (
    SvmFourStageClassifier,
    SvmMarginTwoStageClassifier,
    SvmTwoStageClassifier,
    TwoStageEmotionClassifier,
)
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


class _FixedEstimator:
    def __init__(self, scores: FloatArray, proba: FloatArray) -> None:
        self._scores = scores
        self._proba = proba

    def fit(self, x: FloatArray, y: IntArray) -> _FixedEstimator:
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return self._proba[: x.shape[0]]

    def predict(self, x: FloatArray) -> IntArray:
        return np.asarray(np.argmax(self.predict_proba(x), axis=1), dtype=np.int64)

    def decision_scores(self, x: FloatArray) -> FloatArray:
        return self._scores[: x.shape[0]]


class _QueuedEstimatorFactory:
    def __init__(self, *probas: FloatArray) -> None:
        self._probas = list(probas)

    def __call__(self, n_classes: int) -> _FixedEstimator:
        proba = self._probas.pop(0)
        assert proba.shape[1] == n_classes
        scores = np.zeros_like(proba)
        return _FixedEstimator(scores, proba)


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


def test_svm_margin_two_stage_confident_stage1_labels_win() -> None:
    scores = np.asarray(
        [
            [3.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.0, 0.5, 0.4, 3.0, 0.2, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    proba = np.asarray(
        [
            [0.80, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02],
            [0.05, 0.10, 0.10, 0.60, 0.05, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    stage2 = _FixedClassifier(
        np.asarray(
            [
                [0.01, 0.70, 0.10, 0.05, 0.05, 0.05, 0.04],
                [0.01, 0.70, 0.10, 0.05, 0.05, 0.05, 0.04],
            ],
            dtype=np.float64,
        )
    )
    clf = SvmMarginTwoStageClassifier(
        lambda n: _FixedEstimator(scores, proba),
        stage2,
        EMOTION_ORDER,
        margin_threshold=0.25,
    )
    prediction = clf.fit(_bundle(2), np.asarray([0, 3], dtype=np.int64)).predict(_bundle(2))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.NEUTRAL
    assert prediction.classes[prediction.y_pred[1]] == Emotion.ANGER
    decisions = clf.last_two_stage_decisions
    assert decisions[0].stage1_label == "neutral"
    assert decisions[0].stage1_model_label == Emotion.NEUTRAL
    assert decisions[0].stage1_margin > 0.25
    assert decisions[0].routed_to_stage2 is False
    assert decisions[1].stage1_label == "non_neutral"
    assert decisions[1].stage1_model_label == Emotion.ANGER
    assert decisions[1].routed_to_stage2 is False


def test_svm_margin_two_stage_low_margin_routes_to_stage2() -> None:
    scores = np.asarray([[0.40, 0.39, 0.10, 0.05, 0.04, 0.03, 0.02]], dtype=np.float64)
    proba = np.asarray([[0.40, 0.35, 0.10, 0.05, 0.04, 0.03, 0.03]], dtype=np.float64)
    stage2 = _FixedClassifier(
        np.asarray([[0.90, 0.05, 0.04, 0.60, 0.10, 0.10, 0.11]], dtype=np.float64)
    )
    clf = SvmMarginTwoStageClassifier(
        lambda n: _FixedEstimator(scores, proba),
        stage2,
        EMOTION_ORDER,
        margin_threshold=0.25,
    )
    prediction = clf.fit(_bundle(1), np.asarray([0], dtype=np.int64)).predict(_bundle(1))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.ANGER
    decision = clf.last_two_stage_decisions[0]
    assert decision.stage1_label == "uncertain"
    assert decision.stage1_model_label == Emotion.NEUTRAL
    assert decision.stage2_label == Emotion.ANGER
    assert decision.routed_to_stage2 is True
    assert decision.stage1_margin < 0.25


def test_svm_margin_two_stage_low_confidence_routes_to_stage2_when_configured() -> None:
    scores = np.asarray([[3.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]], dtype=np.float64)
    proba = np.asarray([[0.30, 0.20, 0.15, 0.14, 0.10, 0.06, 0.05]], dtype=np.float64)
    stage2 = _FixedClassifier(
        np.asarray([[0.90, 0.05, 0.04, 0.60, 0.10, 0.10, 0.11]], dtype=np.float64)
    )
    clf = SvmMarginTwoStageClassifier(
        lambda n: _FixedEstimator(scores, proba),
        stage2,
        EMOTION_ORDER,
        margin_threshold=0.25,
        stage1_confidence_threshold=0.50,
    )
    prediction = clf.fit(_bundle(1), np.asarray([0], dtype=np.int64)).predict(_bundle(1))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.ANGER
    decision = clf.last_two_stage_decisions[0]
    assert decision.stage1_margin > 0.25
    assert decision.stage1_confidence == pytest.approx(0.30)
    assert decision.routed_to_stage2 is True
    assert "confidence" in decision.rationale


def test_svm_margin_two_stage_config_roundtrip_and_builds() -> None:
    config = SvmMarginTwoStageConfig(
        stage1=SvmConfig(C=2.0, kernel="linear"),
        stage2=EarlyFusionConfig(base=NearestCentroidConfig()),
        margin_threshold=0.33,
        stage1_confidence_threshold=0.65,
        stage1_use_concepts=False,
    )
    assert to_dict(config)["type"] == "two_stage_svm_margin"
    assert to_dict(config)["margin_threshold"] == 0.33
    assert to_dict(config)["stage1_confidence_threshold"] == 0.65
    loaded = _svm_margin_dialogue_config()
    assert isinstance(loaded.model, SvmMarginTwoStageConfig)
    assert loaded.model.margin_threshold == 0.25
    assert loaded.model.stage1_confidence_threshold is None

    classifier = build_classifier(config, EMOTION_ORDER)
    assert isinstance(classifier, SvmMarginTwoStageClassifier)


def test_svm_margin_two_stage_dialogue_rnn_smoke() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn 미설치")
    pytest.importorskip("torch", reason="PyTorch 미설치")

    result = build_experiment(_svm_margin_dialogue_config()).run()
    assert result.evaluation.metric("accuracy") is not None


def test_svm_two_stage_hard_routes_and_multiplies_conditional_probabilities() -> None:
    stage1 = np.asarray([[0.80, 0.20], [0.20, 0.80]], dtype=np.float64)
    stage2 = np.asarray(
        [
            [0.60, 0.10, 0.10, 0.10, 0.05, 0.05],
            [0.10, 0.05, 0.70, 0.05, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    clf = SvmTwoStageClassifier(
        _QueuedEstimatorFactory(stage1, stage2),
        EMOTION_ORDER,
    )
    y = np.asarray([0, 1, 3], dtype=np.int64)
    prediction = clf.fit(_bundle(3), y).predict(_bundle(2))

    assert prediction.classes[prediction.y_pred[0]] == Emotion.NEUTRAL
    assert prediction.classes[prediction.y_pred[1]] == Emotion.ANGER
    assert prediction.proba.shape == (2, 7)
    assert prediction.proba.sum(axis=1) == pytest.approx(np.ones(2))
    assert prediction.proba[1, EMOTION_ORDER.index(Emotion.ANGER)] == pytest.approx(0.56)
    assert clf.last_svm_stage_paths[1].stages == {"svm1": "non_neutral", "svm2": "anger"}
    assert "SVM2 chose anger" in clf.last_two_stage_decisions[1].rationale


def test_svm_four_stage_hard_routes_all_branches_and_combines_probabilities() -> None:
    stage1 = np.asarray(
        [[0.90, 0.10], [0.10, 0.90], [0.10, 0.90], [0.10, 0.90]],
        dtype=np.float64,
    )
    stage2 = np.asarray(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.70],
            [0.10, 0.10, 0.10, 0.70],
        ],
        dtype=np.float64,
    )
    stage3 = np.asarray(
        [[0.60, 0.40], [0.60, 0.40], [0.80, 0.20], [0.10, 0.90]],
        dtype=np.float64,
    )
    stage4 = np.asarray(
        [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50], [0.20, 0.80]],
        dtype=np.float64,
    )
    clf = SvmFourStageClassifier(
        _QueuedEstimatorFactory(stage1, stage2, stage3, stage4),
        EMOTION_ORDER,
    )
    y = np.asarray([0, 3, 2, 6, 5], dtype=np.int64)
    prediction = clf.fit(_bundle(5), y).predict(_bundle(4))

    predicted = tuple(prediction.classes[idx] for idx in prediction.y_pred)
    assert predicted == (Emotion.NEUTRAL, Emotion.ANGER, Emotion.SADNESS, Emotion.FEAR)
    assert prediction.proba.shape == (4, 7)
    assert prediction.proba.sum(axis=1) == pytest.approx(np.ones(4))
    assert prediction.proba[3, EMOTION_ORDER.index(Emotion.FEAR)] == pytest.approx(
        0.90 * 0.70 * 0.90 * 0.80
    )
    assert clf.last_svm_stage_paths[3].stages == {
        "svm1": "non_neutral",
        "svm2": "else",
        "svm3": "else",
        "svm4": "fear",
    }
    assert "SVM4 chose fear" in clf.last_two_stage_decisions[3].rationale


def test_svm_four_stage_sparse_stage_fallback_does_not_train_svc() -> None:
    def fail_factory(n_classes: int) -> _FixedEstimator:
        raise AssertionError(f"unexpected estimator creation for {n_classes} classes")

    clf = SvmFourStageClassifier(fail_factory, EMOTION_ORDER)
    prediction = clf.fit(_bundle(3), np.asarray([0, 0, 0], dtype=np.int64)).predict(_bundle(2))

    assert tuple(prediction.classes[idx] for idx in prediction.y_pred) == (
        Emotion.NEUTRAL,
        Emotion.NEUTRAL,
    )
    assert prediction.proba.sum(axis=1) == pytest.approx(np.ones(2))


def test_svm_hierarchy_configs_roundtrip_and_build() -> None:
    two = SvmTwoStageConfig(stage=SvmConfig(C=2.0, kernel="linear"), use_concepts=False)
    four = SvmFourStageConfig(stage=SvmConfig(C=0.5, kernel="rbf"), use_concepts=True)

    assert to_dict(two)["type"] == "svm_two_stage"
    assert from_dict({"name": "two", "model": to_dict(two)}).model == two
    assert isinstance(build_classifier(two, EMOTION_ORDER), SvmTwoStageClassifier)

    assert to_dict(four)["type"] == "svm_four_stage"
    assert from_dict({"name": "four", "model": to_dict(four)}).model == four
    assert isinstance(build_classifier(four, EMOTION_ORDER), SvmFourStageClassifier)


def test_svm_two_stage_synthetic_smoke() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn 미설치")

    result = build_experiment(
        from_dict(
            {
                "name": "svm_two_stage_synthetic",
                "dataset": {"type": "synthetic", "n_train": 80, "n_dev": 12, "n_test": 16},
                "extractors": [{"type": "text_concepts"}, {"type": "text_bow", "n_features": 16}],
                "model": {"type": "svm_two_stage", "stage": {"type": "svm"}},
                "evaluation": {"metrics": ["accuracy"], "confusion": True, "scenarios": ["full"]},
                "reporters": [],
            }
        )
    ).run()

    assert result.evaluation.metric("accuracy") is not None


def _svm_margin_dialogue_config() -> ExperimentConfig:
    return from_dict(
        {
            "name": "two_stage_svm_dialogue_rnn",
            "seed": 0,
            "dataset": {
                "type": "synthetic",
                "n_train": 80,
                "n_dev": 20,
                "n_test": 24,
                "seed": 0,
                "with_audio": True,
                "with_video": True,
            },
            "extractors": [
                {"type": "text_concepts"},
                {"type": "text_bow", "n_features": 64},
                {"type": "audio_concepts"},
                {"type": "video_concepts"},
            ],
            "model": {
                "type": "two_stage_svm_margin",
                "margin_threshold": 0.25,
                "stage1_confidence_threshold": None,
                "stage1_use_concepts": True,
                "neutral_label": "neutral",
                "stage1": {"type": "svm", "C": 1.0, "kernel": "rbf"},
                "stage2": {
                    "type": "dialogue_rnn",
                    "rnn_type": "gru",
                    "modality_encoder": {"proj_dim": 64, "hidden_dim": 64, "dropout": 0.1},
                    "fusion": {
                        "modality_dim": 64,
                        "fusion_dim": 128,
                        "dropout": 0.1,
                        "use_gated_fusion": True,
                        "use_interaction_features": True,
                    },
                    "dialogue_context": {
                        "speaker_emb_dim": 16,
                        "hidden_dim": 128,
                        "num_layers": 1,
                        "dropout": 0.1,
                    },
                    "memory_attention": {
                        "enabled": True,
                        "use_memory": True,
                        "hidden_dim": 128,
                        "attn_dim": 128,
                        "use_rope": False,
                        "use_relative_distance_bias": True,
                        "use_same_speaker_bias": True,
                    },
                    "classifier": {"hidden_dim": 128, "dropout": 0.1},
                    "training": {
                        "lr": 0.0002,
                        "weight_decay": 0.01,
                        "gradient_clip_norm": 1.0,
                        "batch_size": 4,
                        "max_epochs": 1,
                        "early_stopping_patience": 1,
                        "validation_fraction": 0.0,
                        "modality_dropout": 0.0,
                        "seed": 0,
                        "device": "cpu",
                    },
                },
            },
            "evaluation": {
                "metrics": ["accuracy", "macro_f1", "weighted_f1"],
                "confusion": True,
                "scenarios": ["full", "no_text", "no_audio", "no_video"],
            },
            "cache": {"type": "memory"},
            "reporters": [],
        }
    )
