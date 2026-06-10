"""지표 계산 정확성."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.core.results import PredictionSet
from meld_emotion.core.types import Emotion
from meld_emotion.evaluation.metrics import (
    METRIC_REGISTRY,
    AccuracyMetric,
    BrierScoreMetric,
    ClasswiseECEMetric,
    ConfidenceBucketAccuracyMetric,
    ExpectedCalibrationErrorMetric,
    HighConfidenceWrongMetric,
    MacroF1Metric,
    NLLMetric,
    PerClassRecallMetric,
    confusion_counts,
)

_CLASSES = (Emotion.NEUTRAL, Emotion.JOY, Emotion.SADNESS)


def _prediction(y_pred: list[int]) -> PredictionSet:
    n = len(y_pred)
    proba = np.zeros((n, len(_CLASSES)), dtype=np.float64)
    for i, c in enumerate(y_pred):
        proba[i, c] = 1.0
    return PredictionSet(
        uids=tuple(str(i) for i in range(n)),
        y_pred=np.array(y_pred, dtype=np.int64),
        proba=proba,
        classes=_CLASSES,
    )


def _prediction_with_proba(proba: list[list[float]]) -> PredictionSet:
    arr = np.asarray(proba, dtype=np.float64)
    return PredictionSet(
        uids=tuple(str(i) for i in range(arr.shape[0])),
        y_pred=np.argmax(arr, axis=1).astype(np.int64),
        proba=arr,
        classes=_CLASSES,
    )


def test_confusion_counts() -> None:
    y_true = np.array([0, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2], dtype=np.int64)
    cm = confusion_counts(y_true, y_pred, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1 and cm[2, 2] == 1


def test_accuracy() -> None:
    y_true = np.array([0, 1, 2, 0], dtype=np.int64)
    result = AccuracyMetric().compute(y_true, _prediction([0, 1, 2, 1]))
    assert result.value == 0.75


def test_macro_f1_perfect() -> None:
    y_true = np.array([0, 1, 2], dtype=np.int64)
    result = MacroF1Metric().compute(y_true, _prediction([0, 1, 2]))
    assert result.value == 1.0
    assert result.per_class is not None
    assert result.per_class[Emotion.JOY] == 1.0


def test_per_class_recall() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    result = PerClassRecallMetric().compute(y_true, _prediction([0, 1, 1, 1]))
    assert result.per_class is not None
    assert result.per_class[Emotion.NEUTRAL] == 0.5  # 0 중 하나만 맞음
    assert result.per_class[Emotion.JOY] == 1.0


def test_nll_metric_exact_value() -> None:
    y_true = np.array([0, 1], dtype=np.int64)
    prediction = _prediction_with_proba([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])

    result = NLLMetric().compute(y_true, prediction)

    expected = float((-np.log(0.8) - np.log(0.5)) / 2.0)
    assert result.value == pytest.approx(expected)


def test_brier_score_metric_exact_value() -> None:
    y_true = np.array([0, 1], dtype=np.int64)
    prediction = _prediction_with_proba([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])

    result = BrierScoreMetric().compute(y_true, prediction)

    expected = ((0.8 - 1.0) ** 2 + 0.1**2 + 0.1**2 + 0.2**2 + (0.5 - 1.0) ** 2 + 0.3**2) / 2.0
    assert result.value == pytest.approx(expected)


def test_ece_metric_with_manually_verifiable_bins() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    prediction = _prediction_with_proba(
        [
            [0.60, 0.30, 0.10],
            [0.55, 0.35, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.85, 0.05],
        ]
    )

    result = ExpectedCalibrationErrorMetric(n_bins=2).compute(y_true, prediction)

    assert result.value == pytest.approx(0.175)
    assert result.details is not None
    assert result.details["bin_1_count"] == 4.0
    assert result.details["bin_1_accuracy"] == pytest.approx(0.5)
    assert result.details["bin_1_confidence"] == pytest.approx(0.675)


def test_classwise_ece_known_small_example() -> None:
    y_true = np.array([0, 1], dtype=np.int64)
    prediction = _prediction_with_proba([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]])

    result = ClasswiseECEMetric(n_bins=2).compute(y_true, prediction)

    assert result.per_class is not None
    assert result.per_class[Emotion.NEUTRAL] == pytest.approx(0.15)
    assert result.per_class[Emotion.JOY] == pytest.approx(0.15)
    assert result.per_class[Emotion.SADNESS] == pytest.approx(0.0)
    assert result.value == pytest.approx(0.1)


def test_confidence_bucket_accuracy_details() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    prediction = _prediction_with_proba(
        [
            [0.60, 0.30, 0.10],
            [0.55, 0.35, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.85, 0.05],
        ]
    )

    result = ConfidenceBucketAccuracyMetric(n_bins=2).compute(y_true, prediction)

    assert result.value == pytest.approx(0.5)
    assert result.details is not None
    assert result.details["bin_1_count"] == 4.0
    assert result.details["bin_1_accuracy"] == pytest.approx(0.5)


def test_high_confidence_wrong_detects_threshold_errors() -> None:
    y_true = np.array([0, 1, 2], dtype=np.int64)
    prediction = _prediction_with_proba(
        [
            [0.95, 0.03, 0.02],
            [0.91, 0.08, 0.01],
            [0.30, 0.20, 0.50],
        ]
    )

    result = HighConfidenceWrongMetric(threshold=0.9).compute(y_true, prediction)

    assert result.value == pytest.approx(1.0 / 3.0)
    assert result.details is not None
    assert result.details["count"] == 1.0


def test_probability_metrics_handle_logits_consistently() -> None:
    y_true = np.array([0, 1], dtype=np.int64)
    logits = np.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    from_logits = NLLMetric().compute(y_true, _prediction_with_proba(logits.tolist()))
    from_proba = NLLMetric().compute(y_true, _prediction_with_proba(proba.tolist()))

    assert from_logits.value == pytest.approx(from_proba.value)


def test_probability_metrics_fail_on_non_finite_scores() -> None:
    y_true = np.array([0], dtype=np.int64)
    prediction = _prediction_with_proba([[float("nan"), 0.0, 0.0]])

    with pytest.raises(ValueError, match="non-finite"):
        NLLMetric().compute(y_true, prediction)


def test_new_metrics_are_registered_for_config_names() -> None:
    for name in (
        "nll",
        "brier_score",
        "expected_calibration_error",
        "classwise_ece",
        "confidence_bucket_accuracy",
        "high_confidence_wrong",
    ):
        metric = METRIC_REGISTRY.create(name)
        assert hasattr(metric, "compute")
