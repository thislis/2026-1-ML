"""분류 지표 (완전 구현, numpy 전용).

정확도, Macro/Weighted F1, 클래스별 재현율, 혼동행렬을 계산한다. sklearn 없이 직접
구현하여 동작이 투명하고 테스트가 쉽다. 모든 지표는
:class:`~meld_emotion.core.protocols.Metric` 을 만족한다.
"""

from __future__ import annotations

import numpy as np

from meld_emotion.core.results import ConfusionMatrixResult, MetricResult, PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, IntArray
from meld_emotion.registry import Registry

#: 지표 이름 → 지표 인스턴스 팩토리 (빌더가 사용).
METRIC_REGISTRY: Registry[object] = Registry("metric")
_EPS = 1.0e-12


def confusion_counts(y_true: IntArray, y_pred: IntArray, k: int) -> IntArray:
    """(k, k) 혼동행렬. ``cm[i, j]`` = 실제 i 를 j 로 예측한 수."""

    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def _precision_recall_f1(
    cm: IntArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    actual_pos = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(actual_pos > 0, tp / actual_pos, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1, actual_pos


def build_confusion(y_true: IntArray, prediction: PredictionSet) -> ConfusionMatrixResult:
    cm = confusion_counts(y_true, prediction.y_pred, len(prediction.classes))
    return ConfusionMatrixResult(matrix=cm, labels=prediction.classes)


def _per_class(values: np.ndarray, classes: tuple[Emotion, ...]) -> dict[Emotion, float]:
    return {emotion: float(values[i]) for i, emotion in enumerate(classes)}


def _probabilities(prediction: PredictionSet) -> np.ndarray:
    """Return probabilities, accepting logits in ``prediction.proba`` for metric tests/tools."""

    scores = np.asarray(prediction.proba, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"prediction probabilities/logits must be 2D: ndim={scores.ndim}")
    if scores.shape != (len(prediction.uids), len(prediction.classes)):
        raise ValueError(
            "prediction probabilities/logits shape mismatch: "
            f"{scores.shape} != {(len(prediction.uids), len(prediction.classes))}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("prediction probabilities/logits contain non-finite values")
    if scores.shape[0] == 0:
        return scores

    row_sums = scores.sum(axis=1)
    looks_like_proba = (
        bool(np.all(scores >= -1.0e-9))
        and bool(np.all(scores <= 1.0 + 1.0e-9))
        and bool(np.allclose(row_sums, 1.0, atol=1.0e-6))
    )
    if looks_like_proba:
        clipped = np.clip(scores, _EPS, 1.0)
        return np.asarray(clipped / clipped.sum(axis=1, keepdims=True), dtype=np.float64)

    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    if np.any(denom <= 0.0):
        raise ValueError("cannot convert logits to probabilities")
    return np.asarray(exp / denom, dtype=np.float64)


def _validate_bins(n_bins: int) -> None:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")


def _confidence_bins(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    _validate_bins(n_bins)
    return np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)


@real
class AccuracyMetric:
    name = "accuracy"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        value = float(np.mean(y_true == prediction.y_pred)) if y_true.size else 0.0
        return MetricResult(name=self.name, value=value)


@real
class MacroF1Metric:
    name = "macro_f1"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        cm = confusion_counts(y_true, prediction.y_pred, len(prediction.classes))
        _, _, f1, _ = _precision_recall_f1(cm)
        return MetricResult(
            name=self.name,
            value=float(np.mean(f1)),
            per_class=_per_class(f1, prediction.classes),
        )


@real
class WeightedF1Metric:
    name = "weighted_f1"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        cm = confusion_counts(y_true, prediction.y_pred, len(prediction.classes))
        _, _, f1, support = _precision_recall_f1(cm)
        total = support.sum()
        value = float(np.sum(f1 * support) / total) if total > 0 else 0.0
        return MetricResult(name=self.name, value=value)


@real
class PerClassRecallMetric:
    name = "per_class_recall"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        cm = confusion_counts(y_true, prediction.y_pred, len(prediction.classes))
        _, recall, _, _ = _precision_recall_f1(cm)
        return MetricResult(
            name=self.name,
            value=float(np.mean(recall)),
            per_class=_per_class(recall, prediction.classes),
        )


@real
class NLLMetric:
    name = "nll"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0)
        true_p = proba[np.arange(y_true.size), y_true.astype(np.int64)]
        return MetricResult(
            name=self.name, value=float(np.mean(-np.log(np.clip(true_p, _EPS, 1.0))))
        )


@real
class BrierScoreMetric:
    name = "brier_score"

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0)
        one_hot = np.zeros_like(proba)
        one_hot[np.arange(y_true.size), y_true.astype(np.int64)] = 1.0
        return MetricResult(
            name=self.name, value=float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
        )


@real
class ExpectedCalibrationErrorMetric:
    name = "expected_calibration_error"

    def __init__(self, n_bins: int = 10) -> None:
        _validate_bins(n_bins)
        self.n_bins = n_bins

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0)
        confidence = proba.max(axis=1)
        y_pred = np.argmax(proba, axis=1)
        bins = _confidence_bins(confidence, self.n_bins)
        total = float(y_true.size)
        ece = 0.0
        details: dict[str, float] = {}
        for bin_idx in range(self.n_bins):
            selected = bins == bin_idx
            count = int(selected.sum())
            if count == 0:
                details[f"bin_{bin_idx}_count"] = 0.0
                details[f"bin_{bin_idx}_accuracy"] = 0.0
                details[f"bin_{bin_idx}_confidence"] = 0.0
                continue
            accuracy = float(np.mean(y_pred[selected] == y_true[selected]))
            avg_conf = float(np.mean(confidence[selected]))
            ece += count / total * abs(accuracy - avg_conf)
            details[f"bin_{bin_idx}_count"] = float(count)
            details[f"bin_{bin_idx}_accuracy"] = accuracy
            details[f"bin_{bin_idx}_confidence"] = avg_conf
        return MetricResult(name=self.name, value=float(ece), details=details)


@real
class ClasswiseECEMetric:
    name = "classwise_ece"

    def __init__(self, n_bins: int = 10) -> None:
        _validate_bins(n_bins)
        self.n_bins = n_bins

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0)
        values = np.zeros(len(prediction.classes), dtype=np.float64)
        total = float(y_true.size)
        for class_idx in range(len(prediction.classes)):
            class_prob = proba[:, class_idx]
            class_true = (y_true == class_idx).astype(np.float64)
            bins = _confidence_bins(class_prob, self.n_bins)
            class_ece = 0.0
            for bin_idx in range(self.n_bins):
                selected = bins == bin_idx
                count = int(selected.sum())
                if count == 0:
                    continue
                avg_true = float(np.mean(class_true[selected]))
                avg_prob = float(np.mean(class_prob[selected]))
                class_ece += count / total * abs(avg_true - avg_prob)
            values[class_idx] = class_ece
        return MetricResult(
            name=self.name,
            value=float(np.mean(values)),
            per_class=_per_class(values, prediction.classes),
        )


@real
class ConfidenceBucketAccuracyMetric:
    name = "confidence_bucket_accuracy"

    def __init__(self, n_bins: int = 10) -> None:
        _validate_bins(n_bins)
        self.n_bins = n_bins

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0)
        confidence = proba.max(axis=1)
        y_pred = np.argmax(proba, axis=1)
        bins = _confidence_bins(confidence, self.n_bins)
        details: dict[str, float] = {}
        for bin_idx in range(self.n_bins):
            selected = bins == bin_idx
            count = int(selected.sum())
            accuracy = float(np.mean(y_pred[selected] == y_true[selected])) if count else 0.0
            details[f"bin_{bin_idx}_count"] = float(count)
            details[f"bin_{bin_idx}_accuracy"] = accuracy
        return MetricResult(
            name=self.name,
            value=float(np.mean(y_pred == y_true)),
            details=details,
        )


@real
class HighConfidenceWrongMetric:
    name = "high_confidence_wrong"

    def __init__(self, threshold: float = 0.90) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult:
        proba = _probabilities(prediction)
        if y_true.size == 0:
            return MetricResult(name=self.name, value=0.0, details={"count": 0.0, "total": 0.0})
        confidence = proba.max(axis=1)
        y_pred = np.argmax(proba, axis=1)
        wrong = (confidence >= self.threshold) & (y_pred != y_true)
        count = int(wrong.sum())
        total = int(y_true.size)
        return MetricResult(
            name=self.name,
            value=float(count / total),
            details={
                "count": float(count),
                "total": float(total),
                "threshold": float(self.threshold),
            },
        )


for _metric in (
    AccuracyMetric,
    MacroF1Metric,
    WeightedF1Metric,
    PerClassRecallMetric,
    NLLMetric,
    BrierScoreMetric,
    ExpectedCalibrationErrorMetric,
    ClasswiseECEMetric,
    ConfidenceBucketAccuracyMetric,
    HighConfidenceWrongMetric,
):
    METRIC_REGISTRY.add(_metric.name, _metric)
