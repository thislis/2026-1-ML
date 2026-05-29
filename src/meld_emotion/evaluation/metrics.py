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


for _metric in (AccuracyMetric, MacroF1Metric, WeightedF1Metric, PerClassRecallMetric):
    METRIC_REGISTRY.add(_metric.name, _metric)
