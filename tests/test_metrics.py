"""지표 계산 정확성."""

from __future__ import annotations

import numpy as np

from meld_emotion.core.results import PredictionSet
from meld_emotion.core.types import Emotion
from meld_emotion.evaluation.metrics import (
    AccuracyMetric,
    MacroF1Metric,
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
