"""평가 지표와 평가 오케스트레이션.

새 지표/시나리오 추가법은 ``evaluation/README.md`` 참고.
"""

from __future__ import annotations

from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.evaluation.metrics import (
    METRIC_REGISTRY,
    AccuracyMetric,
    MacroF1Metric,
    PerClassRecallMetric,
    WeightedF1Metric,
    build_confusion,
    confusion_counts,
)
from meld_emotion.evaluation.robustness import RobustnessEvaluator

__all__ = [
    "METRIC_REGISTRY",
    "AccuracyMetric",
    "Evaluator",
    "MacroF1Metric",
    "PerClassRecallMetric",
    "RobustnessEvaluator",
    "WeightedF1Metric",
    "build_confusion",
    "confusion_counts",
]
