"""평가 오케스트레이션 (완전 구현)."""

from __future__ import annotations

from collections.abc import Sequence

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier, Metric
from meld_emotion.core.results import EvaluationReport
from meld_emotion.core.status import real
from meld_emotion.core.types import IntArray
from meld_emotion.evaluation.metrics import build_confusion


@real
class Evaluator:
    """주어진 지표들로 한 특징 묶음에 대한 평가 리포트를 만든다."""

    def __init__(self, metrics: Sequence[Metric], confusion: bool = True) -> None:
        self._metrics = tuple(metrics)
        self._confusion = confusion

    def evaluate(
        self,
        model: Classifier,
        bundle: FeatureBundle,
        y_true: IntArray,
        scenario: str = "full",
    ) -> EvaluationReport:
        prediction = model.predict(bundle)
        metrics = tuple(metric.compute(y_true, prediction) for metric in self._metrics)
        confusion = build_confusion(y_true, prediction) if self._confusion else None
        return EvaluationReport(scenario=scenario, metrics=metrics, confusion=confusion)
