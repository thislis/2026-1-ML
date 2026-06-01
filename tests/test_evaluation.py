"""Evaluator 와 RobustnessEvaluator."""

from __future__ import annotations

from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.evaluation.metrics import AccuracyMetric, MacroF1Metric
from meld_emotion.evaluation.robustness import RobustnessEvaluator
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.masking import get_scenario
from meld_emotion.models.baselines import NearestCentroidEstimator


def _fitted_model(train_bundle, y_train):
    return EarlyFusionClassifier(NearestCentroidEstimator, EmotionLabelEncoder().classes).fit(
        train_bundle, y_train
    )


def test_evaluator_report(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _fitted_model(train_bundle, y_train)
    evaluator = Evaluator([AccuracyMetric(), MacroF1Metric()], confusion=True)
    report = evaluator.evaluate(model, test_bundle, y_test, "full")
    assert report.scenario == "full"
    assert {m.name for m in report.metrics} == {"accuracy", "macro_f1"}
    assert report.confusion is not None
    assert report.confusion.matrix.shape == (7, 7)


def test_robustness_covers_scenarios(train_bundle, test_bundle, y_train, y_test) -> None:
    model = _fitted_model(train_bundle, y_train)
    evaluator = Evaluator([AccuracyMetric()])
    scenarios = [get_scenario(n) for n in ("full", "no_text", "text_only")]
    report = RobustnessEvaluator(evaluator, scenarios).evaluate(model, test_bundle, y_test)
    names = {r.scenario for r in report.reports}
    assert names == {"full", "no_text", "text_only"}
    # full 이 text_only 보다 정확도가 높거나 같아야 한다(정보가 더 많음).
    full_report = report.by_scenario("full")
    text_report = report.by_scenario("text_only")
    assert full_report is not None and text_report is not None
    full_acc = full_report.metric("accuracy")
    text_acc = text_report.metric("accuracy")
    assert full_acc is not None and text_acc is not None
    assert full_acc.value >= text_acc.value
