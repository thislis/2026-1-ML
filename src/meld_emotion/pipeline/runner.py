"""실험 러너 (완전 구현) — 최상위 오케스트레이션.

데이터 적재 → 특징 추출 → (선택)modality dropout → 분류기 학습 → 평가 → 강건성 → 설명 →
리포팅 순으로 진행한다. 오직 :mod:`meld_emotion.core` 의 Protocol 에만 의존하므로(DIP),
어떤 구체 구현이 주입되든 동일하게 동작한다. 구체 연결은 :mod:`builder` 가 담당한다.
"""

from __future__ import annotations

from collections.abc import Sequence

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import (
    Classifier,
    DatasetSource,
    Explainer,
    LabelEncoder,
    Reporter,
)
from meld_emotion.core.results import (
    CounterfactualResult,
    ExperimentResult,
    ExplanationReport,
    FeatureContribution,
    ModalityContribution,
    RobustnessReport,
)
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, IntArray, Split
from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.evaluation.robustness import RobustnessEvaluator
from meld_emotion.fusion.masking import ModalityDropout
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


@real
class ExperimentRunner:
    """한 실험을 끝까지 실행하고 :class:`ExperimentResult` 를 반환한다."""

    def __init__(
        self,
        name: str,
        source: DatasetSource,
        feature_pipeline: FeaturePipeline,
        label_encoder: LabelEncoder,
        classifier: Classifier,
        evaluator: Evaluator,
        robustness: RobustnessEvaluator | None = None,
        explainers: Sequence[Explainer] = (),
        reporters: Sequence[Reporter] = (),
        train_split: Split = Split.TRAIN,
        eval_split: Split = Split.TEST,
        dropout: ModalityDropout | None = None,
    ) -> None:
        self._name = name
        self._source = source
        self._features = feature_pipeline
        self._encoder = label_encoder
        self._classifier = classifier
        self._evaluator = evaluator
        self._robustness = robustness
        self._explainers = tuple(explainers)
        self._reporters = tuple(reporters)
        self._train_split = train_split
        self._eval_split = eval_split
        self._dropout = dropout

    def run(self) -> ExperimentResult:
        train = list(self._source.load(self._train_split))
        test = list(self._source.load(self._eval_split))

        train_bundle = self._features.fit_transform(train, self._train_split)
        if self._dropout is not None:
            train_bundle = self._dropout.apply(train_bundle)
        y_train = self._labels(train)
        self._classifier.fit(train_bundle, y_train)

        test_bundle = self._features.transform(test, self._eval_split)
        y_test = self._labels(test)

        evaluation = self._evaluator.evaluate(self._classifier, test_bundle, y_test, "full")
        robustness = self._run_robustness(test_bundle, y_test)
        explanation = self._run_explainers(test_bundle, y_test)

        result = ExperimentResult(
            name=self._name,
            evaluation=evaluation,
            robustness=robustness,
            explanation=explanation,
            metadata={
                "classifier": type(self._classifier).__name__,
                "n_train": str(len(train)),
                "n_test": str(len(test)),
                "train_split": self._train_split.value,
                "eval_split": self._eval_split.value,
                "dropout": (
                    "none"
                    if self._dropout is None
                    else f"p={self._dropout.drop_prob}, seed={self._dropout.seed}"
                ),
            },
        )
        for reporter in self._reporters:
            reporter.save(result)
        return result

    def _run_robustness(self, bundle: FeatureBundle, y_true: IntArray) -> RobustnessReport | None:
        if self._robustness is None:
            return None
        return self._robustness.evaluate(self._classifier, bundle, y_true)

    def _run_explainers(self, bundle: FeatureBundle, y_true: IntArray) -> ExplanationReport | None:
        if not self._explainers:
            return None
        reports = [e.explain(self._classifier, bundle, y_true) for e in self._explainers]
        return _merge_explanations(reports)

    def _labels(self, samples: Sequence[RawSample]) -> IntArray:
        emotions: list[Emotion] = []
        for sample in samples:
            if sample.emotion is None:
                raise ValueError(f"샘플 {sample.uid} 에 감정 레이블이 없습니다")
            emotions.append(sample.emotion)
        return self._encoder.encode(emotions)


def _merge_explanations(reports: Sequence[ExplanationReport]) -> ExplanationReport:
    feature: tuple[FeatureContribution, ...] = ()
    modality: tuple[ModalityContribution, ...] = ()
    counterfactual: tuple[CounterfactualResult, ...] = ()
    for report in reports:
        feature += report.feature_contributions
        modality += report.modality_contributions
        counterfactual += report.counterfactuals
    return ExplanationReport(
        feature_contributions=feature,
        modality_contributions=modality,
        counterfactuals=counterfactual,
    )
