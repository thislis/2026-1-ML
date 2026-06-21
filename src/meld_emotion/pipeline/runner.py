"""실험 러너 (완전 구현) — 최상위 오케스트레이션.

데이터 적재 → 특징 추출 → (선택)modality dropout → 분류기 학습 → 평가 → 강건성 → 설명 →
리포팅 순으로 진행한다. 오직 :mod:`meld_emotion.core` 의 Protocol 에만 의존하므로(DIP),
어떤 구체 구현이 주입되든 동일하게 동작한다. 구체 연결은 :mod:`builder` 가 담당한다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from meld_emotion.core.data import ModalityMask, RawSample
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
    DialogueXaiResult,
    ExperimentResult,
    ExplanationReport,
    FeatureContribution,
    MetricResult,
    ModalityContribution,
    RobustnessReport,
)
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, IntArray, Modality, Split
from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.evaluation.robustness import RobustnessEvaluator
from meld_emotion.fusion.masking import ModalityDropout, ModalityScenario, get_scenario
from meld_emotion.models.artifact import save_classifier_artifact
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


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
        train_augmentation_scenarios: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
        artifact_path: str | None = None,
        artifact_config: Mapping[str, Any] | None = None,
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
        self._train_augmentation_scenarios = tuple(train_augmentation_scenarios)
        self._metadata = dict(metadata or {})
        self._artifact_path = artifact_path
        self._artifact_config = dict(artifact_config or {})

    def run(self) -> ExperimentResult:
        try:
            return self._run()
        except Exception:
            logger.exception("실험 실패: %s", self._name)
            raise

    def _run(self) -> ExperimentResult:
        logger.info("실험 시작: %s", self._name)
        logger.info(
            "데이터 적재 시작: train_split=%s eval_split=%s",
            self._train_split.value,
            self._eval_split.value,
        )
        train = list(self._source.load(self._train_split))
        test = list(self._source.load(self._eval_split))
        logger.info("데이터 적재 완료: train=%d eval=%d", len(train), len(test))
        n_train_raw = len(train)
        if self._train_augmentation_scenarios:
            train = _augment_samples_for_scenarios(train, self._train_augmentation_scenarios)
            logger.info(
                "학습 입력 시나리오 증강 적용: raw=%d augmented=%d scenarios=%s",
                n_train_raw,
                len(train),
                ",".join(self._train_augmentation_scenarios),
            )

        logger.info("학습 특징 추출 시작")
        train_bundle = self._features.fit_transform(train, self._train_split)
        self._log_bundle("학습 특징 추출 완료", train_bundle, len(train))
        if self._dropout is not None:
            logger.info(
                "학습 modality dropout 적용: drop_prob=%.3f seed=%d",
                self._dropout.drop_prob,
                self._dropout.seed,
            )
            train_bundle = self._dropout.apply(train_bundle)
        y_train = self._labels_for_bundle(train, train_bundle)
        logger.info("모델 학습 시작: classifier=%s", type(self._classifier).__name__)
        self._classifier.fit(train_bundle, y_train)
        logger.info("모델 학습 완료")

        logger.info("평가 특징 추출 시작")
        test_bundle = self._features.transform(test, self._eval_split)
        self._log_bundle("평가 특징 추출 완료", test_bundle, len(test))
        y_test = self._labels_for_bundle(test, test_bundle)

        logger.info("평가 시작: scenario=full")
        evaluation = self._evaluator.evaluate(self._classifier, test_bundle, y_test, "full")
        logger.info("평가 완료: %s", _metric_summary(evaluation.metrics))
        robustness = self._run_robustness(test, test_bundle, y_test)
        explanation = self._run_explainers(test_bundle, y_test)

        metadata = {
            **self._metadata,
            **_classifier_diagnostics(self._classifier),
            "classifier": type(self._classifier).__name__,
            "n_train": str(train_bundle.n_samples),
            "n_test": str(test_bundle.n_samples),
            "n_train_raw": str(n_train_raw),
            "n_train_augmented_raw": str(len(train)),
            "n_test_raw": str(len(test)),
            "train_split": self._train_split.value,
            "eval_split": self._eval_split.value,
            "dropout": (
                "none"
                if self._dropout is None
                else f"p={self._dropout.drop_prob}, seed={self._dropout.seed}"
            ),
        }
        result = ExperimentResult(
            name=self._name,
            evaluation=evaluation,
            robustness=robustness,
            explanation=explanation,
            metadata=metadata,
        )
        if self._artifact_path is not None:
            artifact_path = Path(self._artifact_path)
            logger.info("classifier artifact 저장 시작: path=%s", artifact_path)
            save_classifier_artifact(
                artifact_path,
                self._classifier,
                self._artifact_config,
                {**metadata, "artifact_path": str(artifact_path)},
            )
            logger.info("classifier artifact 저장 완료: path=%s", artifact_path)
        for reporter in self._reporters:
            logger.info("리포터 저장 시작: reporter=%s", type(reporter).__name__)
            reporter.save(result)
            logger.info("리포터 저장 완료: reporter=%s", type(reporter).__name__)
        logger.info("실험 완료: %s", self._name)
        return result

    def _run_robustness(
        self, samples: Sequence[RawSample], bundle: FeatureBundle, y_true: IntArray
    ) -> RobustnessReport | None:
        if self._robustness is None:
            logger.info("강건성 평가 건너뜀: scenarios 없음")
            return None
        if _has_multimodal_features(bundle):
            logger.info("강건성 평가 시작: 입력 시나리오별 재임베딩")
            reports = []
            for scenario in self._robustness.scenarios:
                logger.info("강건성 재임베딩 시작: scenario=%s", scenario.name)
                scenario_samples = _samples_for_scenario(samples, scenario)
                scenario_bundle = (
                    bundle
                    if scenario.name == "full"
                    else self._features.transform(scenario_samples, self._eval_split)
                )
                scenario_y = self._labels_for_bundle(samples, scenario_bundle)
                reports.append(
                    self._evaluator.evaluate(
                        self._classifier,
                        scenario_bundle,
                        scenario_y,
                        scenario.name,
                    )
                )
                logger.info("강건성 재임베딩 완료: scenario=%s", scenario.name)
            report = RobustnessReport(reports=tuple(reports))
            logger.info("강건성 평가 완료: scenarios=%d", len(report.reports))
            return report
        logger.info("강건성 평가 시작")
        report = self._robustness.evaluate(self._classifier, bundle, y_true)
        logger.info("강건성 평가 완료: scenarios=%d", len(report.reports))
        return report

    def _run_explainers(self, bundle: FeatureBundle, y_true: IntArray) -> ExplanationReport | None:
        if not self._explainers:
            logger.info("설명 생성 건너뜀: explainers 없음")
            return None
        logger.info("설명 생성 시작: explainers=%d", len(self._explainers))
        reports: list[ExplanationReport] = []
        for explainer in self._explainers:
            logger.info("설명기 실행 시작: explainer=%s", type(explainer).__name__)
            reports.append(explainer.explain(self._classifier, bundle, y_true))
            logger.info("설명기 실행 완료: explainer=%s", type(explainer).__name__)
        logger.info("설명 생성 완료")
        return _merge_explanations(reports)

    def _labels(self, samples: Sequence[RawSample]) -> IntArray:
        emotions: list[Emotion] = []
        for sample in samples:
            if sample.emotion is None:
                raise ValueError(f"샘플 {sample.uid} 에 감정 레이블이 없습니다")
            emotions.append(sample.emotion)
        return self._encoder.encode(emotions)

    def _labels_for_bundle(self, samples: Sequence[RawSample], bundle: FeatureBundle) -> IntArray:
        by_uid = {sample.uid: sample for sample in samples}
        return self._labels([by_uid[uid] for uid in bundle.uids])

    @staticmethod
    def _log_bundle(message: str, bundle: FeatureBundle, raw_count: int) -> None:
        n_features = sum(matrix.n_features for matrix in bundle.matrices)
        modalities = ",".join(modality.value for modality in bundle.modalities) or "-"
        logger.info(
            "%s: samples=%d raw_samples=%d matrices=%d features=%d modalities=%s",
            message,
            bundle.n_samples,
            raw_count,
            len(bundle.matrices),
            n_features,
            modalities,
        )
        if bundle.n_samples < raw_count:
            logger.warning(
                "특징 준비 중 샘플 제외 감지: raw_samples=%d kept_samples=%d",
                raw_count,
                bundle.n_samples,
            )


def _merge_explanations(reports: Sequence[ExplanationReport]) -> ExplanationReport:
    feature: tuple[FeatureContribution, ...] = ()
    modality: tuple[ModalityContribution, ...] = ()
    counterfactual: tuple[CounterfactualResult, ...] = ()
    dialogue_xai: tuple[DialogueXaiResult, ...] = ()
    for report in reports:
        feature += report.feature_contributions
        modality += report.modality_contributions
        counterfactual += report.counterfactuals
        dialogue_xai += report.dialogue_xai
    return ExplanationReport(
        feature_contributions=feature,
        modality_contributions=modality,
        counterfactuals=counterfactual,
        dialogue_xai=dialogue_xai,
    )


def _metric_summary(metrics: Sequence[MetricResult]) -> str:
    return ", ".join(f"{metric.name}={metric.value:.4f}" for metric in metrics) or "-"


def _classifier_diagnostics(classifier: Classifier) -> dict[str, str]:
    metadata: dict[str, str] = {}
    gate_stats = getattr(classifier, "last_gate_stats", None)
    if isinstance(gate_stats, Mapping):
        for key, value in gate_stats.items():
            metadata[f"validation_{key}"] = str(value)
    false_positive_counts = getattr(classifier, "last_false_positive_counts", None)
    if false_positive_counts is not None:
        with suppress(TypeError):
            metadata["validation_false_positive_counts"] = ",".join(
                str(int(value)) for value in false_positive_counts
            )
    calibration_params = getattr(classifier, "calibration_params", None)
    if isinstance(calibration_params, Mapping):
        metadata["calibration_params"] = json.dumps(
            calibration_params,
            ensure_ascii=False,
            sort_keys=True,
        )
    return metadata


def _has_multimodal_features(bundle: FeatureBundle) -> bool:
    return bool(bundle.by_modality(Modality.MULTIMODAL))


def _samples_for_scenario(
    samples: Sequence[RawSample],
    scenario: ModalityScenario,
    *,
    identity_prefix: str | None = None,
    dialogue_offset: int = 0,
) -> list[RawSample]:
    result: list[RawSample] = []
    for sample in samples:
        available = tuple(mod for mod in sample.mask.available if scenario.has(mod))
        uid = sample.uid if identity_prefix is None else f"{identity_prefix}:{sample.uid}"
        result.append(
            replace(
                sample,
                uid=uid,
                dialogue_id=sample.dialogue_id + dialogue_offset,
                mask=ModalityMask.of(*available),
            )
        )
    return result


def _augment_samples_for_scenarios(
    samples: Sequence[RawSample], scenario_names: Sequence[str]
) -> list[RawSample]:
    augmented = list(samples)
    if not samples:
        return augmented
    max_dialogue = max(sample.dialogue_id for sample in samples)
    stride = max_dialogue + 1
    for index, name in enumerate(scenario_names, start=1):
        scenario = get_scenario(name)
        augmented.extend(
            _samples_for_scenario(
                samples,
                scenario,
                identity_prefix=scenario.name,
                dialogue_offset=stride * index,
            )
        )
    return augmented
