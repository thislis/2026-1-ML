"""파이프라인 산출물 컨테이너 (불변 dataclass).

예측 결과, 평가 지표, 강건성 리포트, 설명(Explanation) 결과, 그리고 한 실험 전체의
결과를 담는 :class:`ExperimentResult` 를 정의한다. 모두 직렬화(리포팅)를 염두에 둔
순수 데이터 구조이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from meld_emotion.core.types import UID, Emotion, FloatArray, IntArray, Modality


@dataclass(frozen=True, eq=False)
class PredictionSet:
    """한 배치(분할)에 대한 예측 결과.

    ``y_pred`` 는 클래스 인덱스, ``proba`` 는 (n_samples, n_classes) 확률 행렬,
    ``classes`` 는 인덱스 → 감정 매핑이다.
    """

    uids: tuple[UID, ...]
    y_pred: IntArray
    proba: FloatArray
    classes: tuple[Emotion, ...]

    def __post_init__(self) -> None:
        if self.proba.shape != (len(self.uids), len(self.classes)):
            raise ValueError(
                "proba 형상이 (n_samples, n_classes) 와 다릅니다: "
                f"{self.proba.shape} != {(len(self.uids), len(self.classes))}"
            )


@dataclass(frozen=True)
class MetricResult:
    """단일 지표 결과. ``per_class`` 는 클래스별 값(있을 경우)."""

    name: str
    value: float
    per_class: Mapping[Emotion, float] | None = None


@dataclass(frozen=True, eq=False)
class ConfusionMatrixResult:
    """혼동행렬. ``matrix[i, j]`` = 실제 i 를 j 로 예측한 개수."""

    matrix: IntArray
    labels: tuple[Emotion, ...]


@dataclass(frozen=True)
class EvaluationReport:
    """한 (모달리티) 시나리오에 대한 평가 리포트."""

    scenario: str
    metrics: tuple[MetricResult, ...]
    confusion: ConfusionMatrixResult | None = None

    def metric(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)


@dataclass(frozen=True)
class RobustnessReport:
    """여러 모달리티 시나리오(full / no-text / ...)에 대한 평가 모음."""

    reports: tuple[EvaluationReport, ...]

    def by_scenario(self, scenario: str) -> EvaluationReport | None:
        return next((r for r in self.reports if r.scenario == scenario), None)


# --- 설명(Explanation) 결과 -----------------------------------------------------


@dataclass(frozen=True)
class FeatureContribution:
    """개별 특징의 기여도 (예: permutation importance)."""

    name: str
    modality: Modality
    importance: float
    std: float = 0.0


@dataclass(frozen=True)
class ModalityContribution:
    """모달리티 제거(ablation) 시 성능 하락폭으로 측정한 기여도."""

    modality: Modality
    score_drop: float
    baseline_score: float
    ablated_score: float


@dataclass(frozen=True, eq=False)
class CounterfactualResult:
    """반사실(counterfactual) 증거 제거 결과 (샘플 단위)."""

    uid: UID
    original_proba: FloatArray
    modified_proba: FloatArray
    removed: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationReport:
    """설명 단계의 종합 결과."""

    feature_contributions: tuple[FeatureContribution, ...] = ()
    modality_contributions: tuple[ModalityContribution, ...] = ()
    counterfactuals: tuple[CounterfactualResult, ...] = ()


@dataclass(frozen=True)
class ExperimentResult:
    """한 실험 전체의 결과 (리포터가 직렬화하는 최상위 객체)."""

    name: str
    evaluation: EvaluationReport
    robustness: RobustnessReport | None = None
    explanation: ExplanationReport | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


# --- 다중 실험 비교(suite) -------------------------------------------------------


@dataclass(frozen=True)
class ExperimentOutcome:
    """비교 묶음 안에서 한 실험의 결과 또는 실패 사유.

    일부 변형이 미구현 경계(``@unimplemented``)에 닿아도 비교 전체를 멈추지 않기 위해,
    성공 시 ``result``, 실패 시 ``error`` (타입+메시지 한 줄) 중 하나만 채운다.
    """

    name: str
    result: ExperimentResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class ComparisonReport:
    """여러 실험을 함께 실행한 비교 결과 (비교 리포터가 직렬화하는 최상위 객체)."""

    name: str
    outcomes: tuple[ExperimentOutcome, ...]

    def successful(self) -> tuple[ExperimentOutcome, ...]:
        return tuple(o for o in self.outcomes if o.result is not None)

    def failed(self) -> tuple[ExperimentOutcome, ...]:
        return tuple(o for o in self.outcomes if o.result is None)
