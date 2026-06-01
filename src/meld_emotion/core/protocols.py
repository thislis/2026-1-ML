"""파이프라인 단계 간 계약(Protocol).

모든 구체 컴포넌트는 여기 정의된 Protocol 중 하나를 구조적으로 만족한다. 상위 오케스트레이션
(:mod:`meld_emotion.pipeline`)은 구체 클래스가 아니라 이 Protocol 에만 의존하므로(DIP),
구현을 자유롭게 교체할 수 있다. ``@runtime_checkable`` 로 표시된 것은 테스트에서 ``isinstance``
구조 검사가 가능하다(메서드 존재만 검사하며 시그니처까지는 검사하지 않음).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, Self, runtime_checkable

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.results import (
    EvaluationReport,
    ExperimentResult,
    ExplanationReport,
    MetricResult,
    PredictionSet,
)
from meld_emotion.core.types import (
    Emotion,
    FeatureKind,
    FloatArray,
    IntArray,
    Modality,
    Split,
)


@runtime_checkable
class LabelEncoder(Protocol):
    """감정 레이블 ↔ 정수 인덱스 변환."""

    @property
    def classes(self) -> tuple[Emotion, ...]:
        """인덱스 순서대로의 감정 클래스."""

    def encode(self, labels: Sequence[Emotion]) -> IntArray:
        """감정 시퀀스를 클래스 인덱스 배열로 변환."""

    def decode(self, indices: IntArray) -> tuple[Emotion, ...]:
        """클래스 인덱스 배열을 감정 튜플로 변환."""


@runtime_checkable
class DatasetSource(Protocol):
    """원천 데이터셋. 분할을 받아 :class:`RawSample` 들을 산출한다."""

    def load(self, split: Split) -> Iterable[RawSample]:
        """주어진 분할의 샘플들을 (지연 가능하게) 산출한다."""


@runtime_checkable
class FeatureExtractor(Protocol):
    """한 모달리티에서 한 종류(임베딩/개념)의 특징을 뽑는 추출기.

    학습이 필요한 추출기(예: TF-IDF 어휘)는 ``fit`` 에서 상태를 학습하고, 상태가 없는
    추출기는 ``fit`` 을 no-op 으로 둔다. ``transform`` 은 항상 (n_samples, n_features)
    행렬을 돌려준다.
    """

    @property
    def name(self) -> str: ...

    @property
    def modality(self) -> Modality: ...

    @property
    def kind(self) -> FeatureKind: ...

    def fit(self, samples: Sequence[RawSample]) -> Self:
        """학습 분할 샘플로 내부 상태를 학습하고 self 를 반환한다."""

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        """샘플들을 특징 행렬로 변환한다."""


@runtime_checkable
class Estimator(Protocol):
    """평범한 행렬(X, y)을 받는 sklearn 형태의 기초 학습기.

    Early/Late fusion 분류기가 내부적으로 감싸 사용하는 단위이다.
    """

    def fit(self, x: FloatArray, y: IntArray) -> Self: ...

    def predict(self, x: FloatArray) -> IntArray: ...

    def predict_proba(self, x: FloatArray) -> FloatArray: ...


@runtime_checkable
class Classifier(Protocol):
    """융합(fusion)을 인지하는 분류기. :class:`FeatureBundle` 단위로 동작한다.

    Early fusion / Late fusion 이 이 Protocol 의 서로 다른 구현이며, 동일한 자리에 교체
    투입할 수 있다(교수님 피드백: Early/Late fusion 비교).
    """

    @property
    def classes(self) -> tuple[Emotion, ...]: ...

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self: ...

    def predict(self, bundle: FeatureBundle) -> PredictionSet: ...

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray: ...


@runtime_checkable
class Metric(Protocol):
    """예측과 정답으로부터 단일 지표를 계산한다."""

    @property
    def name(self) -> str: ...

    def compute(self, y_true: IntArray, prediction: PredictionSet) -> MetricResult: ...


@runtime_checkable
class Explainer(Protocol):
    """학습된 분류기에 대한 설명을 생성한다.

    각 구현은 :class:`ExplanationReport` 의 해당 필드만 채우고 나머지는 비워 두며,
    러너가 이를 병합한다.
    """

    def explain(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray
    ) -> ExplanationReport: ...


@runtime_checkable
class FeatureCache(Protocol):
    """추출된 특징 행렬을 키로 저장/조회한다(추출-1회, 재사용-N회).

    키는 (추출기 정체성 + 분할)로부터 파생된 문자열이다.
    """

    def get(self, key: str) -> FeatureMatrix | None: ...

    def put(self, key: str, matrix: FeatureMatrix) -> None: ...


@runtime_checkable
class Reporter(Protocol):
    """실험 결과를 외부(파일/콘솔/대시보드)로 내보낸다."""

    def save(self, result: ExperimentResult) -> None: ...


@runtime_checkable
class ExperimentEvaluator(Protocol):
    """학습된 모델과 특징 묶음으로 평가 리포트를 만든다."""

    def evaluate(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray, scenario: str
    ) -> EvaluationReport: ...
