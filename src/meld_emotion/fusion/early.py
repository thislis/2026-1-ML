"""Early fusion 분류기 (완전 구현).

모든 모달리티의 특징을 하나의 설계 행렬로 결합한 뒤 단일 기초 학습기를 학습한다. 개념
특징 포함 여부는 ``use_concepts`` 로 조절한다. :class:`~meld_emotion.core.protocols.Classifier`
를 만족하며, Late fusion 과 동일한 자리에 교체 투입할 수 있다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import numpy as np

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Estimator
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, FloatArray, IntArray


@real
class EarlyFusionClassifier:
    """특징 수준 결합(concatenation) 후 단일 학습기."""

    def __init__(
        self,
        estimator_factory: Callable[[], Estimator],
        classes: tuple[Emotion, ...],
        use_concepts: bool = True,
    ) -> None:
        self._factory = estimator_factory
        self._classes = classes
        self._use_concepts = use_concepts
        self._estimator: Estimator | None = None

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def _design(self, bundle: FeatureBundle) -> FloatArray:
        stacked = bundle.stack() if self._use_concepts else bundle.embedding_matrix()
        return stacked.values

    def _require(self) -> Estimator:
        if self._estimator is None:
            raise RuntimeError("학습되지 않은 분류기입니다. 먼저 fit 을 호출하세요.")
        return self._estimator

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        self._estimator = self._factory().fit(self._design(bundle), y)
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._require().predict_proba(self._design(bundle))

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)
