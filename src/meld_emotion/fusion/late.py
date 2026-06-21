"""Late fusion 분류기 (완전 구현).

모달리티마다 독립적인 기초 학습기를 학습하고, 예측 시 각 학습기의 확률을 결합기로 합친다.
누락된 모달리티는 가용성 가중을 통해 자동으로 제외된다. Early fusion 과 동일한
:class:`~meld_emotion.core.protocols.Classifier` 계약을 만족한다(교수님 피드백: Early/Late 비교).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self, cast

import numpy as np

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Estimator
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import BoolArray, Emotion, FloatArray, IntArray, Modality
from meld_emotion.fusion.combiners import ProbabilityCombiner


@real
class LateFusionClassifier:
    """모달리티별 학습기 + 확률 결합."""

    def __init__(
        self,
        estimator_factory: Callable[[int], Estimator],
        combiner: ProbabilityCombiner,
        classes: tuple[Emotion, ...],
    ) -> None:
        self._factory = estimator_factory
        self._combiner = combiner
        self._classes = classes
        self._estimators: dict[Modality, Estimator] = {}

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def _design(self, bundle: FeatureBundle, modality: Modality) -> FloatArray:
        return bundle.stack(modalities=[modality]).values

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        self._estimators = {}
        per_modality_train: dict[Modality, FloatArray] = {}
        for modality in bundle.modalities:
            design = self._design(bundle, modality)
            estimator = self._factory(len(self._classes)).fit(design, y)
            self._estimators[modality] = estimator
            per_modality_train[modality] = estimator.predict_proba(design)
        if not self._estimators:
            raise ValueError("Late fusion: 학습할 모달리티가 없습니다")
        self._combiner.fit(per_modality_train, y)
        return self

    def _availability(self, bundle: FeatureBundle, modality: Modality) -> BoolArray:
        return bundle.availability.get(modality, np.ones(bundle.n_samples, dtype=np.bool_))

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        if not self._estimators:
            raise RuntimeError("학습되지 않은 분류기입니다. 먼저 fit 을 호출하세요.")
        per_modality = {
            modality: estimator.predict_proba(self._design(bundle, modality))
            for modality, estimator in self._estimators.items()
        }
        availability = {
            modality: self._availability(bundle, modality) for modality in self._estimators
        }
        return self._combiner.combine(per_modality, availability)

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def __getstate__(self) -> dict[str, object]:
        return {
            "_combiner": self._combiner,
            "_classes": self._classes,
            "_estimators": self._estimators,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._factory = _restored_factory
        self._combiner = cast(ProbabilityCombiner, state["_combiner"])
        self._classes = cast(tuple[Emotion, ...], state["_classes"])
        self._estimators = cast(dict[Modality, Estimator], state["_estimators"])


def _restored_factory(_: int) -> Any:
    raise RuntimeError("artifact 로 복원한 LateFusionClassifier 는 재학습할 수 없습니다")
