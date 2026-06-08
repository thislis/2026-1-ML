"""특징 추출기 공통 기반 클래스.

:class:`~meld_emotion.core.protocols.FeatureExtractor` 프로토콜을 만족하는 추상 기반.
상태 학습이 필요 없는 추출기는 ``fit`` 을 그대로 두면 no-op 이다. 모달리티가 없는 샘플에
대해서는 0 벡터를 출력하도록 :meth:`_row_or_zeros` 헬퍼를 제공한다(가용성은 파이프라인이
별도 마스크로 관리).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix, FeatureUnit, SequenceFeatureMatrix
from meld_emotion.core.types import BoolArray, FeatureKind, FloatArray, Modality


class BaseFeatureExtractor(ABC):
    """모든 특징 추출기의 공통 기반."""

    modality: ClassVar[Modality]
    kind: ClassVar[FeatureKind]
    feature_names: ClassVar[tuple[str, ...]] = ()

    @property
    def name(self) -> str:
        return type(self).__name__

    def fit(self, samples: Sequence[RawSample]) -> Self:
        return self

    @abstractmethod
    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix: ...

    # -- 헬퍼 -------------------------------------------------------------------
    def _matrix(self, values: FloatArray, names: Sequence[str]) -> FeatureMatrix:
        return FeatureMatrix(
            values=np.asarray(values, dtype=np.float64),
            names=tuple(names),
            modality=self.modality,
            kind=self.kind,
            source=self.name,
        )

    def _stack_rows(self, rows: Sequence[FloatArray], names: Sequence[str]) -> FeatureMatrix:
        if not rows:
            values = np.zeros((0, len(names)), dtype=np.float64)
        else:
            values = np.vstack(rows).astype(np.float64)
        return self._matrix(values, names)


class BaseSequenceFeatureExtractor(BaseFeatureExtractor):
    """2D pooled matrix 와 3D sequence matrix 를 함께 내는 추출기 기반."""

    @abstractmethod
    def transform_sequence(self, samples: Sequence[RawSample]) -> SequenceFeatureMatrix: ...

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        sequence = self.transform_sequence(samples)
        values = _masked_mean(sequence.values, sequence.mask)
        return self._matrix(values, sequence.names)

    def _sequence_matrix(
        self,
        values: FloatArray,
        mask: BoolArray,
        units: Sequence[Sequence[FeatureUnit]],
        names: Sequence[str],
    ) -> SequenceFeatureMatrix:
        return SequenceFeatureMatrix(
            values=np.asarray(values, dtype=np.float64),
            mask=np.asarray(mask, dtype=bool),
            units=tuple(tuple(row) for row in units),
            names=tuple(names),
            modality=self.modality,
            kind=self.kind,
            source=self.name,
        )


def _masked_mean(values: FloatArray, mask: BoolArray) -> FloatArray:
    if values.shape[0] == 0:
        return np.zeros((0, values.shape[2]), dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)[..., None]
    summed = (values * weights).sum(axis=1)
    counts = weights.sum(axis=1).clip(min=1.0)
    return np.asarray(summed / counts, dtype=np.float64)
