"""특징 파이프라인 (완전 구현).

여러 추출기를 묶어, 학습 분할로 한 번 ``fit`` 한 뒤 임의의 분할을 :class:`FeatureBundle`
로 변환한다. 모달리티별 가용성 마스크를 샘플의 :class:`ModalityMask` 로부터 구성하며,
선택적으로 :class:`FeatureCache` 를 사용해 추출 결과를 재사용한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import FeatureCache, FeatureExtractor
from meld_emotion.core.status import real
from meld_emotion.core.types import MODALITY_ORDER, BoolArray, Modality, Split
from meld_emotion.pipeline.cache import NullFeatureCache


@real
class FeaturePipeline:
    """추출기 묶음 → 특징 묶음 변환기."""

    def __init__(
        self,
        extractors: Sequence[FeatureExtractor],
        cache: FeatureCache | None = None,
    ) -> None:
        if not extractors:
            raise ValueError("최소 한 개의 추출기가 필요합니다")
        self._extractors = tuple(extractors)
        self._cache: FeatureCache = cache if cache is not None else NullFeatureCache()

    def fit(self, samples: Sequence[RawSample]) -> Self:
        for extractor in self._extractors:
            extractor.fit(samples)
        return self

    def transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        matrices = []
        for extractor in self._extractors:
            key = f"{extractor.name}|{split.value}"
            cached = self._cache.get(key)
            if cached is not None and cached.n_samples == len(samples):
                matrices.append(cached)
            else:
                matrix = extractor.transform(samples)
                self._cache.put(key, matrix)
                matrices.append(matrix)
        return FeatureBundle(
            uids=tuple(s.uid for s in samples),
            matrices=tuple(matrices),
            availability=self._availability(samples),
        )

    def fit_transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        return self.fit(samples).transform(samples, split)

    @staticmethod
    def _availability(samples: Sequence[RawSample]) -> dict[Modality, BoolArray]:
        return {
            modality: np.array([s.has(modality) for s in samples], dtype=np.bool_)
            for modality in MODALITY_ORDER
        }
