"""특징 파이프라인 (완전 구현).

여러 추출기를 묶어, 학습 분할로 한 번 ``fit`` 한 뒤 임의의 분할을 :class:`FeatureBundle`
로 변환한다. 모달리티별 가용성 마스크를 샘플의 :class:`ModalityMask` 로부터 구성하며,
선택적으로 :class:`FeatureCache` 를 사용해 추출 결과를 재사용한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, Self

import numpy as np

from meld_emotion.core.data import RawSample, VideoInput
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import FeatureCache, FeatureExtractor
from meld_emotion.core.status import real
from meld_emotion.core.types import MODALITY_ORDER, BoolArray, Modality, Split
from meld_emotion.pipeline.cache import NullFeatureCache


class VideoLoader(Protocol):
    """비디오 입력을 실제 프레임 텐서로 적재하는 최소 계약."""

    def load_video(self, video: VideoInput) -> VideoInput: ...


@real
class FeaturePipeline:
    """추출기 묶음 → 특징 묶음 변환기."""

    def __init__(
        self,
        extractors: Sequence[FeatureExtractor],
        cache: FeatureCache | None = None,
        media_loader: VideoLoader | None = None,
    ) -> None:
        if not extractors:
            raise ValueError("최소 한 개의 추출기가 필요합니다")
        self._extractors = tuple(extractors)
        self._cache: FeatureCache = cache if cache is not None else NullFeatureCache()
        self._media_loader = media_loader
        self._needs_video = any(e.modality == Modality.VIDEO for e in self._extractors)

    def fit(self, samples: Sequence[RawSample]) -> Self:
        prepared = self._prepare_media(samples)
        for extractor in self._extractors:
            extractor.fit(prepared)
        return self

    def transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        prepared = self._prepare_media(samples)
        return self._transform_prepared(prepared, split)

    def fit_transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        prepared = self._prepare_media(samples)
        for extractor in self._extractors:
            extractor.fit(prepared)
        return self._transform_prepared(prepared, split)

    def _transform_prepared(
        self, samples: Sequence[RawSample], split: Split
    ) -> FeatureBundle:
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

    def _prepare_media(self, samples: Sequence[RawSample]) -> tuple[RawSample, ...]:
        if self._media_loader is None or not self._needs_video:
            return tuple(samples)

        prepared: list[RawSample] = []
        for sample in samples:
            video = sample.video
            if video is None or video.frames is not None or video.source_path is None:
                prepared.append(sample)
                continue
            prepared.append(replace(sample, video=self._media_loader.load_video(video)))
        return tuple(prepared)

    @staticmethod
    def _availability(samples: Sequence[RawSample]) -> dict[Modality, BoolArray]:
        return {
            modality: np.array([s.has(modality) for s in samples], dtype=np.bool_)
            for modality in MODALITY_ORDER
        }
