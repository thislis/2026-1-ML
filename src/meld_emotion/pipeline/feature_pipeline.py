"""특징 파이프라인 (완전 구현).

여러 추출기를 묶어, 학습 분할로 한 번 ``fit`` 한 뒤 임의의 분할을 :class:`FeatureBundle`
로 변환한다. 모달리티별 가용성 마스크를 샘플의 :class:`ModalityMask` 로부터 구성하며,
선택적으로 :class:`FeatureCache` 를 사용해 추출 결과를 재사용한다.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, Self

import numpy as np

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.features import FeatureBundle, UtteranceSpec
from meld_emotion.core.protocols import FeatureCache, FeatureExtractor
from meld_emotion.core.status import real
from meld_emotion.core.types import MODALITY_ORDER, BoolArray, Modality, Split
from meld_emotion.pipeline.cache import NullFeatureCache


class MediaLoader(Protocol):
    """원천 미디어 입력을 실제 배열로 적재하는 최소 계약."""

    def load_audio(self, audio: AudioInput) -> AudioInput: ...

    def load_video(self, video: VideoInput) -> VideoInput: ...


@real
class FeaturePipeline:
    """추출기 묶음 → 특징 묶음 변환기."""

    def __init__(
        self,
        extractors: Sequence[FeatureExtractor],
        cache: FeatureCache | None = None,
        media_loader: MediaLoader | None = None,
        media_error_policy: str = "raise",
    ) -> None:
        if not extractors:
            raise ValueError("최소 한 개의 추출기가 필요합니다")
        if media_error_policy not in {"raise", "drop_modality", "drop_sample"}:
            raise ValueError(
                "media_error_policy 는 'raise', 'drop_modality', 'drop_sample' 중 하나여야 합니다"
            )
        self._extractors = tuple(extractors)
        self._cache: FeatureCache = cache if cache is not None else NullFeatureCache()
        self._media_loader = media_loader
        self._media_error_policy = media_error_policy
        self._needs_audio = any(e.modality == Modality.AUDIO for e in self._extractors)
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
            utterances=tuple(
                UtteranceSpec(
                    uid=s.uid,
                    dialogue_id=s.dialogue_id,
                    utterance_id=s.utterance_id,
                    speaker=s.speaker,
                )
                for s in samples
            ),
        )

    def _prepare_media(self, samples: Sequence[RawSample]) -> tuple[RawSample, ...]:
        if self._media_loader is None or not (self._needs_audio or self._needs_video):
            return tuple(samples)

        prepared: list[RawSample] = []
        for sample in samples:
            current = sample
            audio = current.audio
            if (
                self._needs_audio
                and audio is not None
                and audio.waveform is None
                and audio.source_path is not None
            ):
                try:
                    current = replace(current, audio=self._media_loader.load_audio(audio))
                except (FileNotFoundError, ValueError) as exc:
                    handled = self._handle_media_error(current, Modality.AUDIO, exc)
                    if handled is None:
                        continue
                    current = handled

            video = current.video
            if (
                self._needs_video
                and video is not None
                and video.frames is None
                and video.source_path is not None
            ):
                try:
                    current = replace(current, video=self._media_loader.load_video(video))
                except (FileNotFoundError, ValueError) as exc:
                    handled = self._handle_media_error(current, Modality.VIDEO, exc)
                    if handled is None:
                        continue
                    current = handled

            prepared.append(current)
        return tuple(prepared)

    def _handle_media_error(
        self, sample: RawSample, modality: Modality, exc: Exception
    ) -> RawSample | None:
        if self._media_error_policy == "raise":
            raise exc
        if self._media_error_policy == "drop_sample":
            warnings.warn(
                f"{sample.uid} 의 {modality.value} media 를 읽지 못해 샘플 전체를 제외합니다: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        warnings.warn(
            f"{sample.uid} 의 {modality.value} media 를 읽지 못해 해당 모달리티를 누락 처리합니다: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        remaining = tuple(m for m in sample.mask.available if m != modality)
        if modality == Modality.AUDIO:
            return replace(sample, audio=None, mask=ModalityMask.of(*remaining))
        if modality == Modality.VIDEO:
            return replace(sample, video=None, mask=ModalityMask.of(*remaining))
        return replace(sample, mask=ModalityMask.of(*remaining))

    @staticmethod
    def _availability(samples: Sequence[RawSample]) -> dict[Modality, BoolArray]:
        return {
            modality: np.array([s.has(modality) for s in samples], dtype=np.bool_)
            for modality in MODALITY_ORDER
        }
