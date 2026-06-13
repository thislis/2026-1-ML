"""특징 파이프라인 (완전 구현).

여러 추출기를 묶어, 학습 분할로 한 번 ``fit`` 한 뒤 임의의 분할을 :class:`FeatureBundle`
로 변환한다. 모달리티별 가용성 마스크를 샘플의 :class:`ModalityMask` 로부터 구성하며,
선택적으로 :class:`FeatureCache` 를 사용해 추출 결과를 재사용한다.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, Self, runtime_checkable

import numpy as np

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.features import (
    FeatureBundle,
    FeatureMatrix,
    FeatureUnit,
    SequenceFeatureMatrix,
    UtteranceSpec,
)
from meld_emotion.core.protocols import FeatureCache, FeatureExtractor, SequenceFeatureExtractor
from meld_emotion.core.status import real
from meld_emotion.core.types import MODALITY_ORDER, BoolArray, FloatArray, Modality, Split
from meld_emotion.pipeline.cache import NullFeatureCache

logger = logging.getLogger(__name__)


class MediaLoader(Protocol):
    """원천 미디어 입력을 실제 배열로 적재하는 최소 계약."""

    def load_audio(self, audio: AudioInput) -> AudioInput: ...

    def load_video(self, video: VideoInput) -> VideoInput: ...


@runtime_checkable
class BundleFeatureCache(FeatureCache, Protocol):
    """FeatureBundle 단위 캐시까지 지원하는 확장 캐시."""

    def get_bundle(self, key: str) -> FeatureBundle | None: ...

    def put_bundle(self, key: str, bundle: FeatureBundle) -> None: ...


@real
class FeaturePipeline:
    """추출기 묶음 → 특징 묶음 변환기."""

    def __init__(
        self,
        extractors: Sequence[FeatureExtractor],
        cache: FeatureCache | None = None,
        media_loader: MediaLoader | None = None,
        media_error_policy: str = "raise",
        media_chunk_size: int = 16,
    ) -> None:
        if not extractors:
            raise ValueError("최소 한 개의 추출기가 필요합니다")
        if media_error_policy not in {"raise", "drop_modality", "drop_sample"}:
            raise ValueError(
                "media_error_policy 는 'raise', 'drop_modality', 'drop_sample' 중 하나여야 합니다"
            )
        if media_chunk_size <= 0:
            raise ValueError("media_chunk_size 는 양수여야 합니다")
        self._extractors = tuple(extractors)
        self._cache: FeatureCache = cache if cache is not None else NullFeatureCache()
        self._media_loader = media_loader
        self._media_error_policy = media_error_policy
        self._media_chunk_size = media_chunk_size
        self._needs_audio = any(_extractor_needs(e, Modality.AUDIO) for e in self._extractors)
        self._needs_video = any(_extractor_needs(e, Modality.VIDEO) for e in self._extractors)

    def fit(self, samples: Sequence[RawSample]) -> Self:
        logger.info("특징 파이프라인 fit 시작: samples=%d", len(samples))
        prepared = tuple(samples) if self._should_chunk_media() else self._prepare_media(samples)
        for extractor in self._extractors:
            logger.info("특징 추출기 학습 시작: extractor=%s", extractor.name)
            extractor.fit(prepared)
            logger.info("특징 추출기 학습 완료: extractor=%s", extractor.name)
        return self

    def transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        logger.info(
            "특징 파이프라인 transform 시작: split=%s samples=%d",
            split.value,
            len(samples),
        )
        if self._should_chunk_media():
            return self._transform_media_chunked(samples, split)
        prepared = self._prepare_media(samples)
        return self._transform_prepared(prepared, split)

    def fit_transform(self, samples: Sequence[RawSample], split: Split) -> FeatureBundle:
        logger.info(
            "특징 파이프라인 fit_transform 시작: split=%s samples=%d",
            split.value,
            len(samples),
        )
        if self._should_chunk_media():
            for extractor in self._extractors:
                logger.info("특징 추출기 학습 시작: extractor=%s", extractor.name)
                extractor.fit(samples)
                logger.info("특징 추출기 학습 완료: extractor=%s", extractor.name)
            return self._transform_media_chunked(samples, split)
        prepared = self._prepare_media(samples)
        for extractor in self._extractors:
            logger.info("특징 추출기 학습 시작: extractor=%s", extractor.name)
            extractor.fit(prepared)
            logger.info("특징 추출기 학습 완료: extractor=%s", extractor.name)
        return self._transform_prepared(prepared, split)

    def _transform_prepared(
        self, samples: Sequence[RawSample], split: Split
    ) -> FeatureBundle:
        matrices = []
        sequence_matrices: list[SequenceFeatureMatrix] = []
        for extractor in self._extractors:
            key = f"{extractor.name}|{split.value}|{_sample_digest(samples)}"
            cached = self._cache.get(key)
            if cached is not None and cached.n_samples == len(samples):
                logger.debug(
                    "특징 캐시 hit: key=%s samples=%d features=%d",
                    key,
                    cached.n_samples,
                    cached.n_features,
                )
                matrices.append(cached)
            else:
                if cached is not None:
                    logger.warning(
                        "특징 캐시 무시: key=%s cached_samples=%d current_samples=%d",
                        key,
                        cached.n_samples,
                        len(samples),
                    )
                logger.info("특징 변환 시작: extractor=%s split=%s", extractor.name, split.value)
                matrix = extractor.transform(samples)
                self._cache.put(key, matrix)
                logger.info(
                    "특징 변환 완료: extractor=%s samples=%d features=%d modality=%s kind=%s",
                    extractor.name,
                    matrix.n_samples,
                    matrix.n_features,
                    matrix.modality.value,
                    matrix.kind.value,
                )
                matrices.append(matrix)
            if isinstance(extractor, SequenceFeatureExtractor):
                logger.info(
                    "sequence 특징 변환 시작: extractor=%s split=%s",
                    extractor.name,
                    split.value,
                )
                sequence = extractor.transform_sequence(samples)
                _validate_sequence_matrix(sequence, extractor.name, len(samples))
                sequence_matrices.append(sequence)
                logger.info(
                    "sequence 특징 변환 완료: extractor=%s samples=%d length=%d features=%d",
                    extractor.name,
                    sequence.n_samples,
                    sequence.sequence_length,
                    sequence.n_features,
                )
        bundle = FeatureBundle(
            uids=tuple(s.uid for s in samples),
            matrices=tuple(matrices),
            sequence_matrices=tuple(sequence_matrices),
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
        logger.info(
            "특징 묶음 생성 완료: split=%s samples=%d matrices=%d",
            split.value,
            bundle.n_samples,
            len(bundle.matrices),
        )
        return bundle

    def _transform_media_chunked(
        self, samples: Sequence[RawSample], split: Split
    ) -> FeatureBundle:
        """미디어 배열을 chunk 안에서만 들고 최종 feature 행렬만 누적한다."""

        logger.info(
            "청크 미디어 특징 변환 시작: split=%s samples=%d chunk_size=%d",
            split.value,
            len(samples),
            self._media_chunk_size,
        )
        bundle_key = _bundle_cache_key(split, samples)
        cached_bundle = _get_cached_bundle(self._cache, bundle_key)
        if cached_bundle is not None:
            logger.info(
                "특징 bundle 캐시 hit: split=%s samples=%d matrices=%d",
                split.value,
                cached_bundle.n_samples,
                len(cached_bundle.matrices),
            )
            return cached_bundle

        prepared_light: list[RawSample] = []
        value_chunks: list[list[FloatArray]] = [[] for _ in self._extractors]
        specs: list[FeatureMatrix | None] = [None for _ in self._extractors]
        sequence_chunks: list[list[SequenceFeatureMatrix]] = [[] for _ in self._extractors]
        sequence_specs: list[SequenceFeatureMatrix | None] = [None for _ in self._extractors]

        total = len(samples)
        for start in range(0, total, self._media_chunk_size):
            chunk = samples[start : start + self._media_chunk_size]
            prepared_chunk: list[RawSample] = []
            for offset, sample in enumerate(chunk, start=1):
                index = start + offset
                current = self._prepare_sample_media(sample)
                if current is None:
                    _log_media_progress(index, total, len(prepared_light))
                    continue
                prepared_chunk.append(current)
                prepared_light.append(_without_loaded_media(current))
                _log_media_progress(index, total, len(prepared_light))

            if not prepared_chunk:
                continue

            for extractor_index, extractor in enumerate(self._extractors):
                logger.debug(
                    "청크 특징 변환 시작: extractor=%s split=%s chunk_samples=%d",
                    extractor.name,
                    split.value,
                    len(prepared_chunk),
                )
                matrix = extractor.transform(prepared_chunk)
                _validate_chunk_matrix(matrix, extractor.name, len(prepared_chunk))
                if specs[extractor_index] is None:
                    specs[extractor_index] = matrix
                else:
                    _validate_matrix_compatible(specs[extractor_index], matrix)
                value_chunks[extractor_index].append(matrix.values)

                if isinstance(extractor, SequenceFeatureExtractor):
                    sequence = extractor.transform_sequence(prepared_chunk)
                    _validate_sequence_matrix(sequence, extractor.name, len(prepared_chunk))
                    if sequence_specs[extractor_index] is None:
                        sequence_specs[extractor_index] = sequence
                    else:
                        _validate_sequence_compatible(sequence_specs[extractor_index], sequence)
                    sequence_chunks[extractor_index].append(sequence)

        if len(prepared_light) < len(samples):
            logger.warning(
                "미디어 준비 중 샘플 제외: raw_samples=%d kept_samples=%d",
                len(samples),
                len(prepared_light),
            )

        matrices: list[FeatureMatrix] = []
        sequence_matrices: list[SequenceFeatureMatrix] = []
        for extractor_index, extractor in enumerate(self._extractors):
            spec = specs[extractor_index]
            chunks = value_chunks[extractor_index]
            if spec is None:
                empty = extractor.transform(())
                spec = empty
                values = empty.values
            else:
                values = np.concatenate(chunks, axis=0)
            matrix = FeatureMatrix(
                values=np.asarray(values, dtype=np.float64),
                names=spec.names,
                modality=spec.modality,
                kind=spec.kind,
                source=spec.source,
            )
            self._cache.put(f"{extractor.name}|{split.value}|{_sample_digest(prepared_light)}", matrix)
            logger.info(
                "특징 변환 완료: extractor=%s samples=%d features=%d modality=%s kind=%s",
                extractor.name,
                matrix.n_samples,
                matrix.n_features,
                matrix.modality.value,
                matrix.kind.value,
            )
            matrices.append(matrix)

            if isinstance(extractor, SequenceFeatureExtractor):
                sequence = _concat_sequence_chunks(
                    sequence_chunks[extractor_index],
                    sequence_specs[extractor_index],
                    extractor,
                )
                sequence_matrices.append(sequence)
                logger.info(
                    "sequence 특징 변환 완료: extractor=%s samples=%d length=%d features=%d",
                    extractor.name,
                    sequence.n_samples,
                    sequence.sequence_length,
                    sequence.n_features,
                )

        bundle = self._bundle_from_prepared(
            tuple(prepared_light), tuple(matrices), split, tuple(sequence_matrices)
        )
        _put_cached_bundle(self._cache, bundle_key, bundle)
        logger.info("청크 미디어 특징 변환 완료: split=%s samples=%d", split.value, bundle.n_samples)
        return bundle

    def _prepare_media(self, samples: Sequence[RawSample]) -> tuple[RawSample, ...]:
        if self._media_loader is None or not (self._needs_audio or self._needs_video):
            return tuple(samples)

        logger.info(
            "미디어 준비 시작: samples=%d needs_audio=%s needs_video=%s",
            len(samples),
            self._needs_audio,
            self._needs_video,
        )
        prepared: list[RawSample] = []
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            current = self._prepare_sample_media(sample)
            if current is None:
                _log_media_progress(index, total, len(prepared))
                continue

            prepared.append(current)
            _log_media_progress(index, total, len(prepared))
        if len(prepared) < len(samples):
            logger.warning(
                "미디어 준비 중 샘플 제외: raw_samples=%d kept_samples=%d",
                len(samples),
                len(prepared),
            )
        logger.info("미디어 준비 완료: samples=%d", len(prepared))
        return tuple(prepared)

    def _prepare_sample_media(self, sample: RawSample) -> RawSample | None:
        if self._media_loader is None:
            return sample

        current = sample
        audio = current.audio
        if (
            self._needs_audio
            and current.has(Modality.AUDIO)
            and audio is not None
            and audio.waveform is None
            and audio.source_path is not None
        ):
            try:
                current = replace(current, audio=self._media_loader.load_audio(audio))
            except (FileNotFoundError, ValueError) as exc:
                handled = self._handle_media_error(current, Modality.AUDIO, exc)
                if handled is None:
                    return None
                current = handled

        video = current.video
        if (
            self._needs_video
            and current.has(Modality.VIDEO)
            and video is not None
            and video.frames is None
            and video.source_path is not None
        ):
            try:
                current = replace(current, video=self._media_loader.load_video(video))
            except (FileNotFoundError, ValueError) as exc:
                handled = self._handle_media_error(current, Modality.VIDEO, exc)
                if handled is None:
                    return None
                current = handled
        return current

    def _should_chunk_media(self) -> bool:
        return self._media_loader is not None and (self._needs_audio or self._needs_video)

    def _bundle_from_prepared(
        self,
        samples: Sequence[RawSample],
        matrices: tuple[FeatureMatrix, ...],
        split: Split,
        sequence_matrices: tuple[SequenceFeatureMatrix, ...] = (),
    ) -> FeatureBundle:
        bundle = FeatureBundle(
            uids=tuple(s.uid for s in samples),
            matrices=matrices,
            sequence_matrices=sequence_matrices,
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
        logger.info(
            "특징 묶음 생성 완료: split=%s samples=%d matrices=%d",
            split.value,
            bundle.n_samples,
            len(bundle.matrices),
        )
        return bundle

    def _handle_media_error(
        self, sample: RawSample, modality: Modality, exc: Exception
    ) -> RawSample | None:
        if self._media_error_policy == "raise":
            raise exc
        if self._media_error_policy == "drop_sample":
            message = (
                f"{sample.uid} 의 {modality.value} media 를 읽지 못해 샘플 전체를 제외합니다: {exc}"
            )
            logger.warning(message)
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        message = (
            f"{sample.uid} 의 {modality.value} media 를 읽지 못해 해당 모달리티를 누락 처리합니다: {exc}"
        )
        logger.warning(message)
        warnings.warn(
            message,
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
        availability = {
            modality: np.array([s.has(modality) for s in samples], dtype=np.bool_)
            for modality in MODALITY_ORDER
        }
        availability[Modality.MULTIMODAL] = np.array(
            [
                s.has(Modality.TEXT) or s.has(Modality.AUDIO) or s.has(Modality.VIDEO)
                for s in samples
            ],
            dtype=np.bool_,
        )
        return availability


def _log_media_progress(index: int, total: int, kept: int) -> None:
    if index == total or index % 500 == 0:
        logger.info(
            "미디어 준비 진행: processed=%d/%d kept=%d dropped=%d",
            index,
            total,
            kept,
            index - kept,
        )


def _without_loaded_media(sample: RawSample) -> RawSample:
    audio = (
        replace(sample.audio, waveform=None)
        if sample.audio is not None and sample.audio.waveform is not None
        else sample.audio
    )
    video = (
        replace(sample.video, frames=None)
        if sample.video is not None and sample.video.frames is not None
        else sample.video
    )
    return replace(sample, audio=audio, video=video)


def _validate_chunk_matrix(matrix: FeatureMatrix, extractor_name: str, n_samples: int) -> None:
    if matrix.n_samples != n_samples:
        raise ValueError(
            f"{extractor_name} 청크 출력 행 수가 sample 수와 일치하지 않습니다: "
            f"{matrix.n_samples} != {n_samples}"
        )


def _validate_matrix_compatible(reference: FeatureMatrix | None, matrix: FeatureMatrix) -> None:
    if reference is None:
        return
    if (
        matrix.names != reference.names
        or matrix.modality != reference.modality
        or matrix.kind != reference.kind
        or matrix.source != reference.source
    ):
        raise ValueError(
            "청크별 FeatureMatrix schema 가 일치하지 않습니다: "
            f"{reference.source} vs {matrix.source}"
        )


def _validate_sequence_matrix(
    matrix: SequenceFeatureMatrix, extractor_name: str, n_samples: int
) -> None:
    if matrix.n_samples != n_samples:
        raise ValueError(
            f"{extractor_name} sequence 출력 행 수가 sample 수와 일치하지 않습니다: "
            f"{matrix.n_samples} != {n_samples}"
        )


def _validate_sequence_compatible(
    reference: SequenceFeatureMatrix | None, matrix: SequenceFeatureMatrix
) -> None:
    if reference is None:
        return
    if (
        matrix.names != reference.names
        or matrix.modality != reference.modality
        or matrix.kind != reference.kind
        or matrix.source != reference.source
    ):
        raise ValueError(
            "청크별 SequenceFeatureMatrix schema 가 일치하지 않습니다: "
            f"{reference.source} vs {matrix.source}"
        )


def _concat_sequence_chunks(
    chunks: Sequence[SequenceFeatureMatrix],
    spec: SequenceFeatureMatrix | None,
    extractor: FeatureExtractor,
) -> SequenceFeatureMatrix:
    if not chunks:
        if isinstance(extractor, SequenceFeatureExtractor):
            return extractor.transform_sequence(())
        raise ValueError(f"{extractor.name} 는 sequence extractor 가 아닙니다")
    first = spec if spec is not None else chunks[0]
    total = sum(chunk.n_samples for chunk in chunks)
    max_len = max(chunk.sequence_length for chunk in chunks)
    values = np.zeros((total, max_len, first.n_features), dtype=np.float64)
    mask = np.zeros((total, max_len), dtype=bool)
    units: list[tuple[FeatureUnit, ...]] = []
    row = 0
    for chunk in chunks:
        end = row + chunk.n_samples
        values[row:end, : chunk.sequence_length] = chunk.values
        mask[row:end, : chunk.sequence_length] = chunk.mask
        units.extend(chunk.units)
        row = end
    return SequenceFeatureMatrix(
        values=values,
        mask=mask,
        units=tuple(units),
        names=first.names,
        modality=first.modality,
        kind=first.kind,
        source=first.source,
    )


def _bundle_cache_key(split: Split, samples: Sequence[RawSample]) -> str:
    return f"bundle|{split.value}|{len(samples)}|{_sample_digest(samples)}"


def _sample_digest(samples: Sequence[RawSample]) -> str:
    hasher = hashlib.sha1()
    for sample in samples:
        hasher.update(sample.uid.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(",".join(sorted(modality.value for modality in sample.mask.available)).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _extractor_needs(extractor: FeatureExtractor, modality: Modality) -> bool:
    if extractor.modality == modality:
        return True
    required = getattr(extractor, "required_modalities", ())
    return modality in required


def _get_cached_bundle(cache: FeatureCache, key: str) -> FeatureBundle | None:
    if isinstance(cache, BundleFeatureCache):
        return cache.get_bundle(key)
    return None


def _put_cached_bundle(cache: FeatureCache, key: str, bundle: FeatureBundle) -> None:
    if isinstance(cache, BundleFeatureCache):
        cache.put_bundle(key, bundle)
