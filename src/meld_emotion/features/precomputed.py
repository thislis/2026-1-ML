"""MELD 팀이 제공한 precomputed feature pickle 추출기."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, FloatArray, Modality, Split

_SPLIT_INDEX = {Split.TRAIN: 0, Split.DEV: 1, Split.TEST: 2}


@real
class MeldPrecomputedFeatureExtractor:
    """MELD baseline pickle feature 를 `FeatureMatrix` 로 변환한다."""

    def __init__(
        self,
        path: str,
        modality: Modality,
        kind: FeatureKind = FeatureKind.EMBEDDING,
        name_prefix: str = "",
    ) -> None:
        if not path:
            raise ValueError("precomputed feature path 가 필요합니다")
        self._path = Path(path)
        self._modality = modality
        self._kind = kind
        self._name_prefix = name_prefix
        self._loaded: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None
        self._dims: dict[Split, int] = {}

    @property
    def name(self) -> str:
        prefix = f"{self._name_prefix}_" if self._name_prefix else ""
        return f"{prefix}meld_{self._modality.value}_{self._path.stem}"

    @property
    def modality(self) -> Modality:
        return self._modality

    @property
    def kind(self) -> FeatureKind:
        return self._kind

    def fit(self, samples: Sequence[RawSample]) -> Self:
        self._ensure_loaded()
        return self

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        split = _split_for(samples)
        rows = [self._vector_for(sample, split) for sample in samples]
        if rows:
            values = np.vstack(rows).astype(np.float64)
            dim = int(values.shape[1])
            self._dims[split] = dim
        else:
            dim = self._dim_for(split)
            values = np.zeros((0, dim), dtype=np.float64)
        return FeatureMatrix(
            values=values,
            names=tuple(f"{self.name}_{i}" for i in range(dim)),
            modality=self._modality,
            kind=self._kind,
            source=self.name,
        )

    def _vector_for(self, sample: RawSample, split: Split) -> FloatArray:
        data = self._split_data(split)
        utterance_key = f"{sample.dialogue_id}_{sample.utterance_id}"
        if utterance_key in data:
            return _as_vector(data[utterance_key], self._path, utterance_key)

        dialogue_key = str(sample.dialogue_id)
        if dialogue_key in data:
            matrix = np.asarray(data[dialogue_key], dtype=np.float64)
            if matrix.ndim != 2:
                raise ValueError(
                    f"{self._path} dialogue feature 는 2차원이어야 합니다: {dialogue_key}"
                )
            if sample.utterance_id >= matrix.shape[0]:
                raise ValueError(
                    f"{self._path} 에 {utterance_key} row 가 없습니다 "
                    f"(dialogue rows={matrix.shape[0]})"
                )
            return np.asarray(matrix[sample.utterance_id], dtype=np.float64)

        raise ValueError(f"{self._path} 에 feature key 가 없습니다: {utterance_key}")

    def _split_data(self, split: Split) -> Mapping[str, Any]:
        loaded = self._ensure_loaded()
        return loaded[_SPLIT_INDEX[split]]

    def _ensure_loaded(
        self,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        if self._loaded is None:
            with self._path.open("rb") as f:
                raw = pickle.load(f, encoding="latin1")
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError(f"MELD feature pickle 은 split 3개를 담아야 합니다: {self._path}")
            splits = tuple(_expect_mapping(part, self._path) for part in raw)
            self._loaded = (splits[0], splits[1], splits[2])
        return self._loaded

    def _dim_for(self, split: Split) -> int:
        if split in self._dims:
            return self._dims[split]
        data = self._split_data(split)
        for value in data.values():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1:
                dim = int(arr.shape[0])
            elif arr.ndim == 2:
                dim = int(arr.shape[1])
            else:
                continue
            self._dims[split] = dim
            return dim
        raise ValueError(f"{self._path} 의 {split.value} split 에 feature 가 없습니다")


def _split_for(samples: Sequence[RawSample]) -> Split:
    if not samples:
        return Split.TRAIN
    split = samples[0].split
    if any(sample.split != split for sample in samples):
        raise ValueError("precomputed feature transform 은 단일 split 샘플만 지원합니다")
    return split


def _expect_mapping(value: object, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MELD feature split 은 mapping 이어야 합니다: {path}")
    return value


def _as_vector(value: Any, path: Path, key: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{path} feature 는 1차원 벡터여야 합니다: {key}, shape={arr.shape}")
    return arr
