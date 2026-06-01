"""문장 임베딩 텍스트 특징 (임시 기본 동작).

실제 구현은 sentence-transformers(예: all-MiniLM-L6-v2) 로 문장 임베딩을 계산한다.
현재는 토큰 해시를 ``dim`` 차원에 누적해 L2 정규화한 결정적 의사-임베딩을 반환한다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import note_placeholder_use, placeholder
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor
from meld_emotion.features.hashing import stable_hash

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@placeholder("sentence-transformers 로 실제 문장 임베딩을 계산해야 함")
class SentenceEmbeddingExtractor(BaseFeatureExtractor):
    """문장 임베딩(임시: 해싱 의사-임베딩)."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> None:
        self._model_name = model_name
        self._dim = dim

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"emb_{i}" for i in range(self._dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        note_placeholder_use(self)
        rows = [self._embed(s.text) for s in samples]
        return self._stack_rows(rows, self.names)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float64)
        for token in _TOKEN_RE.findall(text.lower()):
            h = stable_hash(token)
            vec[h % self._dim] += 1.0
            vec[(h // self._dim) % self._dim] += 0.5
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
