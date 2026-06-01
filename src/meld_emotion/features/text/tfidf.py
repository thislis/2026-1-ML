"""TF-IDF 텍스트 임베딩 (임시 기본 동작).

실제 구현은 scikit-learn 의 ``TfidfVectorizer`` (학습 분할에서 어휘/IDF 학습) 로 대체한다.
현재는 하위 단계(융합/평가/설명) 테스트가 막히지 않도록, 해싱 기반의 결정적 수치 특징을
반환하는 임시 동작을 제공한다(차원은 작게 제한).
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
_MAX_PLACEHOLDER_DIM = 64


@placeholder("scikit-learn TfidfVectorizer 로 학습 분할에서 어휘·IDF 를 학습해야 함")
class TfidfTextExtractor(BaseFeatureExtractor):
    """TF-IDF 임베딩(임시: 해싱 빈도 특징)."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(self, max_features: int = 5000, ngram_max: int = 2) -> None:
        self._dim = min(max_features, _MAX_PLACEHOLDER_DIM)
        self._ngram_max = ngram_max

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"tfidf_{i}" for i in range(self._dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        note_placeholder_use(self)
        rows = [self._vectorize(s.text) for s in samples]
        return self._stack_rows(rows, self.names)

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float64)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            vec[stable_hash(token) % self._dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
