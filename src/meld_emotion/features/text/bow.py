"""해싱 기반 Bag-of-Words 텍스트 임베딩 (완전 구현, numpy 전용).

sklearn 없이 동작하는 예측용 텍스트 임베딩. 토큰을 해시로 고정 차원 버킷에 누적한다.
어휘 학습이 필요 없어 ``fit`` 은 no-op 이며 fit/transform 결과가 결정적이다. TF-IDF 의
경량 대체재로, 합성 데이터에서도 학습 가능한 신호를 제공한다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor
from meld_emotion.features.hashing import stable_hash

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@real
class BowTextExtractor(BaseFeatureExtractor):
    """고정 차원 해싱 Bag-of-Words."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(self, n_features: int = 256, lowercase: bool = True) -> None:
        if n_features <= 0:
            raise ValueError("n_features 는 양수여야 합니다")
        self._n_features = n_features
        self._lowercase = lowercase

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"bow_{i}" for i in range(self._n_features))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        rows = [self._vectorize(s.text) for s in samples]
        return self._stack_rows(rows, self.names)

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._n_features, dtype=np.float64)
        for token in _TOKEN_RE.findall(text):
            key = token.lower() if self._lowercase else token
            bucket = stable_hash(key) % self._n_features
            vec[bucket] += 1.0
        return vec
