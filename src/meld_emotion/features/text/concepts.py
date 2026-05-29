"""텍스트 개념 특징 추출기 (완전 구현).

제안서의 해석 가능한 텍스트 개념 c_T 를 계산한다: 길이, 문장부호, 대문자 비율, 부정어 수,
긍/부정 단어 비율, 감정 키워드 수. 무거운 라이브러리 없이 순수 파이썬으로 구현되어 단위
테스트가 쉽다.
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
from meld_emotion.features.text.lexicon import (
    NEGATION_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
)

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@real
class TextConceptExtractor(BaseFeatureExtractor):
    """해석 가능한 텍스트 개념 벡터 c_T."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.CONCEPT
    feature_names: ClassVar[tuple[str, ...]] = (
        "n_tokens",
        "n_chars",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "negation_count",
        "positive_ratio",
        "negative_ratio",
        "emotion_keyword_count",
    )

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        rows = [self._features(s.text) for s in samples]
        return self._stack_rows(rows, self.feature_names)

    def _features(self, text: str) -> np.ndarray:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        n_tokens = len(tokens)
        denom = float(n_tokens) if n_tokens else 1.0
        positive = sum(t in POSITIVE_WORDS for t in tokens)
        negative = sum(t in NEGATIVE_WORDS for t in tokens)
        negation = sum(t in NEGATION_WORDS for t in tokens)
        emotion_kw = positive + negative
        uppercase = sum(1 for c in text if c.isupper())
        return np.array(
            [
                float(n_tokens),
                float(len(text)),
                float(text.count("!")),
                float(text.count("?")),
                uppercase / float(len(text)) if text else 0.0,
                float(negation),
                positive / denom,
                negative / denom,
                float(emotion_kw),
            ],
            dtype=np.float64,
        )
