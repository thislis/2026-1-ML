"""텍스트 개념 추출용 소형 어휘 사전.

해석 가능한 개념(긍/부정 비율, 부정어 수, 감정 키워드 수)을 계산하기 위한 최소 사전이다.
실제 실험에서는 더 큰 감성 사전(VADER/NRC 등)으로 교체할 수 있다.
"""

from __future__ import annotations

POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "happy",
        "great",
        "love",
        "wonderful",
        "good",
        "nice",
        "glad",
        "excited",
        "fun",
        "awesome",
        "amazing",
        "best",
        "thanks",
    }
)

NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "sad",
        "sorry",
        "cry",
        "lonely",
        "angry",
        "hate",
        "furious",
        "mad",
        "scared",
        "afraid",
        "terrified",
        "nervous",
        "disgusting",
        "gross",
        "awful",
        "ugh",
        "bad",
        "worst",
        "terrible",
    }
)

NEGATION_WORDS: frozenset[str] = frozenset(
    {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "n't"}
)

EMOTION_KEYWORDS: frozenset[str] = POSITIVE_WORDS | NEGATIVE_WORDS
