"""도메인 열거형과 배열 타입 별칭.

파이프라인 전반에서 공유되는 가장 기본적인 어휘를 정의한다. 다른 어떤 내부 모듈에도
의존하지 않으므로 의존성 그래프의 최하단에 위치한다.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# --- 배열 타입 별칭 -------------------------------------------------------------
# numpy 배열에 의미를 부여하기 위한 별칭. 런타임 동작은 동일하지만 정적 분석 시
# "이 배열이 무엇을 담는가"를 드러낸다.
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

#: 발화(utterance) 하나를 가리키는 전역 고유 식별자.
UID: TypeAlias = str


class Modality(StrEnum):
    """입력 모달리티 종류."""

    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


class Emotion(StrEnum):
    """MELD 의 7개 감정 레이블. 열거 순서가 곧 클래스 인덱스 순서이다."""

    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"


class Sentiment(StrEnum):
    """MELD 의 3개 감성 레이블 (보조 레이블)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Split(StrEnum):
    """데이터 분할."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class FeatureKind(StrEnum):
    """특징의 성격.

    EMBEDDING: 예측 성능을 위한 고차원 특징 (예: TF-IDF, 문장 임베딩, MFCC).
    CONCEPT:   해석 가능한 저차원 보조 특징. 제안서의 개념 벡터 c=[c_T,c_A,c_V] 를 구성.
    """

    EMBEDDING = "embedding"
    CONCEPT = "concept"


#: 감정 클래스의 표준 순서 (레이블 인코딩/혼동행렬 축에 사용).
EMOTION_ORDER: tuple[Emotion, ...] = tuple(Emotion)

#: 모달리티의 표준 순서 (개념 벡터 [c_T, c_A, c_V] 결합 순서에 사용).
MODALITY_ORDER: tuple[Modality, ...] = (Modality.TEXT, Modality.AUDIO, Modality.VIDEO)
