"""감정 레이블 인코더."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import EMOTION_ORDER, Emotion, IntArray


@real
class EmotionLabelEncoder:
    """프로젝트 표준 감정 순서(`EMOTION_ORDER`)에 따른 레이블 인코더."""

    def __init__(self, classes: Sequence[Emotion] = EMOTION_ORDER) -> None:
        self._classes = tuple(classes)
        self._to_index = {label: i for i, label in enumerate(self._classes)}

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def encode(self, labels: Sequence[Emotion]) -> IntArray:
        try:
            values = [self._to_index[Emotion(label)] for label in labels]
        except ValueError as exc:
            raise ValueError(f"알 수 없는 감정 레이블입니다: {exc}") from exc
        except KeyError as exc:
            raise ValueError(f"인코더 클래스에 없는 감정 레이블입니다: {exc}") from exc
        return np.asarray(values, dtype=np.int64)

    def decode(self, indices: IntArray) -> tuple[Emotion, ...]:
        decoded: list[Emotion] = []
        for index in np.asarray(indices, dtype=np.int64).tolist():
            if index < 0 or index >= len(self._classes):
                raise ValueError(f"클래스 인덱스 범위를 벗어났습니다: {index}")
            decoded.append(self._classes[int(index)])
        return tuple(decoded)
