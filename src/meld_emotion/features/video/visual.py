"""시각 단서 임베딩 (임시 기본 동작).

실제 구현은 opencv/mediapipe 로 얼굴 검출 비율, 랜드마크 이동, 입 벌림 대용, 얼굴 움직임
크기 등을 계산한다. 현재는 프레임 텐서의 채널별/풀링 통계를 ``dim`` 차원으로 요약한 결정적
특징을 반환한다. 비디오가 없으면 0 벡터.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import note_placeholder_use, placeholder
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor


@placeholder("opencv/mediapipe 로 얼굴 검출·랜드마크·움직임 단서를 계산해야 함")
class VisualCueExtractor(BaseFeatureExtractor):
    """시각 단서 임베딩(임시: 프레임 통계 요약)."""

    modality: ClassVar[Modality] = Modality.VIDEO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"visual_{i}" for i in range(self._dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        note_placeholder_use(self)
        rows = [self._features(s) for s in samples]
        return self._stack_rows(rows, self.names)

    def _features(self, sample: RawSample) -> np.ndarray:
        if sample.video is None or sample.video.frames is None:
            return np.zeros(self._dim, dtype=np.float64)
        frames = np.asarray(sample.video.frames, dtype=np.float64)
        if frames.size == 0:
            return np.zeros(self._dim, dtype=np.float64)
        # 공간 차원을 평균 풀링하여 (T, C) 로 줄인 뒤 시간축 통계를 모은다.
        pooled = (
            frames.mean(axis=(1, 2)) if frames.ndim == 4 else frames.reshape(frames.shape[0], -1)
        )
        summary = np.concatenate(
            [pooled.mean(axis=0), pooled.std(axis=0), pooled.max(axis=0), pooled.min(axis=0)]
        )
        vec = np.zeros(self._dim, dtype=np.float64)
        n = min(self._dim, summary.size)
        vec[:n] = summary[:n]
        return vec
