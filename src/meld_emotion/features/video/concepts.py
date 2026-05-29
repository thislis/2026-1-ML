"""비디오 개념 특징 추출기 (완전 구현, numpy 전용).

제안서의 해석 가능한 비디오 개념 c_V 를 프레임 텐서(T,H,W,C)로부터 직접 계산한다: 평균/표준
편차 밝기, 프레임 간 차이(움직임 대용), 최대 움직임, 밝기 범위. 비디오가 없으면 0 벡터.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor


@real
class VideoConceptExtractor(BaseFeatureExtractor):
    """해석 가능한 비디오 개념 벡터 c_V."""

    modality: ClassVar[Modality] = Modality.VIDEO
    kind: ClassVar[FeatureKind] = FeatureKind.CONCEPT
    feature_names: ClassVar[tuple[str, ...]] = (
        "mean_intensity",
        "std_intensity",
        "mean_frame_diff",
        "max_frame_diff",
        "intensity_range",
    )

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        rows = [self._features(s) for s in samples]
        return self._stack_rows(rows, self.feature_names)

    def _features(self, sample: RawSample) -> np.ndarray:
        if sample.video is None or sample.video.frames is None:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        frames = np.asarray(sample.video.frames, dtype=np.float64)
        if frames.size == 0:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        if frames.shape[0] >= 2:
            diffs = np.abs(np.diff(frames, axis=0))
            mean_diff = float(np.mean(diffs))
            max_diff = float(np.max(diffs))
        else:
            mean_diff = 0.0
            max_diff = 0.0
        return np.array(
            [
                float(np.mean(frames)),
                float(np.std(frames)),
                mean_diff,
                max_diff,
                float(np.max(frames) - np.min(frames)),
            ],
            dtype=np.float64,
        )
