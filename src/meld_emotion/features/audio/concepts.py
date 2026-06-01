"""오디오 개념 특징 추출기 (완전 구현, numpy 전용).

제안서의 해석 가능한 오디오 개념 c_A 를 파형 배열로부터 직접 계산한다: RMS 에너지, 평균
절댓값, 표준편차, 영교차율, 최대 진폭, 무음 비율. 오디오가 없는 샘플은 0 벡터를 반환한다
(가용성은 파이프라인 마스크가 별도 관리).
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

_SILENCE_THRESHOLD = 0.05


@real
class AudioConceptExtractor(BaseFeatureExtractor):
    """해석 가능한 오디오 개념 벡터 c_A."""

    modality: ClassVar[Modality] = Modality.AUDIO
    kind: ClassVar[FeatureKind] = FeatureKind.CONCEPT
    feature_names: ClassVar[tuple[str, ...]] = (
        "rms_energy",
        "mean_abs",
        "std",
        "zero_crossing_rate",
        "max_abs",
        "silence_ratio",
    )

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        rows = [self._features(s) for s in samples]
        return self._stack_rows(rows, self.feature_names)

    def _features(self, sample: RawSample) -> np.ndarray:
        if sample.audio is None or sample.audio.waveform is None:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        wave = np.asarray(sample.audio.waveform, dtype=np.float64)
        if wave.size == 0:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        abs_wave = np.abs(wave)
        zero_crossings = np.count_nonzero(np.diff(np.signbit(wave)))
        return np.array(
            [
                float(np.sqrt(np.mean(wave**2))),
                float(np.mean(abs_wave)),
                float(np.std(wave)),
                zero_crossings / float(wave.size),
                float(np.max(abs_wave)),
                float(np.mean(abs_wave < _SILENCE_THRESHOLD)),
            ],
            dtype=np.float64,
        )
