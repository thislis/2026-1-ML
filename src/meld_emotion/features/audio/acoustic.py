"""MFCC 음향 임베딩 (임시 기본 동작).

실제 구현은 librosa 로 MFCC 통계(평균/표준편차), 스펙트럴 특징, 피치 등을 계산한다.
현재는 numpy FFT 로 얻은 결정적 로그-스펙트럼 요약을 ``2*n_mfcc`` 차원으로 반환한다.
오디오가 없으면 0 벡터.
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


@placeholder("librosa 로 실제 MFCC/스펙트럴/피치 통계를 계산해야 함")
class MfccAcousticExtractor(BaseFeatureExtractor):
    """MFCC 음향 임베딩(임시: FFT 로그-스펙트럼 요약)."""

    modality: ClassVar[Modality] = Modality.AUDIO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(self, n_mfcc: int = 13) -> None:
        self._n_mfcc = n_mfcc

    @property
    def names(self) -> tuple[str, ...]:
        means = [f"mfcc_mean_{i}" for i in range(self._n_mfcc)]
        stds = [f"mfcc_std_{i}" for i in range(self._n_mfcc)]
        return tuple(means + stds)

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        note_placeholder_use(self)
        rows = [self._features(s) for s in samples]
        return self._stack_rows(rows, self.names)

    def _features(self, sample: RawSample) -> np.ndarray:
        dim = 2 * self._n_mfcc
        if sample.audio is None or sample.audio.waveform is None:
            return np.zeros(dim, dtype=np.float64)
        wave = np.asarray(sample.audio.waveform, dtype=np.float64)
        if wave.size < 2:
            return np.zeros(dim, dtype=np.float64)
        spectrum = np.abs(np.fft.rfft(wave))
        log_spec = np.log1p(spectrum)
        # 로그-스펙트럼을 n_mfcc 구간으로 나누어 평균/표준편차 요약.
        bins = np.array_split(log_spec, self._n_mfcc)
        means = np.array([float(np.mean(b)) if b.size else 0.0 for b in bins])
        stds = np.array([float(np.std(b)) if b.size else 0.0 for b in bins])
        return np.concatenate([means, stds]).astype(np.float64)
