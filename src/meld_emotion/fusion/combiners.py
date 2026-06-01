"""Late fusion 확률 결합기 (mean/weighted 완전 구현, stacking 임시).

각 모달리티 분류기가 낸 확률을 하나로 합친다. 결합 시 모달리티별 **가용성**을 가중치에
곱해, 누락된 모달리티는 자동으로 기여에서 제외된다(강건성 평가와 자연스럽게 연동).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, runtime_checkable

import numpy as np

from meld_emotion.core.status import note_placeholder_use, placeholder, real
from meld_emotion.core.types import BoolArray, FloatArray, Modality


@runtime_checkable
class ProbabilityCombiner(Protocol):
    """모달리티별 확률을 하나의 확률 행렬로 결합한다."""

    def fit(self, per_modality: Mapping[Modality, FloatArray], y: np.ndarray) -> Self: ...

    def combine(
        self,
        per_modality: Mapping[Modality, FloatArray],
        availability: Mapping[Modality, BoolArray],
    ) -> FloatArray: ...


def _weighted_average(
    per_modality: Mapping[Modality, FloatArray],
    availability: Mapping[Modality, BoolArray],
    weights: Mapping[Modality, float],
) -> FloatArray:
    modalities = list(per_modality)
    if not modalities:
        raise ValueError("결합할 모달리티 확률이 없습니다")
    n, k = per_modality[modalities[0]].shape
    acc = np.zeros((n, k), dtype=np.float64)
    weight_sum = np.zeros((n, 1), dtype=np.float64)
    for modality in modalities:
        proba = per_modality[modality]
        avail = availability.get(modality, np.ones(n, dtype=np.bool_)).astype(np.float64)
        w = (weights.get(modality, 1.0) * avail).reshape(n, 1)
        acc += w * proba
        weight_sum += w
    # 가용 모달리티가 하나도 없는 샘플은 균등 분포로.
    zero_rows = weight_sum[:, 0] == 0.0
    weight_sum[zero_rows] = 1.0
    combined = acc / weight_sum
    combined[zero_rows] = 1.0 / k
    return combined


@real
class MeanCombiner:
    """가용성 가중 평균 (모든 모달리티 동일 가중)."""

    def fit(self, per_modality: Mapping[Modality, FloatArray], y: np.ndarray) -> Self:
        return self

    def combine(
        self,
        per_modality: Mapping[Modality, FloatArray],
        availability: Mapping[Modality, BoolArray],
    ) -> FloatArray:
        weights = dict.fromkeys(per_modality, 1.0)
        return _weighted_average(per_modality, availability, weights)


@real
class WeightedCombiner:
    """모달리티별 고정 가중 평균."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self._weights = {Modality(k): float(v) for k, v in (weights or {}).items()}

    def fit(self, per_modality: Mapping[Modality, FloatArray], y: np.ndarray) -> Self:
        return self

    def combine(
        self,
        per_modality: Mapping[Modality, FloatArray],
        availability: Mapping[Modality, BoolArray],
    ) -> FloatArray:
        weights = {m: self._weights.get(m, 1.0) for m in per_modality}
        return _weighted_average(per_modality, availability, weights)


@placeholder(
    "모달리티별 확률을 입력으로 하는 메타 학습기(stacking) 학습 필요 — 현재는 평균으로 대체"
)
class StackingCombiner:
    """스태킹 결합기(임시: 평균으로 대체)."""

    def fit(self, per_modality: Mapping[Modality, FloatArray], y: np.ndarray) -> Self:
        return self

    def combine(
        self,
        per_modality: Mapping[Modality, FloatArray],
        availability: Mapping[Modality, BoolArray],
    ) -> FloatArray:
        note_placeholder_use(self)
        weights = dict.fromkeys(per_modality, 1.0)
        return _weighted_average(per_modality, availability, weights)
