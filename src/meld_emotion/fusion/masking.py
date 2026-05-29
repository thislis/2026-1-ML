"""모달리티 시나리오와 마스킹 (완전 구현).

강건성 평가(모달리티 누락)와 학습 시 modality dropout 을 지원한다. 시나리오는 사용 가능한
모달리티 집합으로 정의되며, :func:`mask_bundle` 은 해당 시나리오에 맞춰 특징 묶음을
가공한다(없는 모달리티의 특징은 0 으로, 가용성은 False 로).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np

from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import BoolArray, Modality


@dataclass(frozen=True)
class ModalityScenario:
    """평가 시나리오: 어떤 모달리티가 사용 가능한지."""

    name: str
    available: frozenset[Modality]

    def has(self, modality: Modality) -> bool:
        return modality in self.available


_ALL = frozenset(Modality)

#: 미리 정의된 강건성 시나리오 (이름 → 시나리오).
SCENARIOS: dict[str, ModalityScenario] = {
    "full": ModalityScenario("full", _ALL),
    "text_only": ModalityScenario("text_only", frozenset({Modality.TEXT})),
    "audio_only": ModalityScenario("audio_only", frozenset({Modality.AUDIO})),
    "video_only": ModalityScenario("video_only", frozenset({Modality.VIDEO})),
    "no_text": ModalityScenario("no_text", _ALL - {Modality.TEXT}),
    "no_audio": ModalityScenario("no_audio", _ALL - {Modality.AUDIO}),
    "no_video": ModalityScenario("no_video", _ALL - {Modality.VIDEO}),
}


def get_scenario(name: str) -> ModalityScenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"알 수 없는 시나리오 '{name}'. 사용 가능: {sorted(SCENARIOS)}") from None


def _zeroed(matrix: FeatureMatrix) -> FeatureMatrix:
    return FeatureMatrix(
        values=np.zeros_like(matrix.values),
        names=matrix.names,
        modality=matrix.modality,
        kind=matrix.kind,
        source=matrix.source,
    )


def mask_bundle(bundle: FeatureBundle, scenario: ModalityScenario) -> FeatureBundle:
    """시나리오에 맞춰 특징 묶음을 마스킹한 새 묶음을 반환한다."""

    n = bundle.n_samples
    matrices = tuple(m if scenario.has(m.modality) else _zeroed(m) for m in bundle.matrices)
    availability: dict[Modality, BoolArray] = {}
    for modality, avail in bundle.availability.items():
        if scenario.has(modality):
            availability[modality] = avail
        else:
            availability[modality] = np.zeros(n, dtype=np.bool_)
    return FeatureBundle(uids=bundle.uids, matrices=matrices, availability=availability)


@real
class ModalityDropout:
    """학습 시 데이터 증강: 각 (샘플, 모달리티)를 확률 p 로 가린다.

    제안서의 modality dropout 에 해당한다. 가린 위치의 특징 행은 0, 가용성은 False 가 된다.
    """

    def __init__(self, drop_prob: float = 0.3, seed: int = 0) -> None:
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob 는 [0,1] 범위여야 합니다")
        self._p = drop_prob
        self._seed = seed

    def fit(self, bundle: FeatureBundle) -> Self:
        return self

    def apply(self, bundle: FeatureBundle) -> FeatureBundle:
        if self._p == 0.0:
            return bundle
        rng = np.random.default_rng(self._seed)
        n = bundle.n_samples
        # 모달리티별 드롭 마스크 (True = 가린다)
        drop: dict[Modality, BoolArray] = {
            mod: rng.random(n) < self._p for mod in bundle.modalities
        }
        matrices = tuple(self._apply_to_matrix(m, drop.get(m.modality)) for m in bundle.matrices)
        availability = self._updated_availability(bundle.availability, drop, n)
        return FeatureBundle(uids=bundle.uids, matrices=matrices, availability=availability)

    @staticmethod
    def _apply_to_matrix(matrix: FeatureMatrix, drop_mask: BoolArray | None) -> FeatureMatrix:
        if drop_mask is None:
            return matrix
        values = matrix.values.copy()
        values[drop_mask] = 0.0
        return FeatureMatrix(
            values=values,
            names=matrix.names,
            modality=matrix.modality,
            kind=matrix.kind,
            source=matrix.source,
        )

    @staticmethod
    def _updated_availability(
        availability: Mapping[Modality, BoolArray],
        drop: Mapping[Modality, BoolArray],
        n: int,
    ) -> dict[Modality, BoolArray]:
        result: dict[Modality, BoolArray] = {}
        for modality, avail in availability.items():
            drop_mask = drop.get(modality)
            if drop_mask is None:
                result[modality] = avail
            else:
                result[modality] = avail & ~drop_mask
        return result
