"""FeaturePipeline 의 변환과 캐시."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureBundle, FeatureUnit, SequenceFeatureMatrix
from meld_emotion.core.types import MODALITY_ORDER, FeatureKind, Modality, Split
from meld_emotion.features.base import BaseSequenceFeatureExtractor
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.pipeline.cache import InMemoryFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


class TinySequenceExtractor(BaseSequenceFeatureExtractor):
    modality = Modality.TEXT
    kind = FeatureKind.EMBEDDING

    @property
    def names(self) -> tuple[str, ...]:
        return ("s0", "s1")

    def transform_sequence(self, samples: Sequence[RawSample]) -> SequenceFeatureMatrix:
        values = np.zeros((len(samples), 3, 2), dtype=np.float64)
        mask = np.zeros((len(samples), 3), dtype=bool)
        units = []
        for row, sample in enumerate(samples):
            values[row, :2] = row + 1.0
            mask[row, :2] = True
            units.append((FeatureUnit(sample.text.split()[0], 0), FeatureUnit("tail", 1)))
        return self._sequence_matrix(values, mask, units, self.names)


def test_transform_produces_bundle(train_bundle: FeatureBundle, train_samples: list) -> None:
    assert train_bundle.n_samples == len(train_samples)
    assert len(train_bundle.matrices) == 4
    for modality in MODALITY_ORDER:
        assert train_bundle.availability[modality].shape == (len(train_samples),)


def test_cache_reuses_matrix(train_samples: list) -> None:
    cache = InMemoryFeatureCache()
    pipeline = FeaturePipeline([TextConceptExtractor(), BowTextExtractor(n_features=8)], cache)
    pipeline.fit(train_samples)
    first = pipeline.transform(train_samples, Split.TRAIN)
    second = pipeline.transform(train_samples, Split.TRAIN)
    # 캐시 히트 시 동일한 행렬 객체가 재사용되어야 한다.
    assert first.matrices[0] is second.matrices[0]


def test_sequence_extractor_is_added_to_bundle(train_samples: list) -> None:
    pipeline = FeaturePipeline([TinySequenceExtractor()])
    bundle = pipeline.fit_transform(train_samples[:4], Split.TRAIN)
    assert len(bundle.matrices) == 1
    assert len(bundle.sequence_matrices) == 1
    assert bundle.matrices[0].values.shape == (4, 2)
    assert bundle.sequence_matrices[0].values.shape == (4, 3, 2)
    assert bundle.sequence_matrices[0].mask[:, :2].all()


def test_requires_extractors() -> None:
    import pytest

    with pytest.raises(ValueError):
        FeaturePipeline([])
