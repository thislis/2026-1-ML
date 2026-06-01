"""FeaturePipeline 의 변환과 캐시."""

from __future__ import annotations

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.types import MODALITY_ORDER, Split
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.pipeline.cache import InMemoryFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


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


def test_requires_extractors() -> None:
    import pytest

    with pytest.raises(ValueError):
        FeaturePipeline([])
