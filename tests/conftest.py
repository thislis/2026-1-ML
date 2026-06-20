"""공용 pytest 픽스처."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.types import MODALITY_ORDER, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.features.audio import AudioConceptExtractor
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor

if TYPE_CHECKING:
    from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

_NATIVE_TEST_MARKERS = {
    "test_xgboost_estimator.py": "xgboost_native",
    "test_catboost_estimator.py": "catboost_native",
}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    marker = _NATIVE_TEST_MARKERS.get(collection_path.name)
    markexpr = str(config.getoption("-m") or "")
    requested_native = {
        native_marker
        for native_marker in _NATIVE_TEST_MARKERS.values()
        if native_marker in markexpr
    }
    if requested_native:
        return marker not in requested_native
    if marker is None:
        return False
    return marker not in markexpr


@pytest.fixture
def source() -> SyntheticDatasetSource:
    return SyntheticDatasetSource(n_train=140, n_dev=28, n_test=70, seed=7)


@pytest.fixture
def train_samples(source: SyntheticDatasetSource) -> list:
    return list(source.load(Split.TRAIN))


@pytest.fixture
def test_samples(source: SyntheticDatasetSource) -> list:
    return list(source.load(Split.TEST))


@pytest.fixture
def encoder() -> EmotionLabelEncoder:
    return EmotionLabelEncoder()


@pytest.fixture
def pipeline() -> FeaturePipeline:
    from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

    return FeaturePipeline(
        [
            TextConceptExtractor(),
            BowTextExtractor(n_features=32),
            AudioConceptExtractor(),
            VideoConceptExtractor(),
        ]
    )


@pytest.fixture
def train_bundle(pipeline: FeaturePipeline, train_samples: list) -> FeatureBundle:
    return pipeline.fit_transform(train_samples, Split.TRAIN)


@pytest.fixture
def test_bundle(
    pipeline: FeaturePipeline, train_samples: list, test_samples: list
) -> FeatureBundle:
    pipeline.fit(train_samples)
    return pipeline.transform(test_samples, Split.TEST)


@pytest.fixture
def y_train(encoder: EmotionLabelEncoder, train_samples: list) -> np.ndarray:
    return encoder.encode([s.emotion for s in train_samples])


@pytest.fixture
def y_test(encoder: EmotionLabelEncoder, test_samples: list) -> np.ndarray:
    return encoder.encode([s.emotion for s in test_samples])


def all_modalities() -> tuple:
    return MODALITY_ORDER
