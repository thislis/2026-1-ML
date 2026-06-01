"""공용 pytest 픽스처."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.types import MODALITY_ORDER, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.features.audio import AudioConceptExtractor
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


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
