"""MELD metadata 와 precomputed feature baseline."""

from __future__ import annotations

from pathlib import Path

from meld_emotion.config.loader import load_config
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    MeldConfig,
    NearestCentroidConfig,
    NullCacheConfig,
    PrecomputedMeldFeatureConfig,
)
from meld_emotion.core.types import EMOTION_ORDER, Modality, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.meld import MeldDatasetSource
from meld_emotion.features.precomputed import MeldPrecomputedFeatureExtractor
from meld_emotion.pipeline.builder import build_experiment, build_extractor
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _ROOT / "MELD.Features.Models" / "features"


def test_label_encoder_uses_project_emotion_order() -> None:
    encoder = EmotionLabelEncoder()
    encoded = encoder.encode(list(EMOTION_ORDER))
    assert encoded.tolist() == list(range(len(EMOTION_ORDER)))
    assert encoder.decode(encoded) == EMOTION_ORDER


def test_meld_metadata_source_sizes() -> None:
    source = MeldDatasetSource(metadata_path=str(_FEATURES / "data_emotion.p"))
    assert len(list(source.load(Split.TRAIN))) == 9989
    assert len(list(source.load(Split.DEV))) == 1109
    assert len(list(source.load(Split.TEST))) == 2610


def test_precomputed_utterance_keyed_feature_shape() -> None:
    samples = list(MeldDatasetSource(metadata_path=str(_FEATURES / "data_emotion.p")).load(Split.DEV))
    extractor = MeldPrecomputedFeatureExtractor(
        path=str(_FEATURES / "text_glove_average_emotion.pkl"),
        modality=Modality.TEXT,
    )
    matrix = extractor.transform(samples[:5])
    assert matrix.values.shape == (5, 300)
    assert matrix.modality == Modality.TEXT


def test_precomputed_dialogue_matrix_feature_shape() -> None:
    samples = list(MeldDatasetSource(metadata_path=str(_FEATURES / "data_emotion.p")).load(Split.DEV))
    extractor = MeldPrecomputedFeatureExtractor(
        path=str(_FEATURES / "text_emotion.pkl"),
        modality=Modality.TEXT,
    )
    matrix = extractor.transform(samples[:5])
    assert matrix.values.shape == (5, 600)


def test_precomputed_extractors_have_distinct_cache_keys() -> None:
    source = MeldDatasetSource(metadata_path=str(_FEATURES / "data_emotion.p"))
    samples = list(source.load(Split.DEV))[:4]
    pipeline = FeaturePipeline(
        [
            MeldPrecomputedFeatureExtractor(
                path=str(_FEATURES / "text_glove_average_emotion.pkl"),
                modality=Modality.TEXT,
            ),
            MeldPrecomputedFeatureExtractor(
                path=str(_FEATURES / "text_glove_CNN_emotion.pkl"),
                modality=Modality.TEXT,
            ),
        ]
    )
    bundle = pipeline.fit_transform(samples, Split.DEV)
    assert bundle.matrices[0].source != bundle.matrices[1].source
    assert bundle.matrices[0].values.shape[1] == 300
    assert bundle.matrices[1].values.shape[1] == 100


def test_builder_creates_precomputed_extractor() -> None:
    extractor = build_extractor(
        PrecomputedMeldFeatureConfig(
            path=str(_FEATURES / "text_glove_average_emotion.pkl"),
            modality="text",
        )
    )
    assert isinstance(extractor, MeldPrecomputedFeatureExtractor)


def test_meld_precomputed_smoke_e2e_centroid() -> None:
    config = ExperimentConfig(
        name="meld_precomputed_smoke",
        train_split="dev",
        eval_split="dev",
        dataset=MeldConfig(metadata_path=str(_FEATURES / "data_emotion.p")),
        extractors=(
            PrecomputedMeldFeatureConfig(
                path=str(_FEATURES / "text_glove_average_emotion.pkl"),
                modality="text",
            ),
            PrecomputedMeldFeatureConfig(
                path=str(_FEATURES / "audio_embeddings_feature_selection_emotion.pkl"),
                modality="audio",
            ),
        ),
        model=EarlyFusionConfig(base=NearestCentroidConfig()),
        evaluation=EvaluationConfig(metrics=("accuracy",), confusion=False, scenarios=()),
        cache=NullCacheConfig(),
        reporters=(),
    )
    result = build_experiment(config).run()
    assert result.evaluation.metric("accuracy") is not None


def test_example_meld_precomputed_config_loads() -> None:
    config = load_config(_ROOT / "configs" / "example_meld_precomputed_svm.yaml")
    assert isinstance(config.dataset, MeldConfig)
    assert len(config.extractors) == 2
