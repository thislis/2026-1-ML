"""구체 컴포넌트가 해당 Protocol 을 만족하는지 검증 (LSP 가드)."""

from __future__ import annotations

from meld_emotion.core.protocols import (
    Classifier,
    DatasetSource,
    Estimator,
    Explainer,
    FeatureCache,
    FeatureExtractor,
    LabelEncoder,
    Metric,
    Reporter,
)
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.evaluation.metrics import AccuracyMetric, MacroF1Metric
from meld_emotion.explain.modality_contribution import ModalityAblationExplainer
from meld_emotion.explain.permutation import PermutationImportanceExplainer
from meld_emotion.features.audio import AudioConceptExtractor, MfccAcousticExtractor
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor, VisualCueExtractor
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.late import LateFusionClassifier
from meld_emotion.models.baselines import (
    MajorityClassEstimator,
    NearestCentroidEstimator,
    RandomEstimator,
)
from meld_emotion.pipeline.cache import InMemoryFeatureCache, NullFeatureCache


def test_label_encoder_protocol() -> None:
    assert isinstance(EmotionLabelEncoder(), LabelEncoder)


def test_dataset_source_protocol() -> None:
    assert isinstance(SyntheticDatasetSource(), DatasetSource)


def test_feature_extractor_protocol() -> None:
    for extractor in (
        TextConceptExtractor(),
        BowTextExtractor(),
        AudioConceptExtractor(),
        MfccAcousticExtractor(),
        VideoConceptExtractor(),
        VisualCueExtractor(),
    ):
        assert isinstance(extractor, FeatureExtractor)


def test_estimator_protocol() -> None:
    for estimator in (
        MajorityClassEstimator(),
        RandomEstimator(),
        NearestCentroidEstimator(),
    ):
        assert isinstance(estimator, Estimator)


def test_classifier_protocol() -> None:
    factory = NearestCentroidEstimator
    classes = EmotionLabelEncoder().classes
    assert isinstance(EarlyFusionClassifier(factory, classes), Classifier)
    from meld_emotion.fusion.combiners import MeanCombiner

    assert isinstance(LateFusionClassifier(factory, MeanCombiner(), classes), Classifier)


def test_metric_protocol() -> None:
    assert isinstance(AccuracyMetric(), Metric)
    assert isinstance(MacroF1Metric(), Metric)


def test_explainer_protocol() -> None:
    assert isinstance(ModalityAblationExplainer(AccuracyMetric()), Explainer)
    assert isinstance(PermutationImportanceExplainer(AccuracyMetric()), Explainer)


def test_cache_protocol() -> None:
    assert isinstance(InMemoryFeatureCache(), FeatureCache)
    assert isinstance(NullFeatureCache(), FeatureCache)


def test_reporter_protocol() -> None:
    from meld_emotion.reporting.report import ConsoleReporter, JsonReporter

    assert isinstance(ConsoleReporter(), Reporter)
    assert isinstance(JsonReporter(), Reporter)
