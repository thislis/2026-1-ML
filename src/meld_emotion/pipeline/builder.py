"""구성 루트(composition root): 설정 → 구체 컴포넌트 연결.

DIP 의 핵심. 오직 이 모듈만이 모든 구체 구현을 import 하고, 설정 dataclass 를 보고 알맞은
객체를 생성해 :class:`ExperimentRunner` 로 조립한다. 새 컴포넌트를 추가하려면 여기 분기를
한 줄 더하고(설정→구체), 설정 dataclass 와 레지스트리 등록을 갖추면 된다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from meld_emotion.config.schema import (
    AudioConceptConfig,
    BowTextConfig,
    CacheConfig,
    CombinerConfig,
    ConsoleReporterConfig,
    CounterfactualConfig,
    DashboardReporterConfig,
    DatasetConfig,
    DialogueFineGrainedXaiConfig,
    DialogueRnnConfig,
    DiskCacheConfig,
    EarlyFusionConfig,
    EmbeddingGemmaTextConfig,
    EstimatorConfig,
    ExperimentConfig,
    ExplainerConfig,
    ExtractorConfig,
    JsonReporterConfig,
    KnnConfig,
    LateFusionConfig,
    LinearRegressionConfig,
    LogRegConfig,
    MajorityConfig,
    MeanCombinerConfig,
    MeldConfig,
    MemoryCacheConfig,
    MfccConfig,
    MlpConfig,
    ModalityAblationConfig,
    ModelConfig,
    NearestCentroidConfig,
    NullCacheConfig,
    PermutationConfig,
    PrecomputedMeldFeatureConfig,
    RandomEstimatorConfig,
    RandomForestConfig,
    ReporterConfig,
    SentenceEmbeddingConfig,
    StackingCombinerConfig,
    SvmConfig,
    SyntheticConfig,
    TextConceptConfig,
    TextTokenEmbeddingConfig,
    TfidfConfig,
    TimeSformerVideoConfig,
    VideoConceptConfig,
    VideoFrameEmbeddingConfig,
    VideoPrismConfig,
    VisualCueConfig,
    Wav2Vec2XlsrAudioConfig,
    Wav2Vec2XlsrAudioSequenceConfig,
    WeightedCombinerConfig,
    XGBoostConfig,
)
from meld_emotion.core.protocols import (
    Classifier,
    DatasetSource,
    Estimator,
    Explainer,
    FeatureCache,
    FeatureExtractor,
    Metric,
    Reporter,
)
from meld_emotion.core.types import Emotion, FeatureKind, Modality, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.media import MediaLoader
from meld_emotion.data.meld import MeldDatasetSource
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.evaluation.metrics import METRIC_REGISTRY
from meld_emotion.evaluation.robustness import RobustnessEvaluator
from meld_emotion.explain.counterfactual import CounterfactualExplainer
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.explain.modality_contribution import ModalityAblationExplainer
from meld_emotion.explain.permutation import PermutationImportanceExplainer
from meld_emotion.features.audio import (
    AudioConceptExtractor,
    MfccAcousticExtractor,
    Wav2Vec2XlsrAudioExtractor,
    Wav2Vec2XlsrAudioSequenceExtractor,
)
from meld_emotion.features.precomputed import MeldPrecomputedFeatureExtractor
from meld_emotion.features.text import (
    BowTextExtractor,
    EmbeddingGemmaTextExtractor,
    SentenceEmbeddingExtractor,
    TextConceptExtractor,
    TextTokenEmbeddingExtractor,
    TfidfTextExtractor,
)
from meld_emotion.features.video import (
    TimeSformerVideoExtractor,
    VideoConceptExtractor,
    VideoFrameEmbeddingExtractor,
    VideoPrismVideoExtractor,
    VisualCueExtractor,
)
from meld_emotion.fusion.combiners import (
    MeanCombiner,
    ProbabilityCombiner,
    StackingCombiner,
    WeightedCombiner,
)
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.late import LateFusionClassifier
from meld_emotion.fusion.masking import ModalityDropout, ModalityScenario, get_scenario
from meld_emotion.models.baselines import (
    LinearRegressionEstimator,
    MajorityClassEstimator,
    NearestCentroidEstimator,
    RandomEstimator,
)
from meld_emotion.models.mlp_estimator import MlpEstimator
from meld_emotion.models.sklearn_estimators import (
    KnnEstimator,
    LogisticRegressionEstimator,
    RandomForestEstimator,
    SvmEstimator,
)
from meld_emotion.models.xgboost_estimators import XGBoostEstimator
from meld_emotion.pipeline.cache import (
    DiskFeatureCache,
    InMemoryFeatureCache,
    NullFeatureCache,
)
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline
from meld_emotion.pipeline.runner import ExperimentRunner
from meld_emotion.reporting.report import (
    ConsoleReporter,
    DashboardExporter,
    JsonReporter,
)

logger = logging.getLogger(__name__)


def build_dataset(config: DatasetConfig) -> DatasetSource:
    if isinstance(config, SyntheticConfig):
        return SyntheticDatasetSource(
            n_train=config.n_train,
            n_dev=config.n_dev,
            n_test=config.n_test,
            seed=config.seed,
            with_audio=config.with_audio,
            with_video=config.with_video,
            missing_rate=config.missing_rate,
        )
    if isinstance(config, MeldConfig):
        return MeldDatasetSource(
            root=config.root,
            csv_train=config.csv_train,
            csv_dev=config.csv_dev,
            csv_test=config.csv_test,
            audio_subdir=config.audio_subdir,
            video_subdir=config.video_subdir,
            audio_subdir_train=config.audio_subdir_train,
            audio_subdir_dev=config.audio_subdir_dev,
            audio_subdir_test=config.audio_subdir_test,
            video_subdir_train=config.video_subdir_train,
            video_subdir_dev=config.video_subdir_dev,
            video_subdir_test=config.video_subdir_test,
            metadata_path=config.metadata_path,
        )
    raise ValueError(f"알 수 없는 데이터셋 설정: {type(config).__name__}")


def build_extractor(config: ExtractorConfig) -> FeatureExtractor:
    if isinstance(config, TextConceptConfig):
        return TextConceptExtractor()
    if isinstance(config, BowTextConfig):
        return BowTextExtractor(n_features=config.n_features, lowercase=config.lowercase)
    if isinstance(config, TfidfConfig):
        return TfidfTextExtractor(max_features=config.max_features, ngram_max=config.ngram_max)
    if isinstance(config, SentenceEmbeddingConfig):
        return SentenceEmbeddingExtractor(model_name=config.model_name, dim=config.dim)
    if isinstance(config, EmbeddingGemmaTextConfig):
        return EmbeddingGemmaTextExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            normalize=config.normalize,
            prompt_name=config.prompt_name,
            device=config.device,
        )
    if isinstance(config, TextTokenEmbeddingConfig):
        return TextTokenEmbeddingExtractor(
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            normalize=config.normalize,
            device=config.device,
        )
    if isinstance(config, AudioConceptConfig):
        return AudioConceptExtractor()
    if isinstance(config, MfccConfig):
        return MfccAcousticExtractor(n_mfcc=config.n_mfcc)
    if isinstance(config, Wav2Vec2XlsrAudioConfig):
        return Wav2Vec2XlsrAudioExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            sampling_rate=config.sampling_rate,
            max_seconds=config.max_seconds,
            chunk_seconds=config.chunk_seconds,
            normalize=config.normalize,
            device=config.device,
        )
    if isinstance(config, Wav2Vec2XlsrAudioSequenceConfig):
        return Wav2Vec2XlsrAudioSequenceExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            sampling_rate=config.sampling_rate,
            max_seconds=config.max_seconds,
            max_steps=config.max_steps,
            normalize=config.normalize,
            device=config.device,
        )
    if isinstance(config, VideoConceptConfig):
        return VideoConceptExtractor()
    if isinstance(config, VisualCueConfig):
        return VisualCueExtractor(dim=config.dim)
    if isinstance(config, TimeSformerVideoConfig):
        return TimeSformerVideoExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            normalize=config.normalize,
            pooling=config.pooling,
            device=config.device,
        )
    if isinstance(config, VideoPrismConfig):
        return VideoPrismVideoExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            normalize=config.normalize,
            prefer_batched_input=config.prefer_batched_input,
        )
    if isinstance(config, VideoFrameEmbeddingConfig):
        return VideoFrameEmbeddingExtractor(
            model_name=config.model_name,
            output_dim=config.output_dim,
            batch_size=config.batch_size,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            normalize=config.normalize,
            device=config.device,
        )
    if isinstance(config, PrecomputedMeldFeatureConfig):
        return MeldPrecomputedFeatureExtractor(
            path=config.path,
            modality=Modality(config.modality),
            kind=FeatureKind(config.kind),
            name_prefix=config.name_prefix,
        )
    raise ValueError(f"알 수 없는 추출기 설정: {type(config).__name__}")


def build_estimator_factory(config: EstimatorConfig) -> Callable[[int], Estimator]:
    """설정→학습기 팩토리. 팩토리는 전체 클래스 수(``n_classes``)를 받아 학습기를 만든다.

    한 분할에 소수 클래스가 누락돼도 ``predict_proba`` 의 열 수가 전체 클래스 수로 고정되도록,
    융합 분류기가 인코더의 클래스 수를 팩토리에 넘긴다(매 호출 새 인스턴스 — Late fusion 이
    모달리티마다 하나씩 학습).
    """

    if isinstance(config, MajorityConfig):
        return lambda n: MajorityClassEstimator(n_classes=n)
    if isinstance(config, RandomEstimatorConfig):
        return lambda n: RandomEstimator(n_classes=n, seed=config.seed)
    if isinstance(config, NearestCentroidConfig):
        return lambda n: NearestCentroidEstimator(n_classes=n, temperature=config.temperature)
    if isinstance(config, LinearRegressionConfig):
        return lambda n: LinearRegressionEstimator(
            n_classes=n,
            alpha=config.alpha,
            fit_intercept=config.fit_intercept,
        )
    if isinstance(config, SvmConfig):
        return lambda n: SvmEstimator(n_classes=n, C=config.C, kernel=config.kernel)
    if isinstance(config, LogRegConfig):
        return lambda n: LogisticRegressionEstimator(
            n_classes=n, C=config.C, max_iter=config.max_iter
        )
    if isinstance(config, RandomForestConfig):
        return lambda n: RandomForestEstimator(
            n_classes=n, n_estimators=config.n_estimators, max_depth=config.max_depth
        )
    if isinstance(config, KnnConfig):
        return lambda n: KnnEstimator(n_classes=n, n_neighbors=config.n_neighbors)
    if isinstance(config, XGBoostConfig):
        return lambda n: XGBoostEstimator(
            n_classes=n,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            seed=config.seed,
        )
    if isinstance(config, MlpConfig):
        return lambda n: MlpEstimator(
            n_classes=n,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            early_stopping_patience=config.early_stopping_patience,
            validation_split=config.validation_split,
            class_weight=config.class_weight,
            class_weights=config.class_weights,
            random_seed=config.random_seed,
            device=config.device,
        )
    raise ValueError(f"알 수 없는 학습기 설정: {type(config).__name__}")


def build_combiner(config: CombinerConfig) -> ProbabilityCombiner:
    if isinstance(config, MeanCombinerConfig):
        return MeanCombiner()
    if isinstance(config, WeightedCombinerConfig):
        return WeightedCombiner(weights=config.weights)
    if isinstance(config, StackingCombinerConfig):
        return StackingCombiner()
    raise ValueError(f"알 수 없는 결합기 설정: {type(config).__name__}")


def build_classifier(config: ModelConfig, classes: tuple[Emotion, ...]) -> Classifier:
    if isinstance(config, EarlyFusionConfig):
        return EarlyFusionClassifier(
            build_estimator_factory(config.base), classes, use_concepts=config.use_concepts
        )
    if isinstance(config, LateFusionConfig):
        return LateFusionClassifier(
            build_estimator_factory(config.base), build_combiner(config.combiner), classes
        )
    if isinstance(config, DialogueRnnConfig):
        from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier

        return TorchDialogueEmotionClassifier(config, classes)
    raise ValueError(f"알 수 없는 모델 설정: {type(config).__name__}")


def build_metric(name: str) -> Metric:
    metric = METRIC_REGISTRY.create(name)
    assert isinstance(metric, Metric)
    return metric


def build_explainer(config: ExplainerConfig) -> Explainer:
    if isinstance(config, PermutationConfig):
        return PermutationImportanceExplainer(
            metric=build_metric(config.metric),
            n_repeats=config.n_repeats,
            seed=config.seed,
            top_k=config.top_k,
            kinds=tuple(FeatureKind(k) for k in config.kinds),
        )
    if isinstance(config, ModalityAblationConfig):
        return ModalityAblationExplainer(metric=build_metric(config.metric))
    if isinstance(config, CounterfactualConfig):
        return CounterfactualExplainer(top_k=config.top_k, sample_limit=config.sample_limit)
    if isinstance(config, DialogueFineGrainedXaiConfig):
        return DialogueFineGrainedXaiExplainer(
            method=config.method,
            n_steps=config.n_steps,
            top_k=config.top_k,
            max_targets=config.max_targets,
            target=config.target,
        )
    raise ValueError(f"알 수 없는 설명기 설정: {type(config).__name__}")


def build_cache(config: CacheConfig) -> FeatureCache:
    if isinstance(config, MemoryCacheConfig):
        return InMemoryFeatureCache()
    if isinstance(config, NullCacheConfig):
        return NullFeatureCache()
    if isinstance(config, DiskCacheConfig):
        return DiskFeatureCache(path=config.path)
    raise ValueError(f"알 수 없는 캐시 설정: {type(config).__name__}")


def build_reporter(config: ReporterConfig) -> Reporter:
    if isinstance(config, ConsoleReporterConfig):
        return ConsoleReporter()
    if isinstance(config, JsonReporterConfig):
        return JsonReporter(path=config.path)
    if isinstance(config, DashboardReporterConfig):
        return DashboardExporter(path=config.path)
    raise ValueError(f"알 수 없는 리포터 설정: {type(config).__name__}")


def build_scenarios(names: Sequence[str]) -> list[ModalityScenario]:
    return [get_scenario(name) for name in names]


def build_experiment(
    config: ExperimentConfig, feature_cache: FeatureCache | None = None
) -> ExperimentRunner:
    """실험 설정으로부터 완전히 연결된 :class:`ExperimentRunner` 를 만든다."""

    logger.info(
        "실험 구성 시작: name=%s dataset=%s model=%s extractors=%s",
        config.name,
        type(config.dataset).__name__,
        type(config.model).__name__,
        ",".join(type(extractor).__name__ for extractor in config.extractors),
    )
    encoder = EmotionLabelEncoder()
    extractors = [build_extractor(e) for e in config.extractors]
    media_loader = MediaLoader(
        audio_sample_rate=config.media.audio_sample_rate,
        video_max_frames=config.media.video_max_frames,
        video_frame_size=config.media.video_frame_size,
        max_audio_seconds=config.media.max_audio_seconds,
        min_audio_seconds=config.media.min_audio_seconds,
    )
    feature_pipeline = FeaturePipeline(
        extractors,
        feature_cache if feature_cache is not None else build_cache(config.cache),
        media_loader,
        media_error_policy=config.media.on_error,
    )
    classifier = build_classifier(config.model, encoder.classes)

    metrics = [build_metric(name) for name in config.evaluation.metrics]
    evaluator = Evaluator(metrics, confusion=config.evaluation.confusion)
    scenarios = build_scenarios(config.evaluation.scenarios)
    robustness = RobustnessEvaluator(evaluator, scenarios) if scenarios else None

    dropout = (
        ModalityDropout(drop_prob=config.dropout.drop_prob, seed=config.dropout.seed)
        if config.dropout is not None
        else None
    )

    runner = ExperimentRunner(
        name=config.name,
        source=build_dataset(config.dataset),
        feature_pipeline=feature_pipeline,
        label_encoder=encoder,
        classifier=classifier,
        evaluator=evaluator,
        robustness=robustness,
        explainers=[build_explainer(x) for x in config.explainers],
        reporters=[build_reporter(r) for r in config.reporters],
        train_split=Split(config.train_split),
        eval_split=Split(config.eval_split),
        dropout=dropout,
    )
    logger.info(
        "실험 구성 완료: name=%s metrics=%s scenarios=%s reporters=%d",
        config.name,
        ",".join(config.evaluation.metrics),
        ",".join(config.evaluation.scenarios),
        len(config.reporters),
    )
    return runner
