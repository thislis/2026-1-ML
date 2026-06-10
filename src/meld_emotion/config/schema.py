"""실험 설정 dataclass (타입이 명시된 단일 진실 공급원).

모든 설정은 불변 dataclass 이며, 파이썬 코드에서 직접 생성하면 IDE/mypy 의 검사를 온전히
받는다. 각 설정 클래스는 다형성 식별자 ``type`` 을 :class:`ClassVar` 로 가지며, 이는 생성자
인자가 아니라 **YAML 경계**에서만 사용된다(:mod:`meld_emotion.config.loader`).

새 컴포넌트를 추가하려면: 여기 설정 dataclass 를 하나 만들고 해당 카테고리 레지스트리에
등록한 뒤, :mod:`meld_emotion.pipeline.builder` 에서 구체 클래스로 연결한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from meld_emotion.registry import Registry

# --- 카테고리 레지스트리 (type 이름 → 설정 클래스) -------------------------------
DATASET_CONFIGS: Registry[DatasetConfig] = Registry("dataset")
EXTRACTOR_CONFIGS: Registry[ExtractorConfig] = Registry("extractor")
ESTIMATOR_CONFIGS: Registry[EstimatorConfig] = Registry("estimator")
COMBINER_CONFIGS: Registry[CombinerConfig] = Registry("combiner")
MODEL_CONFIGS: Registry[ModelConfig] = Registry("model")
EXPLAINER_CONFIGS: Registry[ExplainerConfig] = Registry("explainer")
CACHE_CONFIGS: Registry[CacheConfig] = Registry("cache")
REPORTER_CONFIGS: Registry[ReporterConfig] = Registry("reporter")


# --- 데이터셋 ------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class SyntheticConfig(DatasetConfig):
    type: ClassVar[str] = "synthetic"
    n_train: int = 240
    n_dev: int = 60
    n_test: int = 60
    seed: int = 0
    with_audio: bool = True
    with_video: bool = True
    missing_rate: float = 0.0  # 모달리티 누락 시뮬레이션 비율


@dataclass(frozen=True)
class MeldConfig(DatasetConfig):
    type: ClassVar[str] = "meld"
    root: str = "data/MELD"
    csv_train: str = "train_sent_emo.csv"
    csv_dev: str = "dev_sent_emo.csv"
    csv_test: str = "test_sent_emo.csv"
    audio_subdir: str = "audio"
    video_subdir: str = "video"
    audio_subdir_train: str | None = None
    audio_subdir_dev: str | None = None
    audio_subdir_test: str | None = None
    video_subdir_train: str | None = None
    video_subdir_dev: str | None = None
    video_subdir_test: str | None = None
    metadata_path: str | None = None


DATASET_CONFIGS.add(SyntheticConfig.type, SyntheticConfig)
DATASET_CONFIGS.add(MeldConfig.type, MeldConfig)


# --- 특징 추출기 ---------------------------------------------------------------
@dataclass(frozen=True)
class ExtractorConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class TextConceptConfig(ExtractorConfig):
    type: ClassVar[str] = "text_concepts"


@dataclass(frozen=True)
class BowTextConfig(ExtractorConfig):
    type: ClassVar[str] = "text_bow"
    n_features: int = 256
    lowercase: bool = True


@dataclass(frozen=True)
class TfidfConfig(ExtractorConfig):
    type: ClassVar[str] = "text_tfidf"
    max_features: int = 5000
    ngram_max: int = 2


@dataclass(frozen=True)
class SentenceEmbeddingConfig(ExtractorConfig):
    type: ClassVar[str] = "text_embeddings"
    model_name: str = "all-MiniLM-L6-v2"
    dim: int = 384


@dataclass(frozen=True)
class EmbeddingGemmaTextConfig(ExtractorConfig):
    type: ClassVar[str] = "text_embeddinggemma"
    model_name: str = "google/embeddinggemma-300m"
    output_dim: int = 768
    batch_size: int = 32
    normalize: bool = True
    prompt_name: str | None = "Classification"
    device: str | None = None


@dataclass(frozen=True)
class TextTokenEmbeddingConfig(ExtractorConfig):
    type: ClassVar[str] = "text_token_embeddings"
    model_name: str = "bert-base-uncased"
    max_tokens: int = 64
    output_dim: int = 768
    batch_size: int = 16
    normalize: bool = True
    device: str | None = None


@dataclass(frozen=True)
class AudioConceptConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_concepts"


@dataclass(frozen=True)
class MfccConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_mfcc"
    n_mfcc: int = 13


@dataclass(frozen=True)
class Wav2Vec2XlsrAudioConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_wav2vec2_xlsr"
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    output_dim: int = 1024
    batch_size: int = 4
    sampling_rate: int = 16000
    max_seconds: float | None = None
    chunk_seconds: float | None = 30.0
    normalize: bool = True
    device: str | None = None


@dataclass(frozen=True)
class Wav2Vec2XlsrAudioSequenceConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_wav2vec2_xlsr_sequence"
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    output_dim: int = 1024
    batch_size: int = 4
    sampling_rate: int = 16000
    max_seconds: float | None = None
    max_steps: int = 128
    normalize: bool = True
    device: str | None = None


@dataclass(frozen=True)
class VideoConceptConfig(ExtractorConfig):
    type: ClassVar[str] = "video_concepts"


@dataclass(frozen=True)
class VisualCueConfig(ExtractorConfig):
    type: ClassVar[str] = "video_visual"
    dim: int = 16


@dataclass(frozen=True)
class TimeSformerVideoConfig(ExtractorConfig):
    type: ClassVar[str] = "video_timesformer"
    model_name: str = "facebook/timesformer-base-finetuned-k400"
    output_dim: int = 768
    batch_size: int = 2
    num_frames: int = 8
    frame_size: int = 224
    normalize: bool = True
    pooling: str = "cls"
    device: str | None = None


@dataclass(frozen=True)
class VideoPrismConfig(ExtractorConfig):
    type: ClassVar[str] = "video_videoprism"
    model_name: str = "google/videoprism-base-f16r288"
    output_dim: int = 768
    num_frames: int | None = 16
    frame_size: int = 288
    normalize: bool = True
    prefer_batched_input: bool = True


@dataclass(frozen=True)
class VideoFrameEmbeddingConfig(ExtractorConfig):
    type: ClassVar[str] = "video_frame_embeddings"
    model_name: str = "openai/clip-vit-base-patch32"
    output_dim: int = 768
    batch_size: int = 8
    num_frames: int = 16
    frame_size: int = 224
    normalize: bool = True
    device: str | None = None


@dataclass(frozen=True)
class PrecomputedMeldFeatureConfig(ExtractorConfig):
    type: ClassVar[str] = "meld_precomputed"
    path: str = ""
    modality: str = "text"
    kind: str = "embedding"
    name_prefix: str = ""


for _ec in (
    TextConceptConfig,
    BowTextConfig,
    TfidfConfig,
    SentenceEmbeddingConfig,
    EmbeddingGemmaTextConfig,
    TextTokenEmbeddingConfig,
    AudioConceptConfig,
    MfccConfig,
    Wav2Vec2XlsrAudioConfig,
    Wav2Vec2XlsrAudioSequenceConfig,
    VideoConceptConfig,
    VisualCueConfig,
    TimeSformerVideoConfig,
    VideoPrismConfig,
    VideoFrameEmbeddingConfig,
    PrecomputedMeldFeatureConfig,
):
    EXTRACTOR_CONFIGS.add(_ec.type, _ec)


# --- 기초 학습기(Estimator) -----------------------------------------------------
@dataclass(frozen=True)
class EstimatorConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class MajorityConfig(EstimatorConfig):
    type: ClassVar[str] = "majority"


@dataclass(frozen=True)
class RandomEstimatorConfig(EstimatorConfig):
    type: ClassVar[str] = "random"
    seed: int = 0


@dataclass(frozen=True)
class NearestCentroidConfig(EstimatorConfig):
    type: ClassVar[str] = "centroid"
    temperature: float = 1.0


@dataclass(frozen=True)
class LinearRegressionConfig(EstimatorConfig):
    type: ClassVar[str] = "linear_regression"
    alpha: float = 1e-6
    fit_intercept: bool = True


@dataclass(frozen=True)
class SvmConfig(EstimatorConfig):
    type: ClassVar[str] = "svm"
    C: float = 1.0
    kernel: str = "rbf"


@dataclass(frozen=True)
class LogRegConfig(EstimatorConfig):
    type: ClassVar[str] = "logreg"
    C: float = 1.0
    max_iter: int = 1000


@dataclass(frozen=True)
class RandomForestConfig(EstimatorConfig):
    type: ClassVar[str] = "random_forest"
    n_estimators: int = 200
    max_depth: int | None = None


@dataclass(frozen=True)
class KnnConfig(EstimatorConfig):
    type: ClassVar[str] = "knn"
    n_neighbors: int = 5


@dataclass(frozen=True)
class XGBoostConfig(EstimatorConfig):
    type: ClassVar[str] = "xgboost"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class MlpConfig(EstimatorConfig):
    type: ClassVar[str] = "mlp"
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5
    validation_split: float = 0.1
    class_weight: str = "none"
    class_weights: tuple[float, ...] = ()
    random_seed: int = 0
    device: str = "cpu"


for _est in (
    MajorityConfig,
    RandomEstimatorConfig,
    NearestCentroidConfig,
    LinearRegressionConfig,
    SvmConfig,
    LogRegConfig,
    RandomForestConfig,
    KnnConfig,
    XGBoostConfig,
    MlpConfig,
):
    ESTIMATOR_CONFIGS.add(_est.type, _est)


# --- Late fusion 결합기(Combiner) ----------------------------------------------
@dataclass(frozen=True)
class CombinerConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class MeanCombinerConfig(CombinerConfig):
    type: ClassVar[str] = "mean"


@dataclass(frozen=True)
class WeightedCombinerConfig(CombinerConfig):
    type: ClassVar[str] = "weighted"
    weights: dict[str, float] = field(default_factory=dict)  # modality 이름 → 가중치


@dataclass(frozen=True)
class StackingCombinerConfig(CombinerConfig):
    type: ClassVar[str] = "stacking"
    meta: EstimatorConfig = field(default_factory=lambda: LogRegConfig())


for _cb in (MeanCombinerConfig, WeightedCombinerConfig, StackingCombinerConfig):
    COMBINER_CONFIGS.add(_cb.type, _cb)


# --- 모델(융합 분류기) ----------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class EarlyFusionConfig(ModelConfig):
    type: ClassVar[str] = "early"
    base: EstimatorConfig = field(default_factory=lambda: MajorityConfig())
    use_concepts: bool = True  # 개념 특징도 입력에 포함할지


@dataclass(frozen=True)
class LateFusionConfig(ModelConfig):
    type: ClassVar[str] = "late"
    base: EstimatorConfig = field(default_factory=lambda: MajorityConfig())
    combiner: CombinerConfig = field(default_factory=lambda: MeanCombinerConfig())


@dataclass(frozen=True)
class EnsembleDistillationSettings:
    enabled: bool = False
    teacher: str = "svm"
    teacher_probs_path: str | None = None
    temperature: float = 2.0
    weight: float = 0.3


@dataclass(frozen=True)
class EnsembleSettings:
    mode: str = "late_logits"
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.5
    svm_logits_path: str | None = None
    logreg_logits_path: str | None = None
    artifact_format: str = "auto"  # auto | logits | proba
    distillation: EnsembleDistillationSettings = field(
        default_factory=EnsembleDistillationSettings
    )


@dataclass(frozen=True)
class EnsembleConfig(ModelConfig):
    type: ClassVar[str] = "ensemble"
    base: ModelConfig = field(default_factory=lambda: DialogueRnnConfig())
    ensemble: EnsembleSettings = field(default_factory=EnsembleSettings)


@dataclass(frozen=True)
class MoeExpertSettings:
    text: bool = True
    audio: bool = True
    video: bool = True
    context: bool = True
    neutral: bool = True
    rare: bool = True
    svm_logreg: bool = True


@dataclass(frozen=True)
class RareExpertSettings:
    enabled: bool = True
    target_classes: tuple[int, ...] = (5, 6)
    loss_weight: float = 0.5
    hard_negative_weight: float = 1.5


@dataclass(frozen=True)
class MoeSettings:
    enabled: bool = True
    routing: str = "top2"
    top_k: int = 2
    load_balancing_loss_weight: float = 0.01
    expert_dropout: float = 0.1
    class_aware_routing: bool = True
    experts: MoeExpertSettings = field(default_factory=MoeExpertSettings)
    rare_expert: RareExpertSettings = field(default_factory=RareExpertSettings)
    svm_logits_path: str | None = None
    logreg_logits_path: str | None = None
    artifact_format: str = "auto"


@dataclass(frozen=True)
class MoeConfig(ModelConfig):
    type: ClassVar[str] = "moe"
    moe: MoeSettings = field(default_factory=MoeSettings)


@dataclass(frozen=True)
class ModalityEncoderSettings:
    text_input_dim: int = 0
    audio_input_dim: int = 0
    video_input_dim: int = 0
    encoder_type: str = "rnn"
    sequence_fallback_policy: str = "pooled"  # pooled | error
    proj_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    num_heads: int = 4
    conv_kernel_size: int = 15
    ffn_multiplier: float = 4.0
    dropout: float = 0.2
    attention_dropout: float = 0.1
    pooling_type: str = "attentive"


@dataclass(frozen=True)
class FusionSettings:
    modality_dim: int = 128
    fusion_dim: int = 256
    dropout: float = 0.3
    use_gated_fusion: bool = True
    use_interaction_features: bool = True
    use_interaction: bool = True
    gate_entropy_weight: float = 0.0


@dataclass(frozen=True)
class DialogueContextSettings:
    use_context: bool = True
    use_speaker: bool = True
    speaker_emb_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.3


@dataclass(frozen=True)
class MemoryAttentionSettings:
    use_memory: bool = True
    enabled: bool = True
    hidden_dim: int = 256
    attn_dim: int = 256
    use_rope: bool = False
    use_relative_distance_bias: bool = True
    use_same_speaker_bias: bool = True
    max_relative_distance: int = 32


@dataclass(frozen=True)
class ClassifierHeadSettings:
    classifier_head_type: str = "concat"
    use_context: bool = True
    use_memory: bool = True
    gate_hidden_dim: int = 128
    gate_dropout: float = 0.1
    aux_text_loss_weight: float = 0.0
    aux_audio_loss_weight: float = 0.0
    aux_video_loss_weight: float = 0.0
    hidden_dim: int = 256
    dropout: float = 0.3


@dataclass(frozen=True)
class DialogueTrainingSettings:
    lr: float = 0.0002
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    batch_size: int = 8
    max_epochs: int = 50
    early_stopping_patience: int = 8
    validation_fraction: float = 0.1
    modality_dropout: float = 0.2
    text_dropout: float = 0.0
    context_dropout: float = 0.0
    seed: int = 0
    device: str = "cpu"
    best_checkpoint_path: str | None = None


@dataclass(frozen=True)
class LogitAdjustmentSettings:
    enabled: bool = False
    tau: float = 1.0


@dataclass(frozen=True)
class HardNegativeMiningSettings:
    enabled: bool = False
    weight: float = 1.0
    target_classes: tuple[int, ...] = ()


@dataclass(frozen=True)
class LossSettings:
    type: str = "cross_entropy"
    gamma: float = 2.0
    class_balanced_beta: float = 0.999
    label_smoothing: float = 0.0
    logit_adjustment: LogitAdjustmentSettings = field(default_factory=LogitAdjustmentSettings)
    hard_negative_mining: HardNegativeMiningSettings = field(
        default_factory=HardNegativeMiningSettings
    )


@dataclass(frozen=True)
class CalibrationSettings:
    enabled: bool = False
    temperature_scaling: bool = False
    threshold_tuning: bool = False
    rare_class_margin_enabled: bool = False
    rare_classes: tuple[int, ...] = ()
    rare_class_threshold: float = 0.0
    rare_class_margin: float = 0.0


@dataclass(frozen=True)
class NeutralGateSettings:
    enabled: bool = False
    threshold: float = 0.5
    threshold_tuning: bool = False
    neutral_class_index: int = 0
    binary_loss_weight: float = 0.0


@dataclass(frozen=True)
class DialogueRnnConfig(ModelConfig):
    type: ClassVar[str] = "dialogue_rnn"
    num_classes: int = 7
    rnn_type: str = "gru"
    modality_encoder: ModalityEncoderSettings = field(default_factory=ModalityEncoderSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    dialogue_context: DialogueContextSettings = field(default_factory=DialogueContextSettings)
    memory_attention: MemoryAttentionSettings = field(default_factory=MemoryAttentionSettings)
    classifier: ClassifierHeadSettings = field(default_factory=ClassifierHeadSettings)
    training: DialogueTrainingSettings = field(default_factory=DialogueTrainingSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    neutral_gate: NeutralGateSettings = field(default_factory=NeutralGateSettings)


MODEL_CONFIGS.add(EarlyFusionConfig.type, EarlyFusionConfig)
MODEL_CONFIGS.add(LateFusionConfig.type, LateFusionConfig)
MODEL_CONFIGS.add(DialogueRnnConfig.type, DialogueRnnConfig)
MODEL_CONFIGS.add(EnsembleConfig.type, EnsembleConfig)
MODEL_CONFIGS.add(MoeConfig.type, MoeConfig)


# --- 평가 ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1", "per_class_recall")
    confusion: bool = True
    scenarios: tuple[str, ...] = ("full",)


# --- 학습 시 모달리티 드롭아웃 --------------------------------------------------
@dataclass(frozen=True)
class DropoutConfig:
    """학습 시 modality dropout 증강 설정(제안서). ``None`` 이면 적용하지 않는다."""

    drop_prob: float = 0.3
    seed: int = 0


# --- Raw media 적재 -------------------------------------------------------------
@dataclass(frozen=True)
class MediaConfig:
    """MP4 등 raw media 를 특징 추출 전 lazy-load 하는 설정."""

    audio_sample_rate: int = 16000
    video_max_frames: int = 32
    video_frame_size: tuple[int, int] = (64, 64)  # (height, width)
    on_error: str = "raise"  # raise | drop_modality | drop_sample
    max_audio_seconds: float | None = None  # 초과 시 샘플 전체 제외
    min_audio_seconds: float | None = None  # 미만 시 샘플 전체 제외


# --- 설명(Explainer) -----------------------------------------------------------
@dataclass(frozen=True)
class ExplainerConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class PermutationConfig(ExplainerConfig):
    type: ClassVar[str] = "permutation"
    metric: str = "macro_f1"
    n_repeats: int = 5
    seed: int = 0
    top_k: int = 20
    kinds: tuple[str, ...] = ("concept",)  # 중요도 대상 특징 종류 (기본: 개념만; 비용 절감)


@dataclass(frozen=True)
class ModalityAblationConfig(ExplainerConfig):
    type: ClassVar[str] = "modality_ablation"
    metric: str = "macro_f1"


@dataclass(frozen=True)
class CounterfactualConfig(ExplainerConfig):
    type: ClassVar[str] = "counterfactual"
    top_k: int = 5
    sample_limit: int = 20


@dataclass(frozen=True)
class DialogueFineGrainedXaiConfig(ExplainerConfig):
    type: ClassVar[str] = "dialogue_finegrained_xai"
    method: str = "integrated_gradients"
    n_steps: int = 32
    top_k: int = 10
    max_targets: int = 32
    target: str = "predicted"


for _xc in (
    PermutationConfig,
    ModalityAblationConfig,
    CounterfactualConfig,
    DialogueFineGrainedXaiConfig,
):
    EXPLAINER_CONFIGS.add(_xc.type, _xc)


# --- 특징 캐시 -----------------------------------------------------------------
@dataclass(frozen=True)
class CacheConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class MemoryCacheConfig(CacheConfig):
    type: ClassVar[str] = "memory"


@dataclass(frozen=True)
class NullCacheConfig(CacheConfig):
    type: ClassVar[str] = "null"


@dataclass(frozen=True)
class DiskCacheConfig(CacheConfig):
    type: ClassVar[str] = "disk"
    path: str = ".feature_cache"


for _cc in (MemoryCacheConfig, NullCacheConfig, DiskCacheConfig):
    CACHE_CONFIGS.add(_cc.type, _cc)


# --- 리포터 -------------------------------------------------------------------
@dataclass(frozen=True)
class ReporterConfig:
    type: ClassVar[str] = "base"


@dataclass(frozen=True)
class ConsoleReporterConfig(ReporterConfig):
    type: ClassVar[str] = "console"


@dataclass(frozen=True)
class JsonReporterConfig(ReporterConfig):
    type: ClassVar[str] = "json"
    path: str = "outputs/result.json"


@dataclass(frozen=True)
class DashboardReporterConfig(ReporterConfig):
    type: ClassVar[str] = "dashboard"
    path: str = "outputs/dashboard.json"


for _rc in (ConsoleReporterConfig, JsonReporterConfig, DashboardReporterConfig):
    REPORTER_CONFIGS.add(_rc.type, _rc)


# --- 최상위 실험 설정 -----------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """한 실험을 완전히 기술하는 최상위 설정."""

    name: str = "experiment"
    seed: int = 0
    output_dir: str = "outputs"
    train_split: str = "train"  # 학습에 사용할 분할
    eval_split: str = "test"  # 평가·강건성·설명을 수행할 분할 (예: dev 로 바꿔 개발셋 평가)
    dataset: DatasetConfig = field(default_factory=lambda: SyntheticConfig())
    extractors: tuple[ExtractorConfig, ...] = field(default_factory=lambda: (TextConceptConfig(),))
    model: ModelConfig = field(default_factory=lambda: EarlyFusionConfig())
    dropout: DropoutConfig | None = None  # 학습 시 modality dropout (None = 미적용)
    media: MediaConfig = field(default_factory=MediaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainers: tuple[ExplainerConfig, ...] = ()
    cache: CacheConfig = field(default_factory=lambda: MemoryCacheConfig())
    reporters: tuple[ReporterConfig, ...] = field(
        default_factory=lambda: (ConsoleReporterConfig(),)
    )


# --- 다중 실험 비교(suite) 설정 -------------------------------------------------
@dataclass(frozen=True)
class SuiteConfig:
    """여러 :class:`ExperimentConfig` 를 함께 실행·비교하기 위한 묶음 설정.

    ``experiments`` 는 (보통 ``base`` 와 병합되어) 완성된 실험 설정들이며, ``metrics`` 는
    비교표에 스칼라로 나열할 지표 이름들이다. YAML 경계 처리는
    :func:`meld_emotion.config.loader.load_suite` 가 담당한다.
    """

    name: str = "suite"
    experiments: tuple[ExperimentConfig, ...] = ()
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1")
    robustness_metric: str = "macro_f1"
    output_path: str = "outputs/comparison.json"
