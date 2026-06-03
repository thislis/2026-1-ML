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
class AudioConceptConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_concepts"


@dataclass(frozen=True)
class MfccConfig(ExtractorConfig):
    type: ClassVar[str] = "audio_mfcc"
    n_mfcc: int = 13


@dataclass(frozen=True)
class VideoConceptConfig(ExtractorConfig):
    type: ClassVar[str] = "video_concepts"


@dataclass(frozen=True)
class VisualCueConfig(ExtractorConfig):
    type: ClassVar[str] = "video_visual"
    dim: int = 16


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
    AudioConceptConfig,
    MfccConfig,
    VideoConceptConfig,
    VisualCueConfig,
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


MODEL_CONFIGS.add(EarlyFusionConfig.type, EarlyFusionConfig)
MODEL_CONFIGS.add(LateFusionConfig.type, LateFusionConfig)


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


for _xc in (PermutationConfig, ModalityAblationConfig, CounterfactualConfig):
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
