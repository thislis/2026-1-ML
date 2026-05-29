"""핵심 계약(Protocol)과 데이터 타입. 다른 어떤 내부 구현에도 의존하지 않는다."""

from __future__ import annotations

from meld_emotion.core.data import (
    AudioInput,
    ModalityMask,
    RawSample,
    VideoInput,
)
from meld_emotion.core.features import (
    ColumnSpec,
    FeatureBundle,
    FeatureMatrix,
    StackedFeatures,
)
from meld_emotion.core.protocols import (
    Classifier,
    DatasetSource,
    Estimator,
    ExperimentEvaluator,
    Explainer,
    FeatureCache,
    FeatureExtractor,
    LabelEncoder,
    Metric,
    Reporter,
)
from meld_emotion.core.results import (
    ConfusionMatrixResult,
    CounterfactualResult,
    EvaluationReport,
    ExperimentResult,
    ExplanationReport,
    FeatureContribution,
    MetricResult,
    ModalityContribution,
    PredictionSet,
    RobustnessReport,
)
from meld_emotion.core.status import (
    ComponentStatus,
    note_placeholder_use,
    placeholder,
    raise_unimplemented,
    real,
    unimplemented,
)
from meld_emotion.core.types import (
    UID,
    BoolArray,
    Emotion,
    FeatureKind,
    FloatArray,
    IntArray,
    Modality,
    Sentiment,
    Split,
)

__all__ = [
    # types
    "Modality",
    "Emotion",
    "Sentiment",
    "Split",
    "FeatureKind",
    "FloatArray",
    "IntArray",
    "BoolArray",
    "UID",
    # data
    "AudioInput",
    "VideoInput",
    "ModalityMask",
    "RawSample",
    # features
    "FeatureMatrix",
    "FeatureBundle",
    "StackedFeatures",
    "ColumnSpec",
    # results
    "PredictionSet",
    "MetricResult",
    "ConfusionMatrixResult",
    "EvaluationReport",
    "RobustnessReport",
    "FeatureContribution",
    "ModalityContribution",
    "CounterfactualResult",
    "ExplanationReport",
    "ExperimentResult",
    # protocols
    "LabelEncoder",
    "DatasetSource",
    "FeatureExtractor",
    "Estimator",
    "Classifier",
    "Metric",
    "Explainer",
    "FeatureCache",
    "Reporter",
    "ExperimentEvaluator",
    # status
    "ComponentStatus",
    "real",
    "placeholder",
    "unimplemented",
    "note_placeholder_use",
    "raise_unimplemented",
]
