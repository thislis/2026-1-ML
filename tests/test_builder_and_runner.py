"""빌더(구성 루트)와 러너의 end-to-end 동작."""

from __future__ import annotations

from meld_emotion.config.schema import BowTextConfig as Bow
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    LateFusionConfig,
    MeanCombinerConfig,
    ModalityAblationConfig,
    NearestCentroidConfig,
    NullCacheConfig,
    SyntheticConfig,
    TextConceptConfig,
)
from meld_emotion.pipeline.builder import build_experiment


def _config(model) -> ExperimentConfig:
    return ExperimentConfig(
        name="e2e",
        dataset=SyntheticConfig(n_train=140, n_dev=0, n_test=70),
        extractors=(TextConceptConfig(), Bow(n_features=24)),
        model=model,
        evaluation=EvaluationConfig(
            metrics=("accuracy", "macro_f1"),
            scenarios=("full", "no_text"),
        ),
        explainers=(ModalityAblationConfig(metric="accuracy"),),
        cache=NullCacheConfig(),
        reporters=(),
    )


def test_end_to_end_early() -> None:
    config = _config(EarlyFusionConfig(base=NearestCentroidConfig()))
    result = build_experiment(config).run()
    assert result.name == "e2e"
    accuracy = result.evaluation.metric("accuracy")
    assert accuracy is not None and accuracy.value > 0.5
    assert result.robustness is not None
    assert {r.scenario for r in result.robustness.reports} == {"full", "no_text"}
    assert result.explanation is not None
    assert len(result.explanation.modality_contributions) >= 1


def test_end_to_end_late() -> None:
    config = _config(LateFusionConfig(base=NearestCentroidConfig(), combiner=MeanCombinerConfig()))
    result = build_experiment(config).run()
    accuracy = result.evaluation.metric("accuracy")
    assert accuracy is not None and accuracy.value > 0.5


def test_json_reporter_writes(tmp_path) -> None:
    from meld_emotion.config.schema import JsonReporterConfig

    out = tmp_path / "r.json"
    config = _config(EarlyFusionConfig(base=NearestCentroidConfig()))
    config = ExperimentConfig(
        name=config.name,
        dataset=config.dataset,
        extractors=config.extractors,
        model=config.model,
        evaluation=config.evaluation,
        explainers=config.explainers,
        cache=config.cache,
        reporters=(JsonReporterConfig(path=str(out)),),
    )
    build_experiment(config).run()
    assert out.exists()


def test_meld_source_metadata_runs() -> None:
    from pathlib import Path

    from meld_emotion.config.schema import MeldConfig

    root = Path(__file__).resolve().parents[1]
    config = _config(EarlyFusionConfig(base=NearestCentroidConfig()))
    config = ExperimentConfig(
        name="meld",
        train_split="dev",
        eval_split="dev",
        dataset=MeldConfig(
            metadata_path=str(root / "MELD.Features.Models" / "features" / "data_emotion.p")
        ),
        extractors=config.extractors,
        model=config.model,
        evaluation=EvaluationConfig(metrics=("accuracy",), confusion=False, scenarios=()),
        reporters=(),
    )
    result = build_experiment(config).run()
    assert result.evaluation.metric("accuracy") is not None
