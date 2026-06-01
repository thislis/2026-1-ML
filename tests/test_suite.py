"""다중 실험 비교(suite): 깊은 병합, 경계 내성, 비교 리포트."""

from __future__ import annotations

from pathlib import Path

import pytest

from meld_emotion.config.loader import load_suite, suite_from_dict
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    LateFusionConfig,
    MeanCombinerConfig,
    NearestCentroidConfig,
    SyntheticConfig,
)
from meld_emotion.pipeline.suite import SuiteRunner
from meld_emotion.reporting.report import ComparisonReporter


def _small_synth() -> dict:
    return {"type": "synthetic", "n_train": 70, "n_dev": 0, "n_test": 70}


def test_deep_merge_overrides_only_diff() -> None:
    suite = suite_from_dict(
        {
            "base": {"seed": 3, "dataset": {"type": "synthetic", "n_train": 10, "n_test": 10}},
            "experiments": [{"name": "a", "dataset": {"n_train": 20}}],
        }
    )
    cfg = suite.experiments[0]
    assert cfg.seed == 3  # base 유지
    assert isinstance(cfg.dataset, SyntheticConfig)
    assert cfg.dataset.n_train == 20  # override
    assert cfg.dataset.n_test == 10  # base 의 다른 키 보존
    assert cfg.dataset.type == "synthetic"  # 병합으로 type 유지


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="중복"):
        suite_from_dict({"experiments": [{"name": "dup"}, {"name": "dup"}]})


def test_empty_experiments_rejected() -> None:
    with pytest.raises(ValueError):
        suite_from_dict({"experiments": []})


def _runner() -> SuiteRunner:
    evaluation = EvaluationConfig(metrics=("accuracy", "macro_f1"), scenarios=("full", "no_text"))
    dataset = SyntheticConfig(n_train=140, n_dev=0, n_test=70)
    configs = [
        ExperimentConfig(
            name="early",
            dataset=dataset,
            model=EarlyFusionConfig(base=NearestCentroidConfig()),
            evaluation=evaluation,
            reporters=(),
        ),
        ExperimentConfig(
            name="late",
            dataset=dataset,
            model=LateFusionConfig(base=NearestCentroidConfig(), combiner=MeanCombinerConfig()),
            evaluation=evaluation,
            reporters=(),
        ),
    ]
    return SuiteRunner("cmp", configs)


def test_suite_runs_and_collects_results() -> None:
    report = _runner().run()
    assert {o.name for o in report.outcomes} == {"early", "late"}
    assert all(o.ok for o in report.outcomes)
    early = next(o for o in report.outcomes if o.name == "early")
    assert early.result is not None
    acc = early.result.evaluation.metric("accuracy")
    assert acc is not None and acc.value > 0.5


def test_suite_tolerates_unimplemented_variant() -> None:
    data = {
        "base": {"dataset": _small_synth(), "reporters": []},
        "experiments": [
            {"name": "ok", "model": {"type": "early", "base": {"type": "centroid"}}},
            {
                "name": "boundary",
                "dataset": {"type": "meld"},
                "model": {"type": "early", "base": {"type": "centroid"}},
            },
        ],
    }
    report = SuiteRunner("t", suite_from_dict(data).experiments).run()
    outcomes = {o.name: o for o in report.outcomes}
    assert outcomes["ok"].ok
    assert not outcomes["boundary"].ok
    assert outcomes["boundary"].error is not None
    assert "NotImplementedError" in outcomes["boundary"].error


def test_comparison_reporter_writes_and_formats(tmp_path: Path) -> None:
    report = _runner().run()
    out = tmp_path / "cmp.json"
    reporter = ComparisonReporter(metrics=("accuracy", "macro_f1"), path=str(out))
    text = reporter.format(report)
    assert "early" in text and "late" in text
    assert "Robustness" in text  # 시나리오가 있으므로 강건성 표가 포함됨
    reporter.save(report)
    assert out.exists() and out.read_text(encoding="utf-8").strip()


def test_example_suite_loads_and_runs() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "example_suite.yaml"
    suite = load_suite(path)
    assert suite.name == "fusion_comparison"
    assert {c.name for c in suite.experiments} == {
        "early_centroid",
        "late_centroid_mean",
        "early_majority",
    }
    report = SuiteRunner(suite.name, suite.experiments).run()
    assert all(o.ok for o in report.outcomes)
