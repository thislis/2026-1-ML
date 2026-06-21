"""CatBoost baseline wrapper 계약."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.config.schema import (
    CatBoostConfig,
    EarlyFusionConfig,
    ExperimentConfig,
    SyntheticConfig,
)
from meld_emotion.core.protocols import Estimator
from meld_emotion.models.catboost_estimators import CatBoostEstimator
from meld_emotion.pipeline.builder import build_experiment

pytestmark = pytest.mark.catboost_native
pytest.importorskip("catboost", reason="catboost 미설치 (uv sync --extra catboost 로 설치)")

_N_EMOTIONS = 7


def _xy_missing_top_classes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5, dtype=np.int64), 10)
    x = rng.normal(loc=y[:, None] * 2.0, scale=1.0, size=(y.size, 8)).astype(np.float64)
    return x, y


def test_satisfies_estimator_protocol() -> None:
    assert isinstance(CatBoostEstimator(), Estimator)


def test_proba_full_width_when_classes_missing() -> None:
    x, y = _xy_missing_top_classes()
    est = CatBoostEstimator(n_classes=_N_EMOTIONS, iterations=5).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (y.size, _N_EMOTIONS)
    assert np.allclose(proba[:, 5], 0.0) and np.allclose(proba[:, 6], 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_matches_proba_argmax() -> None:
    x, y = _xy_missing_top_classes()
    est = CatBoostEstimator(n_classes=_N_EMOTIONS, iterations=5).fit(x, y)
    assert np.array_equal(est.predict(x), np.argmax(est.predict_proba(x), axis=1))


def test_builder_early_fusion_runs() -> None:
    config = ExperimentConfig(
        name="catboost",
        dataset=SyntheticConfig(n_train=140, n_dev=0, n_test=70),
        model=EarlyFusionConfig(base=CatBoostConfig(iterations=5, depth=3)),
        reporters=(),
    )
    result = build_experiment(config).run()
    assert result.evaluation.metric("accuracy") is not None
