"""PyTorch MLP pooled-feature estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 미설치 (uv sync --extra deep 로 설치)")

from meld_emotion.config.schema import MlpConfig
from meld_emotion.models.mlp_estimator import MlpEstimator
from meld_emotion.pipeline.builder import build_estimator_factory


def _toy_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3, dtype=np.int64), 12)
    centers = np.asarray(
        [
            [-2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    x = centers[y] + rng.normal(scale=0.1, size=(y.size, centers.shape[1]))
    return x.astype(np.float64), y


def _estimator(**kwargs: object) -> MlpEstimator:
    params: dict[str, Any] = {
        "hidden_dim": 16,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "batch_size": 8,
        "max_epochs": 8,
        "early_stopping_patience": 2,
        "validation_split": 0.2,
        "random_seed": 0,
        "device": "cpu",
    }
    params.update(kwargs)
    return MlpEstimator(n_classes=5, **params)


def test_mlp_can_fit_small_synthetic_dataset() -> None:
    x, y = _toy_xy()
    est = _estimator().fit(x, y)

    pred = est.predict(x)

    assert pred.shape == (y.size,)
    assert float(np.mean(pred == y)) > 0.5


def test_mlp_predict_proba_shape_and_rows_sum() -> None:
    x, y = _toy_xy()
    est = _estimator().fit(x, y)

    proba = est.predict_proba(x[:5])

    assert proba.shape == (5, 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_mlp_balanced_class_weights_do_not_crash() -> None:
    x, y = _toy_xy()
    imbalanced = y != 2
    est = _estimator(class_weight="balanced").fit(x[imbalanced], y[imbalanced])

    proba = est.predict_proba(x[:3])

    assert proba.shape == (3, 5)


def test_mlp_explicit_class_weights_do_not_crash() -> None:
    x, y = _toy_xy()
    est = _estimator(class_weight="explicit", class_weights=(1.0, 2.0, 1.0, 0.0, 0.0)).fit(x, y)

    assert est.predict_proba(x[:2]).shape == (2, 5)


def test_mlp_early_stopping_path_with_validation_split() -> None:
    x, y = _toy_xy()
    est = _estimator(max_epochs=20, early_stopping_patience=1, validation_split=0.3).fit(x, y)

    assert est.predict(x[:4]).shape == (4,)


def test_mlp_trains_without_validation_split() -> None:
    x, y = _toy_xy()
    est = _estimator(validation_split=0.0).fit(x, y)

    assert est.predict_proba(x[:4]).shape == (4, 5)


def test_mlp_save_load_roundtrip(tmp_path: Path) -> None:
    x, y = _toy_xy()
    est = _estimator().fit(x, y)
    path = tmp_path / "mlp.pt"

    est.save(path)
    restored = MlpEstimator.load(path, device="cpu")

    assert np.allclose(est.predict_proba(x[:5]), restored.predict_proba(x[:5]), atol=1e-6)


def test_builder_creates_mlp_estimator_from_config() -> None:
    factory = build_estimator_factory(
        MlpConfig(
            hidden_dim=16,
            dropout=0.0,
            learning_rate=0.01,
            batch_size=8,
            max_epochs=4,
            early_stopping_patience=2,
            validation_split=0.0,
            random_seed=0,
            device="cpu",
        )
    )

    assert isinstance(factory(5), MlpEstimator)
