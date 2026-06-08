"""Calibration and prediction postprocessing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch 미설치 (uv sync --extra deep 로 설치)")
from torch.nn import functional as F  # noqa: E402

from meld_emotion.models.calibration import (  # noqa: E402
    CalibrationParams,
    PredictionPostprocessor,
    fit_temperature,
    tune_class_thresholds,
)


def test_temperature_scaling_fit_does_not_increase_nll_on_toy_dev_set() -> None:
    logits = torch.tensor([[6.0, 0.0], [0.0, 6.0], [6.0, 0.0], [0.0, 6.0]])
    labels = torch.tensor([0, 1, 1, 0])
    before = F.cross_entropy(logits, labels)

    temperature = fit_temperature(logits, labels)
    after = F.cross_entropy(logits / temperature, labels)

    assert temperature > 0.0
    assert float(after) <= float(before) + 1.0e-6


def test_calibration_params_roundtrip() -> None:
    params = CalibrationParams(
        temperature=1.5,
        class_thresholds=(0.2, 0.7),
        rare_classes=(1,),
        rare_class_threshold=0.3,
        rare_class_margin=0.1,
        class_labels=("neutral", "joy"),
    )

    restored = CalibrationParams.from_dict(params.to_dict())

    assert restored == params


def test_threshold_rules_are_deterministic() -> None:
    probs = np.asarray([[0.45, 0.44, 0.11], [0.40, 0.41, 0.19]], dtype=np.float64)
    params = CalibrationParams(class_thresholds=(0.9, 0.4, 0.9))

    pred = PredictionPostprocessor(params).predict(probs)

    assert pred.tolist() == [1, 1]


def test_rare_class_margin_rule_is_deterministic() -> None:
    probs = np.asarray([[0.50, 0.46, 0.04], [0.70, 0.20, 0.10]], dtype=np.float64)
    params = CalibrationParams(
        rare_classes=(1,),
        rare_class_threshold=0.4,
        rare_class_margin=0.05,
    )

    pred = PredictionPostprocessor(params).predict(probs)

    assert pred.tolist() == [1, 0]


def test_threshold_tuning_returns_one_threshold_per_class() -> None:
    probs = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    thresholds = tune_class_thresholds(probs, labels)

    assert len(thresholds) == 2
    assert all(0.05 <= threshold <= 0.95 for threshold in thresholds)
