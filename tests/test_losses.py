"""Dialogue RNN loss helpers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch 미설치 (uv sync --extra deep 로 설치)")
from torch.nn import functional as F  # noqa: E402

from meld_emotion.config.schema import (  # noqa: E402
    HardNegativeMiningSettings,
    LogitAdjustmentSettings,
    LossSettings,
)
from meld_emotion.models.losses import (  # noqa: E402
    class_balanced_weights,
    compute_dialogue_loss,
    false_positive_counts,
)


def test_class_balanced_focal_loss_is_finite() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([10.0, 2.0, 1.0])

    loss = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(type="class_balanced_focal", gamma=2.0),
        class_counts=counts,
    )

    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_class_balanced_gamma_zero_matches_weighted_ce() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([10.0, 2.0, 1.0])
    settings = LossSettings(type="class_balanced_focal", gamma=0.0, class_balanced_beta=0.9)

    loss = compute_dialogue_loss(logits, labels, settings=settings, class_counts=counts)
    weights = class_balanced_weights(counts, beta=0.9, device=logits.device)
    expected = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=weights)

    assert loss == pytest.approx(expected)


def test_label_smoothing_changes_loss_smoothly() -> None:
    logits = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([1.0, 1.0, 1.0])

    plain = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(type="cross_entropy", label_smoothing=0.0),
        class_counts=counts,
    )
    smooth = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(type="cross_entropy", label_smoothing=0.2),
        class_counts=counts,
    )

    assert torch.isfinite(smooth)
    assert smooth != pytest.approx(plain)


def test_logit_adjustment_only_when_enabled() -> None:
    logits = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([100.0, 1.0, 1.0])

    disabled = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(
            type="cross_entropy",
            logit_adjustment=LogitAdjustmentSettings(enabled=False, tau=1.0),
        ),
        class_counts=counts,
    )
    enabled = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(
            type="cross_entropy",
            logit_adjustment=LogitAdjustmentSettings(enabled=True, tau=1.0),
        ),
        class_counts=counts,
    )

    assert enabled != pytest.approx(disabled)


def test_false_positive_counts_by_predicted_class() -> None:
    y_true = np.array([0, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([1, 1, 1, 0], dtype=np.int64)

    counts = false_positive_counts(y_true, y_pred, n_classes=3)

    assert counts.tolist() == [1, 2, 0]


def test_hard_negative_weighting_is_off_by_default() -> None:
    logits = torch.tensor([[[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([1.0, 1.0, 1.0])

    default = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(type="cross_entropy"),
        class_counts=counts,
    )
    explicit_off = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(
            type="cross_entropy",
            hard_negative_mining=HardNegativeMiningSettings(enabled=False, weight=3.0),
        ),
        class_counts=counts,
    )

    assert explicit_off == pytest.approx(default)


def test_hard_negative_weighting_increases_target_false_positive_loss() -> None:
    logits = torch.tensor([[[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]]])
    labels = torch.tensor([[0, 1]])
    counts = torch.tensor([1.0, 1.0, 1.0])

    base = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(type="cross_entropy"),
        class_counts=counts,
    )
    mined = compute_dialogue_loss(
        logits,
        labels,
        settings=LossSettings(
            type="cross_entropy",
            hard_negative_mining=HardNegativeMiningSettings(
                enabled=True,
                weight=3.0,
                target_classes=(1,),
            ),
        ),
        class_counts=counts,
    )

    assert mined > base
