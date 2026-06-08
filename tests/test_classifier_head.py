"""Dialogue classifier head modes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch 미설치 (uv sync --extra deep 로 설치)")

from meld_emotion.models.classifier import EmotionClassifierHead  # noqa: E402


def _inputs() -> tuple[object, object, object]:
    return (
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 5),
        torch.randn(2, 3, 5),
    )


def test_concat_head_still_returns_logits_shape() -> None:
    fused, context, memory = _inputs()
    head = EmotionClassifierHead(
        fusion_dim=4,
        context_dim=5,
        memory_dim=5,
        hidden_dim=6,
        num_classes=7,
        classifier_head_type="concat",
    )

    logits = head(fused, context, memory)

    assert logits.shape == (2, 3, 7)
    assert head.last_alpha_context is None
    assert head.last_alpha_memory is None


def test_gated_residual_head_returns_logits_and_bounded_alphas() -> None:
    fused, context, memory = _inputs()
    head = EmotionClassifierHead(
        fusion_dim=4,
        context_dim=5,
        memory_dim=5,
        hidden_dim=6,
        num_classes=7,
        classifier_head_type="gated_residual",
    )

    logits = head(fused, context, memory)

    assert logits.shape == (2, 3, 7)
    assert head.last_alpha_context is not None
    assert head.last_alpha_memory is not None
    assert bool((head.last_alpha_context >= 0.0).all())
    assert bool((head.last_alpha_context <= 1.0).all())
    assert bool((head.last_alpha_memory >= 0.0).all())
    assert bool((head.last_alpha_memory <= 1.0).all())


def test_gated_residual_disabling_context_or_memory_zeroes_alpha() -> None:
    fused, context, memory = _inputs()
    head = EmotionClassifierHead(
        fusion_dim=4,
        context_dim=5,
        memory_dim=5,
        hidden_dim=6,
        num_classes=7,
        classifier_head_type="gated_residual",
        use_context=False,
        use_memory=False,
    )

    _ = head(fused, context, memory)

    assert head.last_alpha_context is not None
    assert head.last_alpha_memory is not None
    assert bool((head.last_alpha_context == 0.0).all())
    assert bool((head.last_alpha_memory == 0.0).all())


def test_fused_only_mode_matches_utterance_head_path() -> None:
    fused, context, memory = _inputs()
    head = EmotionClassifierHead(
        fusion_dim=4,
        context_dim=5,
        memory_dim=5,
        hidden_dim=6,
        num_classes=7,
        dropout=0.0,
        classifier_head_type="gated_residual",
        use_context=False,
        use_memory=False,
    )

    logits = head(fused, context, memory)
    expected = head.utterance_head(fused)

    assert torch.allclose(logits, expected)
