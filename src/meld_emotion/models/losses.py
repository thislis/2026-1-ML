"""Loss helpers for Dialogue RNN training."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from meld_emotion.config.schema import LossSettings

IGNORE_INDEX = -100
_EPS = 1.0e-12


def compute_dialogue_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    settings: LossSettings,
    class_counts: Tensor,
    class_weights: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Compute configured utterance loss while respecting padded labels."""

    if settings.type not in {"cross_entropy", "class_balanced_focal"}:
        raise ValueError("loss.type must be 'cross_entropy' or 'class_balanced_focal'")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    valid = flat_labels != ignore_index
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0

    adjusted = _apply_logit_adjustment(flat_logits, class_counts, settings)
    if settings.type == "cross_entropy":
        reduction_weights = class_weights
        per_sample = F.cross_entropy(
            adjusted,
            flat_labels,
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=settings.label_smoothing,
        )
    else:
        cb_weights = class_balanced_weights(
            class_counts,
            beta=settings.class_balanced_beta,
            device=flat_logits.device,
        )
        reduction_weights = cb_weights
        per_sample = F.cross_entropy(
            adjusted,
            flat_labels,
            weight=cb_weights,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=settings.label_smoothing,
        )
        true_prob = torch.softmax(adjusted, dim=-1)[
            torch.arange(flat_labels.numel(), device=flat_logits.device),
            flat_labels.clamp_min(0),
        ]
        focal = (1.0 - true_prob).clamp_min(0.0).pow(settings.gamma)
        per_sample = per_sample * focal

    per_sample = _apply_hard_negative_weights(adjusted, flat_labels, per_sample, valid, settings)
    return _weighted_mean(per_sample, flat_labels, valid, reduction_weights)


def class_balanced_weights(class_counts: Tensor, *, beta: float, device: torch.device) -> Tensor:
    """Effective-number class weights normalized to mean 1 over present classes."""

    if beta < 0.0 or beta >= 1.0:
        raise ValueError("loss.class_balanced_beta must be in [0, 1)")
    counts = class_counts.to(device=device, dtype=torch.float32).clamp_min(0.0)
    weights = torch.zeros_like(counts)
    present = counts > 0.0
    if bool(present.any()):
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights[present] = (1.0 - beta) / effective[present].clamp_min(_EPS)
        weights[present] = weights[present] / weights[present].mean().clamp_min(_EPS)
    return weights


def false_positive_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Count false positives by predicted class."""

    counts = np.zeros(n_classes, dtype=np.int64)
    for truth, pred in zip(y_true.astype(np.int64), y_pred.astype(np.int64), strict=True):
        if truth == ignore_index:
            continue
        if pred != truth and 0 <= pred < n_classes:
            counts[pred] += 1
    return counts


def _apply_logit_adjustment(logits: Tensor, class_counts: Tensor, settings: LossSettings) -> Tensor:
    if not settings.logit_adjustment.enabled:
        return logits
    counts = class_counts.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    total = counts.sum().clamp_min(_EPS)
    priors = (counts / total).clamp_min(_EPS)
    return logits + float(settings.logit_adjustment.tau) * torch.log(priors)


def _apply_hard_negative_weights(
    logits: Tensor,
    labels: Tensor,
    per_sample: Tensor,
    valid: Tensor,
    settings: LossSettings,
) -> Tensor:
    hard_negative = settings.hard_negative_mining
    if not hard_negative.enabled:
        return per_sample
    if hard_negative.weight < 0.0:
        raise ValueError("loss.hard_negative_mining.weight cannot be negative")
    pred = torch.argmax(logits.detach(), dim=-1)
    wrong = (pred != labels) & valid
    if hard_negative.target_classes:
        targets = torch.as_tensor(
            hard_negative.target_classes,
            dtype=pred.dtype,
            device=pred.device,
        )
        target_match = (pred[:, None] == targets[None, :]).any(dim=1)
        wrong = wrong & target_match
    weights = torch.ones_like(per_sample)
    weights = torch.where(wrong, torch.full_like(weights, float(hard_negative.weight)), weights)
    return per_sample * weights


def _weighted_mean(
    per_sample: Tensor,
    labels: Tensor,
    valid: Tensor,
    class_weights: Tensor | None,
) -> Tensor:
    if class_weights is None:
        return per_sample[valid].mean()
    sample_weights = class_weights.to(device=per_sample.device, dtype=per_sample.dtype)[
        labels[valid]
    ]
    denom = sample_weights.sum().clamp_min(_EPS)
    return per_sample[valid].sum() / denom
