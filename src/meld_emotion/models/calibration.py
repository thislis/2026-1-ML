"""Prediction calibration and deterministic postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

_EPS = 1.0e-8


@dataclass(frozen=True)
class CalibrationParams:
    temperature: float = 1.0
    class_thresholds: tuple[float, ...] = ()
    rare_classes: tuple[int, ...] = ()
    rare_class_threshold: float = 0.0
    rare_class_margin: float = 0.0
    neutral_gate_enabled: bool = False
    neutral_class_index: int = 0
    neutral_emotion_threshold: float = 0.5
    neutral_gate_tuned: bool = False
    neutral_gate_before_accuracy: float = 0.0
    neutral_gate_after_accuracy: float = 0.0
    class_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "class_thresholds": list(self.class_thresholds),
            "rare_classes": list(self.rare_classes),
            "rare_class_threshold": self.rare_class_threshold,
            "rare_class_margin": self.rare_class_margin,
            "neutral_gate_enabled": self.neutral_gate_enabled,
            "neutral_class_index": self.neutral_class_index,
            "neutral_emotion_threshold": self.neutral_emotion_threshold,
            "neutral_gate_tuned": self.neutral_gate_tuned,
            "neutral_gate_before_accuracy": self.neutral_gate_before_accuracy,
            "neutral_gate_after_accuracy": self.neutral_gate_after_accuracy,
            "class_labels": list(self.class_labels),
        }

    @classmethod
    def from_dict(cls, data: object) -> CalibrationParams:
        if not isinstance(data, dict):
            raise ValueError("calibration params must be a mapping")
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            class_thresholds=tuple(float(v) for v in data.get("class_thresholds", ())),
            rare_classes=tuple(int(v) for v in data.get("rare_classes", ())),
            rare_class_threshold=float(data.get("rare_class_threshold", 0.0)),
            rare_class_margin=float(data.get("rare_class_margin", 0.0)),
            neutral_gate_enabled=bool(data.get("neutral_gate_enabled", False)),
            neutral_class_index=int(data.get("neutral_class_index", 0)),
            neutral_emotion_threshold=float(data.get("neutral_emotion_threshold", 0.5)),
            neutral_gate_tuned=bool(data.get("neutral_gate_tuned", False)),
            neutral_gate_before_accuracy=float(data.get("neutral_gate_before_accuracy", 0.0)),
            neutral_gate_after_accuracy=float(data.get("neutral_gate_after_accuracy", 0.0)),
            class_labels=tuple(str(v) for v in data.get("class_labels", ())),
        )


class PredictionPostprocessor:
    """Apply temperature scaling, class thresholds, and rare-class margin overrides."""

    def __init__(self, params: CalibrationParams | None = None) -> None:
        self.params = params if params is not None else CalibrationParams()

    def calibrate_logits(self, logits: Tensor) -> Tensor:
        temperature = max(float(self.params.temperature), _EPS)
        return logits / temperature

    def probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(self.calibrate_logits(logits), dim=-1)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"probabilities must be 2D: ndim={probs.ndim}")
        base = np.argmax(probs, axis=1).astype(np.int64)
        thresholded = self._apply_thresholds(probs, base)
        rare_adjusted = self._apply_rare_margin(probs, thresholded)
        return self._apply_neutral_gate(probs, rare_adjusted)

    def _apply_thresholds(self, probs: np.ndarray, base: np.ndarray) -> np.ndarray:
        thresholds = self.params.class_thresholds
        if not thresholds:
            return base
        if len(thresholds) != probs.shape[1]:
            raise ValueError(
                f"class_thresholds length mismatch: {len(thresholds)} != {probs.shape[1]}"
            )
        result = base.copy()
        threshold_arr = np.asarray(thresholds, dtype=np.float64)
        eligible = probs >= threshold_arr[None, :]
        for row in range(probs.shape[0]):
            classes = np.flatnonzero(eligible[row])
            if classes.size == 0:
                continue
            margins = probs[row, classes] - threshold_arr[classes]
            best_margin = margins.max()
            tied = classes[np.flatnonzero(np.isclose(margins, best_margin))]
            result[row] = int(tied[np.argmax(probs[row, tied])])
        return result

    def _apply_neutral_gate(self, probs: np.ndarray, current: np.ndarray) -> np.ndarray:
        if not self.params.neutral_gate_enabled:
            return current
        neutral = int(self.params.neutral_class_index)
        if neutral < 0 or neutral >= probs.shape[1]:
            raise ValueError(
                f"neutral_class_index out of range: {neutral} for {probs.shape[1]} classes"
            )
        threshold = float(self.params.neutral_emotion_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("neutral_emotion_threshold must be in [0, 1]")
        result = current.copy()
        p_emotion = 1.0 - probs[:, neutral]
        result[p_emotion < threshold] = neutral
        return result

    def _apply_rare_margin(self, probs: np.ndarray, current: np.ndarray) -> np.ndarray:
        if not self.params.rare_classes:
            return current
        result = current.copy()
        rare = np.asarray(self.params.rare_classes, dtype=np.int64)
        rare = rare[(rare >= 0) & (rare < probs.shape[1])]
        if rare.size == 0:
            return result
        top_prob = probs.max(axis=1)
        for row in range(probs.shape[0]):
            rare_probs = probs[row, rare]
            best_idx = int(np.argmax(rare_probs))
            rare_class = int(rare[best_idx])
            rare_prob = float(rare_probs[best_idx])
            if (
                rare_prob >= self.params.rare_class_threshold
                and top_prob[row] - rare_prob <= self.params.rare_class_margin
            ):
                result[row] = rare_class
        return result


def fit_temperature(logits: Tensor, labels: Tensor, *, max_iter: int = 50) -> float:
    """Fit a positive temperature by minimizing NLL on dev logits."""

    if logits.numel() == 0:
        return 1.0
    log_temperature = torch.zeros((), dtype=logits.dtype, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp_min(_EPS)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return float(torch.exp(log_temperature.detach()).clamp_min(_EPS).cpu().item())


def tune_class_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, ...]:
    """Choose one-vs-rest thresholds that maximize per-class F1 on dev probabilities."""

    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    thresholds: list[float] = []
    for class_idx in range(probs.shape[1]):
        best_threshold = 1.0
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            pred_pos = probs[:, class_idx] >= threshold
            true_pos = y == class_idx
            tp = float((pred_pos & true_pos).sum())
            fp = float((pred_pos & ~true_pos).sum())
            fn = float((~pred_pos & true_pos).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return tuple(thresholds)


def tune_neutral_emotion_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    neutral_class_index: int = 0,
) -> float:
    """Tune p(non-neutral) threshold for neutral override by validation accuracy."""

    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2D")
    if neutral_class_index < 0 or neutral_class_index >= probs.shape[1]:
        raise ValueError("neutral_class_index out of range")
    base = np.argmax(probs, axis=1).astype(np.int64)
    p_emotion = 1.0 - probs[:, neutral_class_index]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = base.copy()
        pred[p_emotion < threshold] = neutral_class_index
        score = float(np.mean(pred == y)) if y.size else 0.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
