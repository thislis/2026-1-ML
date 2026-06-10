"""Lightweight mixture-of-experts emotion classifier."""

from __future__ import annotations

from typing import Self

import numpy as np

from meld_emotion.config.schema import MoeSettings
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, FloatArray, IntArray, Modality
from meld_emotion.models.ensemble import _load_artifact


@real
class MoeEmotionClassifier:
    """Configurable top-k MoE over modality, neutral, rare, and artifact experts."""

    def __init__(self, settings: MoeSettings, classes: tuple[Emotion, ...]) -> None:
        if settings.routing != "top2":
            raise ValueError("moe.routing currently supports 'top2'")
        if settings.top_k <= 0:
            raise ValueError("moe.top_k must be positive")
        if not 0.0 <= settings.expert_dropout <= 1.0:
            raise ValueError("moe.expert_dropout must be in [0, 1]")
        self._settings = settings
        self._classes = classes
        self._experts = self._active_experts()
        self._centroids: dict[str, np.ndarray] = {}
        self._priors = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
        self._last_gate_stats: dict[str, float] = {}

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    @property
    def last_gate_stats(self) -> dict[str, float]:
        return dict(self._last_gate_stats)

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        labels = np.asarray(y, dtype=np.int64)
        counts = np.bincount(labels, minlength=len(self._classes)).astype(np.float64)
        self._priors = (counts + 1.0) / (counts.sum() + len(self._classes))
        for expert in self._experts:
            if expert == "svm_logreg":
                continue
            design = self._design(bundle, expert)
            self._centroids[expert] = _class_centroids(design, labels, len(self._classes))
        self._last_gate_stats = {"expert_dropout_rate": float(self._settings.expert_dropout)}
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        per_expert = {expert: self._expert_logits(bundle, expert) for expert in self._experts}
        gates, selected = self._routes(bundle, per_expert)
        logits = np.zeros((bundle.n_samples, len(self._classes)), dtype=np.float64)
        for idx, expert in enumerate(self._experts):
            logits += gates[:, idx : idx + 1] * per_expert[expert]
        self._last_gate_stats = {
            **self._route_stats(gates, selected),
            "load_balancing_loss": _load_balance_loss(gates),
            "expert_dropout_rate": float(self._settings.expert_dropout),
        }
        return _softmax(logits)

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        return PredictionSet(
            uids=bundle.uids,
            y_pred=np.argmax(proba, axis=1).astype(np.int64),
            proba=proba,
            classes=self._classes,
        )

    def _active_experts(self) -> tuple[str, ...]:
        flags = self._settings.experts
        names = (
            ("text", flags.text),
            ("audio", flags.audio),
            ("video", flags.video),
            ("context", flags.context),
            ("neutral", flags.neutral),
            ("rare", flags.rare and self._settings.rare_expert.enabled),
            ("svm_logreg", flags.svm_logreg),
        )
        active = tuple(name for name, enabled in names if enabled)
        if not active:
            raise ValueError("at least one MoE expert must be enabled")
        return active

    def _design(self, bundle: FeatureBundle, expert: str) -> np.ndarray:
        if expert == "text":
            return bundle.stack(modalities=(Modality.TEXT,)).values
        if expert == "audio":
            return bundle.stack(modalities=(Modality.AUDIO,)).values
        if expert == "video":
            return bundle.stack(modalities=(Modality.VIDEO,)).values
        return bundle.stack().values

    def _expert_logits(self, bundle: FeatureBundle, expert: str) -> np.ndarray:
        if expert == "svm_logreg":
            chunks = []
            if self._settings.svm_logits_path is not None:
                chunks.append(
                    _load_artifact(
                        self._settings.svm_logits_path,
                        bundle.uids,
                        len(self._classes),
                        self._settings.artifact_format,
                    )
                )
            if self._settings.logreg_logits_path is not None:
                chunks.append(
                    _load_artifact(
                        self._settings.logreg_logits_path,
                        bundle.uids,
                        len(self._classes),
                        self._settings.artifact_format,
                    )
                )
            if chunks:
                return np.asarray(np.mean(np.stack(chunks, axis=0), axis=0), dtype=np.float64)
            return np.tile(np.log(self._priors), (bundle.n_samples, 1))
        design = self._design(bundle, expert)
        centroids = self._centroids.get(expert)
        if centroids is None or design.shape[1] == 0:
            return np.tile(np.log(self._priors), (bundle.n_samples, 1))
        diff = design[:, None, :] - centroids[None, :, :]
        logits = -np.sqrt(np.sum(diff * diff, axis=2))
        if expert == "neutral":
            logits[:, 0] += 1.0
        if expert == "rare":
            for class_idx in self._settings.rare_expert.target_classes:
                if 0 <= class_idx < logits.shape[1]:
                    logits[:, class_idx] += self._settings.rare_expert.loss_weight
        return np.asarray(logits, dtype=np.float64)

    def _routes(
        self,
        bundle: FeatureBundle,
        per_expert: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros((bundle.n_samples, len(self._experts)), dtype=np.float64)
        for idx, expert in enumerate(self._experts):
            raw[:, idx] = self._route_score(bundle, expert, per_expert[expert])
        top_k = min(self._settings.top_k, len(self._experts))
        selected = np.argsort(raw, axis=1)[:, -top_k:]
        masked = np.full_like(raw, -1.0e9)
        rows = np.arange(raw.shape[0])[:, None]
        masked[rows, selected] = raw[rows, selected]
        return _softmax(masked), np.asarray(selected, dtype=np.int64)

    def _route_score(
        self,
        bundle: FeatureBundle,
        expert: str,
        logits: np.ndarray,
    ) -> np.ndarray:
        if expert in {"text", "audio", "video"}:
            modality = Modality(expert)
            avail = bundle.availability.get(modality)
            available = np.ones(bundle.n_samples, dtype=np.float64) if avail is None else avail
            return available.astype(np.float64)
        if expert == "rare" and self._settings.class_aware_routing:
            rare = [c for c in self._settings.rare_expert.target_classes if 0 <= c < logits.shape[1]]
            if rare:
                return np.asarray(0.5 + np.max(_softmax(logits)[:, rare], axis=1), dtype=np.float64)
        return np.ones(bundle.n_samples, dtype=np.float64) * 0.5

    def _route_stats(self, gates: np.ndarray, selected: np.ndarray) -> dict[str, float]:
        stats: dict[str, float] = {}
        for idx, expert in enumerate(self._experts):
            stats[f"moe_gate_{expert}_mean"] = float(np.mean(gates[:, idx]))
            stats[f"moe_selected_{expert}_rate"] = float(np.mean(selected == idx))
        return stats


def _class_centroids(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((n_classes, 0), dtype=np.float64)
    centroids = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    global_mean = np.mean(x, axis=0)
    for class_idx in range(n_classes):
        rows = y == class_idx
        centroids[class_idx] = np.mean(x[rows], axis=0) if rows.any() else global_mean
    return centroids


def _load_balance_loss(gates: np.ndarray) -> float:
    usage = np.mean(gates, axis=0)
    target = np.full_like(usage, 1.0 / usage.size)
    return float(np.mean((usage - target) ** 2))


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(centered)
    return np.asarray(
        exp / exp.sum(axis=1, keepdims=True).clip(min=1.0e-12),
        dtype=np.float64,
    )
