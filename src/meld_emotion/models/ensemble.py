"""Artifact-backed ensemble classifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Self

import numpy as np

from meld_emotion.config.schema import EnsembleSettings
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import Emotion, FloatArray, IntArray


@real
class ArtifactEnsembleClassifier:
    """Combine a neural/base classifier with SVM/LogReg probability artifacts."""

    def __init__(
        self,
        base: Classifier,
        settings: EnsembleSettings,
        classes: tuple[Emotion, ...],
    ) -> None:
        self._base = base
        self._settings = settings
        self._classes = classes
        if settings.mode not in {"late_logits", "residual_correction"}:
            raise ValueError("ensemble.mode must be 'late_logits' or 'residual_correction'")

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        distillation = self._settings.distillation
        if distillation.enabled:
            if distillation.teacher_probs_path is None:
                raise ValueError("ensemble.distillation.teacher_probs_path is required")
            teacher_probs = _load_probability_artifact(
                distillation.teacher_probs_path,
                bundle.uids,
                len(self._classes),
            )
            fit_with_distillation = getattr(self._base, "fit_with_distillation", None)
            if not callable(fit_with_distillation):
                raise TypeError(
                    "ensemble.distillation requires a base classifier with fit_with_distillation"
                )
            fit_with_distillation(
                bundle,
                y,
                teacher_probs,
                temperature=distillation.temperature,
                weight=distillation.weight,
            )
            return self
        self._base.fit(bundle, y)
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        base_probs = self._base.predict_proba(bundle)
        base_logits = _probs_to_logits(base_probs)
        settings = self._settings
        if settings.mode == "late_logits":
            logits = settings.alpha * base_logits
            if settings.beta:
                logits = logits + settings.beta * self._required_artifact_logits(
                    settings.svm_logits_path,
                    bundle,
                    "svm_logits_path",
                )
            if settings.gamma:
                logits = logits + settings.gamma * self._required_artifact_logits(
                    settings.logreg_logits_path,
                    bundle,
                    "logreg_logits_path",
                )
            return _softmax(logits)

        svm_logits = self._required_artifact_logits(
            settings.svm_logits_path,
            bundle,
            "svm_logits_path",
        )
        return _softmax(svm_logits + settings.alpha * base_logits)

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        return PredictionSet(
            uids=bundle.uids,
            y_pred=np.argmax(proba, axis=1).astype(np.int64),
            proba=proba,
            classes=self._classes,
        )

    def _required_artifact_logits(
        self,
        path: str | None,
        bundle: FeatureBundle,
        field_name: str,
    ) -> np.ndarray:
        if path is None:
            raise ValueError(f"ensemble.{field_name} is required for mode={self._settings.mode}")
        return _load_artifact(path, bundle.uids, len(self._classes), self._settings.artifact_format)


def _load_artifact(
    path: str,
    expected_uids: tuple[str, ...],
    n_classes: int,
    expected_kind: str,
) -> np.ndarray:
    values, kind = _read_artifact_values(path, expected_uids, n_classes, expected_kind)
    if kind == "proba":
        return _probs_to_logits(values)
    return values


def _load_probability_artifact(
    path: str,
    expected_uids: tuple[str, ...],
    n_classes: int,
) -> np.ndarray:
    values, _ = _read_artifact_values(path, expected_uids, n_classes, "proba")
    row_sums = values.sum(axis=1)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"ensemble teacher probabilities must be finite and non-negative: {path}")
    if np.any(row_sums <= 0.0):
        raise ValueError(f"ensemble teacher probability rows must have positive mass: {path}")
    return np.asarray(values / row_sums[:, None], dtype=np.float64)


def _read_artifact_values(
    path: str,
    expected_uids: tuple[str, ...],
    n_classes: int,
    expected_kind: str,
) -> tuple[np.ndarray, str]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"ensemble artifact not found: {artifact_path}")
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"ensemble artifact must be a JSON object: {artifact_path}")
    uids = data.get("uids")
    if not isinstance(uids, list) or tuple(str(uid) for uid in uids) != expected_uids:
        raise ValueError(f"ensemble artifact UID alignment mismatch: {artifact_path}")

    kind = _artifact_kind(data, expected_kind)
    values = np.asarray(data[kind], dtype=np.float64)
    expected_shape = (len(expected_uids), n_classes)
    if values.shape != expected_shape:
        raise ValueError(
            f"ensemble artifact {kind} shape mismatch: {values.shape} != {expected_shape}"
        )
    return values, kind


def _artifact_kind(data: Mapping[str, object], expected_kind: str) -> str:
    if expected_kind in {"logits", "proba"}:
        if expected_kind not in data:
            raise ValueError(f"ensemble artifact missing {expected_kind!r}")
        return expected_kind
    if expected_kind != "auto":
        raise ValueError("ensemble.artifact_format must be 'auto', 'logits', or 'proba'")
    if "logits" in data:
        return "logits"
    if "proba" in data:
        return "proba"
    raise ValueError("ensemble artifact must contain 'logits' or 'proba'")


def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"probabilities must be 2D: {values.shape}")
    return np.log(np.clip(values, 1.0e-12, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(centered)
    return np.asarray(
        exp / exp.sum(axis=1, keepdims=True).clip(min=1.0e-12),
        dtype=np.float64,
    )
