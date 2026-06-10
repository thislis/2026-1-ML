"""Artifact-backed ensemble classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import numpy as np

from meld_emotion.config.loader import load_config
from meld_emotion.config.schema import EnsembleDistillationSettings, EnsembleSettings
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.types import Emotion, FloatArray, IntArray
from meld_emotion.models.ensemble import ArtifactEnsembleClassifier


class _DummyClassifier:
    def __init__(self, proba: np.ndarray, classes: tuple[Emotion, ...]) -> None:
        self._proba = proba
        self._classes = classes

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return np.asarray(self._proba[: bundle.n_samples], dtype=np.float64)

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        return PredictionSet(
            uids=bundle.uids,
            y_pred=np.argmax(proba, axis=1).astype(np.int64),
            proba=proba,
            classes=self._classes,
        )


class _DistillableDummyClassifier(_DummyClassifier):
    def __init__(self, proba: np.ndarray, classes: tuple[Emotion, ...]) -> None:
        super().__init__(proba, classes)
        self.distillation_seen: dict[str, object] | None = None

    def fit_with_distillation(
        self,
        bundle: FeatureBundle,
        y: IntArray,
        teacher_probs: FloatArray,
        *,
        temperature: float,
        weight: float,
    ) -> Self:
        self.distillation_seen = {
            "uids": bundle.uids,
            "y": np.asarray(y, dtype=np.int64),
            "teacher_probs": np.asarray(teacher_probs, dtype=np.float64),
            "temperature": temperature,
            "weight": weight,
        }
        return self


def _artifact(path: Path, uids: tuple[str, ...], key: str, values: list[list[float]]) -> None:
    path.write_text(json.dumps({"uids": list(uids), key: values}), encoding="utf-8")


def test_artifact_ensemble_late_logits_combines_aligned_artifacts(
    test_bundle: FeatureBundle,
    tmp_path: Path,
) -> None:
    classes = tuple(Emotion)
    uids = test_bundle.uids
    svm_path = tmp_path / "svm.json"
    logreg_path = tmp_path / "logreg.json"
    _artifact(svm_path, uids, "proba", [[0.9, 0.1, 0, 0, 0, 0, 0]] * test_bundle.n_samples)
    _artifact(logreg_path, uids, "proba", [[0.8, 0.2, 0, 0, 0, 0, 0]] * test_bundle.n_samples)
    base = _DummyClassifier(
        np.tile([0.1, 0.9, 0, 0, 0, 0, 0], (test_bundle.n_samples, 1)),
        classes,
    )
    clf = ArtifactEnsembleClassifier(
        base,
        EnsembleSettings(
            mode="late_logits",
            alpha=0.0,
            beta=1.0,
            gamma=1.0,
            svm_logits_path=str(svm_path),
            logreg_logits_path=str(logreg_path),
            artifact_format="proba",
        ),
        classes,
    )

    pred = clf.fit(test_bundle, np.zeros(test_bundle.n_samples, dtype=np.int64)).predict(
        test_bundle
    )

    assert pred.y_pred.tolist() == [0] * test_bundle.n_samples


def test_artifact_ensemble_rejects_uid_mismatch(test_bundle: FeatureBundle, tmp_path: Path) -> None:
    classes = tuple(Emotion)
    path = tmp_path / "bad.json"
    _artifact(path, tuple(reversed(test_bundle.uids)), "proba", [[1, 0, 0, 0, 0, 0, 0]] * test_bundle.n_samples)
    clf = ArtifactEnsembleClassifier(
        _DummyClassifier(np.tile([1, 0, 0, 0, 0, 0, 0], (test_bundle.n_samples, 1)), classes),
        EnsembleSettings(mode="late_logits", alpha=0.0, beta=1.0, svm_logits_path=str(path)),
        classes,
    )

    try:
        clf.predict_proba(test_bundle)
    except ValueError as exc:
        assert "UID alignment mismatch" in str(exc)
    else:
        raise AssertionError("UID mismatch should fail")


def test_artifact_ensemble_distillation_requires_teacher_path(test_bundle: FeatureBundle) -> None:
    classes = tuple(Emotion)
    clf = ArtifactEnsembleClassifier(
        _DummyClassifier(np.tile([1, 0, 0, 0, 0, 0, 0], (test_bundle.n_samples, 1)), classes),
        EnsembleSettings(
            mode="late_logits",
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            distillation=EnsembleDistillationSettings(enabled=True),
        ),
        classes,
    )

    try:
        clf.fit(test_bundle, np.zeros(test_bundle.n_samples, dtype=np.int64))
    except ValueError as exc:
        assert "teacher_probs_path is required" in str(exc)
    else:
        raise AssertionError("missing distillation teacher should fail")


def test_artifact_ensemble_distillation_passes_teacher_probs(
    test_bundle: FeatureBundle,
    tmp_path: Path,
) -> None:
    classes = tuple(Emotion)
    teacher_path = tmp_path / "teacher.json"
    _artifact(
        teacher_path,
        test_bundle.uids,
        "proba",
        [[0.2, 0.8, 0, 0, 0, 0, 0]] * test_bundle.n_samples,
    )
    base = _DistillableDummyClassifier(
        np.tile([1, 0, 0, 0, 0, 0, 0], (test_bundle.n_samples, 1)),
        classes,
    )
    clf = ArtifactEnsembleClassifier(
        base,
        EnsembleSettings(
            mode="late_logits",
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            distillation=EnsembleDistillationSettings(
                enabled=True,
                teacher_probs_path=str(teacher_path),
                temperature=3.0,
                weight=0.4,
            ),
        ),
        classes,
    )

    clf.fit(test_bundle, np.arange(test_bundle.n_samples, dtype=np.int64) % len(classes))

    assert base.distillation_seen is not None
    assert base.distillation_seen["uids"] == test_bundle.uids
    assert base.distillation_seen["temperature"] == 3.0
    assert base.distillation_seen["weight"] == 0.4
    np.testing.assert_allclose(
        base.distillation_seen["teacher_probs"],
        np.tile([0.2, 0.8, 0, 0, 0, 0, 0], (test_bundle.n_samples, 1)),
    )


def test_ensemble_config_loads() -> None:
    cfg = load_config("configs/conformer_sequence_ensemble.yaml")
    assert cfg.model.type == "ensemble"
