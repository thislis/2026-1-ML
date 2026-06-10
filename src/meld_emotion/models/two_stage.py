"""Two-stage Neutral/Non-Neutral then emotion classifier wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import UID, Emotion, FloatArray, IntArray


@dataclass(frozen=True)
class TwoStageDecision:
    """Machine-readable decision trace for the hierarchical classifier."""

    uid: UID
    neutral_probability: float
    non_neutral_probability: float
    stage1_label: str
    stage2_label: Emotion | None
    final_label: Emotion
    final_probability: float
    emotion_scores: Mapping[Emotion, float]
    rationale: str


@real
class TwoStageEmotionClassifier:
    """Expose the requested Model 1/Model 2 decision flow around any base classifier.

    The wrapped classifier estimates the seven MELD emotion probabilities. Model 1 interprets
    the neutral class probability as ``P(neutral)`` and ``1 - P(neutral)`` as
    ``P(non-neutral)``. Model 2 is the conditional non-neutral classifier obtained by
    renormalizing the non-neutral emotion probabilities. This keeps the implementation
    compatible with existing early/late/dialogue gated models while making the hierarchical
    decision explicit and auditable.
    """

    def __init__(
        self,
        base: Classifier,
        neutral_threshold: float = 0.5,
        neutral_label: Emotion = Emotion.NEUTRAL,
    ) -> None:
        if not 0.0 <= neutral_threshold <= 1.0:
            raise ValueError("neutral_threshold must be in [0, 1]")
        if neutral_label not in base.classes:
            raise ValueError(f"neutral_label {neutral_label.value!r} is not in classifier classes")
        self._base = base
        self._threshold = neutral_threshold
        self._neutral_label = neutral_label
        self._last_decisions: tuple[TwoStageDecision, ...] = ()

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._base.classes

    @property
    def base(self) -> Classifier:
        return self._base

    @property
    def last_two_stage_decisions(self) -> tuple[TwoStageDecision, ...]:
        return self._last_decisions

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        self._base.fit(bundle, y)
        self._last_decisions = ()
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._normalized(self._base.predict_proba(bundle))

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        decisions = self.stage_outputs(bundle, proba=proba)
        index = {emotion: idx for idx, emotion in enumerate(self.classes)}
        y_pred = np.asarray([index[item.final_label] for item in decisions], dtype=np.int64)
        self._last_decisions = decisions
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self.classes)

    def stage_outputs(
        self, bundle: FeatureBundle, proba: FloatArray | None = None
    ) -> tuple[TwoStageDecision, ...]:
        scores = self.predict_proba(bundle) if proba is None else self._normalized(proba)
        neutral_idx = self.classes.index(self._neutral_label)
        decisions = tuple(
            self._decision_for_row(uid, scores[row], neutral_idx)
            for row, uid in enumerate(bundle.uids)
        )
        self._last_decisions = decisions
        return decisions

    def _decision_for_row(
        self, uid: UID, row: FloatArray, neutral_idx: int
    ) -> TwoStageDecision:
        neutral_probability = float(row[neutral_idx])
        non_neutral_probability = float(max(0.0, 1.0 - neutral_probability))
        emotion_indices = [idx for idx in range(len(self.classes)) if idx != neutral_idx]
        emotion_mass = float(row[emotion_indices].sum())
        if emotion_indices and emotion_mass > 0.0:
            stage2_idx = max(emotion_indices, key=lambda idx: float(row[idx] / emotion_mass))
        elif emotion_indices:
            stage2_idx = max(emotion_indices, key=lambda idx: float(row[idx]))
        else:
            stage2_idx = neutral_idx
        stage2_label = self.classes[stage2_idx] if stage2_idx != neutral_idx else None
        is_non_neutral = non_neutral_probability >= self._threshold
        final_label = stage2_label if is_non_neutral and stage2_label is not None else self._neutral_label
        final_probability = (
            float(row[stage2_idx]) if final_label != self._neutral_label else neutral_probability
        )
        stage1_label = "non_neutral" if is_non_neutral else "neutral"
        emotion_scores = {
            self.classes[idx]: (
                float(row[idx] / emotion_mass)
                if idx != neutral_idx and emotion_mass > 0.0
                else float(row[idx])
            )
            for idx in range(len(self.classes))
            if idx != neutral_idx
        }
        rationale = (
            f"Model 1 chose {stage1_label} with "
            f"P(non_neutral)={non_neutral_probability:.6f}; "
            f"Model 2 top emotion="
            f"{stage2_label.value if stage2_label is not None else 'n/a'}."
        )
        return TwoStageDecision(
            uid=uid,
            neutral_probability=neutral_probability,
            non_neutral_probability=non_neutral_probability,
            stage1_label=stage1_label,
            stage2_label=stage2_label,
            final_label=final_label,
            final_probability=final_probability,
            emotion_scores=emotion_scores,
            rationale=rationale,
        )

    def _normalized(self, proba: FloatArray) -> FloatArray:
        values = np.asarray(proba, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.classes):
            raise ValueError(
                "base classifier probability shape must be (n_samples, n_classes): "
                f"{values.shape} != (*, {len(self.classes)})"
            )
        row_sum = values.sum(axis=1, keepdims=True)
        safe = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0.0)
        empty = row_sum[:, 0] <= 0.0
        if np.any(empty):
            safe[empty] = 1.0 / len(self.classes)
        return cast(FloatArray, safe.astype(np.float64))
