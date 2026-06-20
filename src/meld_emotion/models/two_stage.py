"""Two-stage Neutral/Non-Neutral then emotion classifier wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier, Estimator
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import UID, Emotion, FeatureKind, FloatArray, IntArray


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
    stage1_margin: float = 0.0
    routed_to_stage2: bool = False
    stage1_model_label: Emotion | None = None
    stage1_confidence: float = 0.0


@dataclass(frozen=True)
class SvmStagePath:
    """Detailed hard-routing trace for SVM-only hierarchical classifiers."""

    uid: UID
    stages: Mapping[str, str]
    final_label: Emotion
    final_probability: float


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


@real
class SvmMarginTwoStageClassifier:
    """Route confident 7-class SVM predictions and defer low-margin cases to Stage 2."""

    def __init__(
        self,
        stage1_factory: Callable[[int], Estimator],
        stage2: Classifier,
        classes: tuple[Emotion, ...],
        margin_threshold: float = 0.25,
        stage1_confidence_threshold: float | None = None,
        stage1_use_concepts: bool = True,
        neutral_label: Emotion = Emotion.NEUTRAL,
    ) -> None:
        if margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if stage1_confidence_threshold is not None and not 0.0 <= stage1_confidence_threshold <= 1.0:
            raise ValueError("stage1_confidence_threshold must be in [0, 1]")
        if neutral_label not in classes:
            raise ValueError(f"neutral_label {neutral_label.value!r} is not in classifier classes")
        self._stage1 = stage1_factory(len(classes))
        self._stage2 = stage2
        self._classes = classes
        self._margin_threshold = margin_threshold
        self._stage1_confidence_threshold = stage1_confidence_threshold
        self._stage1_use_concepts = stage1_use_concepts
        self._neutral_label = neutral_label
        self._last_decisions: tuple[TwoStageDecision, ...] = ()

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    @property
    def last_two_stage_decisions(self) -> tuple[TwoStageDecision, ...]:
        return self._last_decisions

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        self._stage1.fit(self._stage1_design(bundle), y)
        self._stage2.fit(bundle, y)
        self._last_decisions = ()
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._route(bundle)[0]

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba, decisions = self._route(bundle)
        index = {emotion: idx for idx, emotion in enumerate(self._classes)}
        y_pred = np.asarray([index[item.final_label] for item in decisions], dtype=np.int64)
        self._last_decisions = decisions
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def stage_outputs(
        self, bundle: FeatureBundle, proba: FloatArray | None = None
    ) -> tuple[TwoStageDecision, ...]:
        if proba is not None:
            # Decisions depend on SVM margins as well as probabilities, so recompute the route.
            # The argument is accepted for inference compatibility with TwoStageEmotionClassifier.
            pass
        _, decisions = self._route(bundle)
        self._last_decisions = decisions
        return decisions

    def _stage1_design(self, bundle: FeatureBundle) -> FloatArray:
        if self._stage1_use_concepts:
            return bundle.stack().values
        return bundle.stack(kind=FeatureKind.EMBEDDING).values

    def _route(self, bundle: FeatureBundle) -> tuple[FloatArray, tuple[TwoStageDecision, ...]]:
        design = self._stage1_design(bundle)
        stage1_proba = _normalize_rows(self._stage1.predict_proba(design), len(self._classes))
        stage1_scores = self._stage1_scores(design, stage1_proba)
        stage2_proba = _normalize_rows(self._stage2.predict_proba(bundle), len(self._classes))
        neutral_idx = self._classes.index(self._neutral_label)

        rows: list[FloatArray] = []
        decisions: list[TwoStageDecision] = []
        for row_idx, uid in enumerate(bundle.uids):
            margin = _top_margin(stage1_scores[row_idx])
            confidence = float(np.max(stage1_proba[row_idx]))
            routed = self._should_route(margin, confidence)
            stage1_idx = int(np.argmax(stage1_scores[row_idx]))
            stage1_label = self._classes[stage1_idx]
            stage2_row = _non_neutral_distribution(stage2_proba[row_idx], neutral_idx)
            if routed:
                final_idx = int(np.argmax(stage2_row))
                final_label = self._classes[final_idx]
                final_row = stage2_row
                stage1_state = "uncertain"
            else:
                final_idx = stage1_idx
                final_label = stage1_label
                final_row = stage1_proba[row_idx]
                stage1_state = (
                    "neutral" if stage1_label == self._neutral_label else "non_neutral"
                )
            stage2_idx = int(np.argmax(stage2_row))
            stage2_label = self._classes[stage2_idx] if stage2_idx != neutral_idx else None
            final_probability = float(final_row[final_idx])
            rows.append(final_row)
            decisions.append(
                self._decision(
                    uid=uid,
                    final_row=final_row,
                    neutral_idx=neutral_idx,
                    stage1_label=stage1_label,
                    stage1_state=stage1_state,
                    stage2_label=stage2_label,
                    final_label=final_label,
                    final_probability=final_probability,
                    margin=margin,
                    confidence=confidence,
                    routed=routed,
                )
            )
        return np.vstack(rows).astype(np.float64), tuple(decisions)

    def _should_route(self, margin: float, confidence: float) -> bool:
        if self._stage1_confidence_threshold is not None:
            return confidence < self._stage1_confidence_threshold
        return margin < self._margin_threshold

    def _stage1_scores(self, design: FloatArray, fallback: FloatArray) -> FloatArray:
        decision_scores = getattr(self._stage1, "decision_scores", None)
        if callable(decision_scores):
            scores = np.asarray(decision_scores(design), dtype=np.float64)
        else:
            scores = np.asarray(fallback, dtype=np.float64)
        if scores.shape != (design.shape[0], len(self._classes)):
            raise ValueError(
                "stage1 decision scores shape mismatch: "
                f"{scores.shape} != {(design.shape[0], len(self._classes))}"
            )
        return scores

    def _decision(
        self,
        *,
        uid: UID,
        final_row: FloatArray,
        neutral_idx: int,
        stage1_label: Emotion,
        stage1_state: str,
        stage2_label: Emotion | None,
        final_label: Emotion,
        final_probability: float,
        margin: float,
        confidence: float,
        routed: bool,
    ) -> TwoStageDecision:
        neutral_probability = float(final_row[neutral_idx])
        non_neutral_probability = float(max(0.0, 1.0 - neutral_probability))
        emotion_scores = {
            self._classes[idx]: float(final_row[idx])
            for idx in range(len(self._classes))
            if idx != neutral_idx
        }
        route = "Stage 2" if routed else "Stage 1"
        if self._stage1_confidence_threshold is None:
            criterion = (
                f"margin={margin:.6f} "
                f"({'<' if routed else '>='} {self._margin_threshold:.6f})"
            )
        else:
            criterion = (
                f"confidence={confidence:.6f} "
                f"({'<' if routed else '>='} {self._stage1_confidence_threshold:.6f})"
            )
        rationale = f"Stage 1 SVM {criterion}; {route} chose {final_label.value}."
        return TwoStageDecision(
            uid=uid,
            neutral_probability=neutral_probability,
            non_neutral_probability=non_neutral_probability,
            stage1_label=stage1_state,
            stage2_label=stage2_label,
            final_label=final_label,
            final_probability=final_probability,
            emotion_scores=emotion_scores,
            rationale=rationale,
            stage1_margin=margin,
            routed_to_stage2=routed,
            stage1_model_label=stage1_label,
            stage1_confidence=confidence,
        )


class _LocalStageClassifier:
    """Train one local-label stage, with deterministic sparse-data fallback."""

    def __init__(self, factory: Callable[[int], Estimator], labels: Sequence[str]) -> None:
        if not labels:
            raise ValueError("stage labels must not be empty")
        self._labels = tuple(labels)
        self._label_to_index = {label: idx for idx, label in enumerate(self._labels)}
        self._model: Estimator | None = None
        self._constant: str | None = None
        self._uniform = False
        self._factory = factory

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def fit(self, x: FloatArray, y: Sequence[str]) -> Self:
        values = tuple(y)
        if not values:
            self._model = None
            self._constant = None
            self._uniform = True
            return self
        unknown = sorted(set(values).difference(self._label_to_index))
        if unknown:
            raise ValueError(f"unknown local stage labels: {unknown!r}")
        unique = tuple(dict.fromkeys(values))
        if len(unique) == 1:
            self._model = None
            self._constant = unique[0]
            self._uniform = False
            return self
        local_y = np.asarray([self._label_to_index[label] for label in values], dtype=np.int64)
        self._model = self._factory(len(self._labels)).fit(x, local_y)
        self._constant = None
        self._uniform = False
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        n_samples = x.shape[0]
        n_labels = len(self._labels)
        if n_samples == 0:
            return np.zeros((0, n_labels), dtype=np.float64)
        if self._uniform:
            return np.full((n_samples, n_labels), 1.0 / n_labels, dtype=np.float64)
        if self._constant is not None:
            values = np.zeros((n_samples, n_labels), dtype=np.float64)
            values[:, self._label_to_index[self._constant]] = 1.0
            return values
        if self._model is None:
            raise RuntimeError("local stage classifier is not fitted")
        return _normalize_rows(self._model.predict_proba(x), n_labels)


@real
class SvmTwoStageClassifier:
    """Train separate SVMs for Neutral/Non-Neutral and non-neutral emotion."""

    def __init__(
        self,
        stage_factory: Callable[[int], Estimator],
        classes: tuple[Emotion, ...],
        use_concepts: bool = True,
        neutral_label: Emotion = Emotion.NEUTRAL,
    ) -> None:
        if neutral_label not in classes:
            raise ValueError(f"neutral_label {neutral_label.value!r} is not in classifier classes")
        self._classes = classes
        self._use_concepts = use_concepts
        self._neutral_label = neutral_label
        self._stage1 = _LocalStageClassifier(stage_factory, ("neutral", "non_neutral"))
        self._non_neutral_labels = tuple(
            emotion.value for emotion in classes if emotion != neutral_label
        )
        self._stage2 = _LocalStageClassifier(stage_factory, self._non_neutral_labels)
        self._last_decisions: tuple[TwoStageDecision, ...] = ()
        self._last_paths: tuple[SvmStagePath, ...] = ()

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    @property
    def last_two_stage_decisions(self) -> tuple[TwoStageDecision, ...]:
        return self._last_decisions

    @property
    def last_svm_stage_paths(self) -> tuple[SvmStagePath, ...]:
        return self._last_paths

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        design = self._design(bundle)
        labels = tuple(self._classes[int(idx)] for idx in y)
        stage1_y = tuple(
            "neutral" if label == self._neutral_label else "non_neutral" for label in labels
        )
        non_neutral_mask = np.asarray(
            [label != self._neutral_label for label in labels], dtype=np.bool_
        )
        stage2_y = tuple(label.value for label in labels if label != self._neutral_label)
        self._stage1.fit(design, stage1_y)
        self._stage2.fit(design[non_neutral_mask], stage2_y)
        self._last_decisions = ()
        self._last_paths = ()
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._route(bundle)[0]

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba, decisions, paths = self._route(bundle)
        index = {emotion: idx for idx, emotion in enumerate(self._classes)}
        y_pred = np.asarray([index[item.final_label] for item in decisions], dtype=np.int64)
        self._last_decisions = decisions
        self._last_paths = paths
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def stage_outputs(
        self, bundle: FeatureBundle, proba: FloatArray | None = None
    ) -> tuple[TwoStageDecision, ...]:
        if proba is not None:
            # SVM hard-routing is part of the explanation, so recompute from stages.
            pass
        _, decisions, paths = self._route(bundle)
        self._last_decisions = decisions
        self._last_paths = paths
        return decisions

    def _design(self, bundle: FeatureBundle) -> FloatArray:
        if self._use_concepts:
            return bundle.stack().values
        return bundle.stack(kind=FeatureKind.EMBEDDING).values

    def _route(
        self, bundle: FeatureBundle
    ) -> tuple[FloatArray, tuple[TwoStageDecision, ...], tuple[SvmStagePath, ...]]:
        design = self._design(bundle)
        stage1 = self._stage1.predict_proba(design)
        stage2 = self._stage2.predict_proba(design)
        neutral_idx = self._classes.index(self._neutral_label)
        emotion_to_index = {emotion: idx for idx, emotion in enumerate(self._classes)}
        label_to_emotion = {emotion.value: emotion for emotion in self._classes}
        rows: list[FloatArray] = []
        decisions: list[TwoStageDecision] = []
        paths: list[SvmStagePath] = []
        for row_idx, uid in enumerate(bundle.uids):
            row = np.zeros(len(self._classes), dtype=np.float64)
            neutral_probability = float(stage1[row_idx, 0])
            non_neutral_probability = float(stage1[row_idx, 1])
            row[neutral_idx] = neutral_probability
            for local_idx, label in enumerate(self._non_neutral_labels):
                emotion = label_to_emotion[label]
                row[emotion_to_index[emotion]] = non_neutral_probability * float(
                    stage2[row_idx, local_idx]
                )
            row = _normalize_vector(row)
            stage1_label = self._stage1.labels[int(np.argmax(stage1[row_idx]))]
            stage2_label_text = self._stage2.labels[int(np.argmax(stage2[row_idx]))]
            stage2_emotion = label_to_emotion[stage2_label_text]
            final_label = self._neutral_label if stage1_label == "neutral" else stage2_emotion
            final_probability = float(row[emotion_to_index[final_label]])
            path = SvmStagePath(
                uid=uid,
                stages={"svm1": stage1_label, "svm2": stage2_label_text},
                final_label=final_label,
                final_probability=final_probability,
            )
            rows.append(row)
            paths.append(path)
            decisions.append(
                self._decision(
                    uid=uid,
                    row=row,
                    neutral_idx=neutral_idx,
                    stage1_label=stage1_label,
                    stage2_label=stage2_emotion,
                    final_label=final_label,
                    final_probability=final_probability,
                    path=path,
                )
            )
        proba = np.vstack(rows).astype(np.float64) if rows else np.zeros((0, len(self._classes)))
        return proba, tuple(decisions), tuple(paths)

    def _decision(
        self,
        *,
        uid: UID,
        row: FloatArray,
        neutral_idx: int,
        stage1_label: str,
        stage2_label: Emotion,
        final_label: Emotion,
        final_probability: float,
        path: SvmStagePath,
    ) -> TwoStageDecision:
        emotion_scores = {
            self._classes[idx]: float(row[idx])
            for idx in range(len(self._classes))
            if idx != neutral_idx
        }
        rationale = (
            "SVM1 chose "
            f"{path.stages['svm1']}; SVM2 chose {path.stages['svm2']}; "
            f"final={final_label.value}."
        )
        return TwoStageDecision(
            uid=uid,
            neutral_probability=float(row[neutral_idx]),
            non_neutral_probability=float(max(0.0, 1.0 - row[neutral_idx])),
            stage1_label=stage1_label,
            stage2_label=stage2_label,
            final_label=final_label,
            final_probability=final_probability,
            emotion_scores=emotion_scores,
            rationale=rationale,
        )


@real
class SvmFourStageClassifier:
    """Train the SVM1→SVM2→SVM3→SVM4 hierarchy from the project note."""

    _DIRECT_STAGE2 = (Emotion.ANGER, Emotion.JOY, Emotion.SURPRISE)

    def __init__(
        self,
        stage_factory: Callable[[int], Estimator],
        classes: tuple[Emotion, ...],
        use_concepts: bool = True,
        neutral_label: Emotion = Emotion.NEUTRAL,
    ) -> None:
        if neutral_label not in classes:
            raise ValueError(f"neutral_label {neutral_label.value!r} is not in classifier classes")
        self._classes = classes
        self._use_concepts = use_concepts
        self._neutral_label = neutral_label
        self._stage1 = _LocalStageClassifier(stage_factory, ("neutral", "non_neutral"))
        self._stage2 = _LocalStageClassifier(
            stage_factory,
            (*tuple(emotion.value for emotion in self._DIRECT_STAGE2), "else"),
        )
        self._stage3 = _LocalStageClassifier(stage_factory, (Emotion.SADNESS.value, "else"))
        self._stage4 = _LocalStageClassifier(
            stage_factory, (Emotion.DISGUST.value, Emotion.FEAR.value)
        )
        self._last_decisions: tuple[TwoStageDecision, ...] = ()
        self._last_paths: tuple[SvmStagePath, ...] = ()

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    @property
    def last_two_stage_decisions(self) -> tuple[TwoStageDecision, ...]:
        return self._last_decisions

    @property
    def last_svm_stage_paths(self) -> tuple[SvmStagePath, ...]:
        return self._last_paths

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        design = self._design(bundle)
        labels = tuple(self._classes[int(idx)] for idx in y)
        stage1_y = tuple(
            "neutral" if label == self._neutral_label else "non_neutral" for label in labels
        )
        stage2_mask = np.asarray(
            [label != self._neutral_label for label in labels], dtype=np.bool_
        )
        stage2_y = tuple(self._stage2_label(label) for label in labels if label != self._neutral_label)
        stage3_mask = np.asarray([self._goes_to_stage3(label) for label in labels], dtype=np.bool_)
        stage3_y = tuple(self._stage3_label(label) for label in labels if self._goes_to_stage3(label))
        stage4_mask = np.asarray([self._goes_to_stage4(label) for label in labels], dtype=np.bool_)
        stage4_y = tuple(label.value for label in labels if self._goes_to_stage4(label))

        self._stage1.fit(design, stage1_y)
        self._stage2.fit(design[stage2_mask], stage2_y)
        self._stage3.fit(design[stage3_mask], stage3_y)
        self._stage4.fit(design[stage4_mask], stage4_y)
        self._last_decisions = ()
        self._last_paths = ()
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        return self._route(bundle)[0]

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba, decisions, paths = self._route(bundle)
        index = {emotion: idx for idx, emotion in enumerate(self._classes)}
        y_pred = np.asarray([index[item.final_label] for item in decisions], dtype=np.int64)
        self._last_decisions = decisions
        self._last_paths = paths
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def stage_outputs(
        self, bundle: FeatureBundle, proba: FloatArray | None = None
    ) -> tuple[TwoStageDecision, ...]:
        if proba is not None:
            # SVM hard-routing is part of the explanation, so recompute from stages.
            pass
        _, decisions, paths = self._route(bundle)
        self._last_decisions = decisions
        self._last_paths = paths
        return decisions

    def _design(self, bundle: FeatureBundle) -> FloatArray:
        if self._use_concepts:
            return bundle.stack().values
        return bundle.stack(kind=FeatureKind.EMBEDDING).values

    def _route(
        self, bundle: FeatureBundle
    ) -> tuple[FloatArray, tuple[TwoStageDecision, ...], tuple[SvmStagePath, ...]]:
        design = self._design(bundle)
        stage1 = self._stage1.predict_proba(design)
        stage2 = self._stage2.predict_proba(design)
        stage3 = self._stage3.predict_proba(design)
        stage4 = self._stage4.predict_proba(design)
        neutral_idx = self._classes.index(self._neutral_label)
        emotion_to_index = {emotion: idx for idx, emotion in enumerate(self._classes)}
        rows: list[FloatArray] = []
        decisions: list[TwoStageDecision] = []
        paths: list[SvmStagePath] = []

        for row_idx, uid in enumerate(bundle.uids):
            row = np.zeros(len(self._classes), dtype=np.float64)
            neutral_probability = float(stage1[row_idx, 0])
            non_neutral_probability = float(stage1[row_idx, 1])
            row[neutral_idx] = neutral_probability

            for local_idx, emotion in enumerate(self._DIRECT_STAGE2):
                row[emotion_to_index[emotion]] = non_neutral_probability * float(
                    stage2[row_idx, local_idx]
                )
            stage2_else = non_neutral_probability * float(stage2[row_idx, 3])
            row[emotion_to_index[Emotion.SADNESS]] = stage2_else * float(stage3[row_idx, 0])
            stage3_else = stage2_else * float(stage3[row_idx, 1])
            row[emotion_to_index[Emotion.DISGUST]] = stage3_else * float(stage4[row_idx, 0])
            row[emotion_to_index[Emotion.FEAR]] = stage3_else * float(stage4[row_idx, 1])
            row = _normalize_vector(row)

            stage1_label = self._stage1.labels[int(np.argmax(stage1[row_idx]))]
            stage2_label = self._stage2.labels[int(np.argmax(stage2[row_idx]))]
            stage3_label = self._stage3.labels[int(np.argmax(stage3[row_idx]))]
            stage4_label = self._stage4.labels[int(np.argmax(stage4[row_idx]))]
            final_label = self._final_label(stage1_label, stage2_label, stage3_label, stage4_label)
            final_probability = float(row[emotion_to_index[final_label]])
            path = SvmStagePath(
                uid=uid,
                stages={
                    "svm1": stage1_label,
                    "svm2": stage2_label,
                    "svm3": stage3_label,
                    "svm4": stage4_label,
                },
                final_label=final_label,
                final_probability=final_probability,
            )
            rows.append(row)
            paths.append(path)
            decisions.append(
                self._decision(
                    uid=uid,
                    row=row,
                    neutral_idx=neutral_idx,
                    stage1_label=stage1_label,
                    final_label=final_label,
                    final_probability=final_probability,
                    path=path,
                )
            )
        proba = np.vstack(rows).astype(np.float64) if rows else np.zeros((0, len(self._classes)))
        return proba, tuple(decisions), tuple(paths)

    def _stage2_label(self, label: Emotion) -> str:
        if label in self._DIRECT_STAGE2:
            return label.value
        return "else"

    def _stage3_label(self, label: Emotion) -> str:
        if label == Emotion.SADNESS:
            return Emotion.SADNESS.value
        return "else"

    def _goes_to_stage3(self, label: Emotion) -> bool:
        return label not in (self._neutral_label, *self._DIRECT_STAGE2)

    def _goes_to_stage4(self, label: Emotion) -> bool:
        return label in (Emotion.DISGUST, Emotion.FEAR)

    def _final_label(
        self, stage1_label: str, stage2_label: str, stage3_label: str, stage4_label: str
    ) -> Emotion:
        if stage1_label == "neutral":
            return self._neutral_label
        if stage2_label != "else":
            return Emotion(stage2_label)
        if stage3_label == Emotion.SADNESS.value:
            return Emotion.SADNESS
        return Emotion(stage4_label)

    def _decision(
        self,
        *,
        uid: UID,
        row: FloatArray,
        neutral_idx: int,
        stage1_label: str,
        final_label: Emotion,
        final_probability: float,
        path: SvmStagePath,
    ) -> TwoStageDecision:
        emotion_scores = {
            self._classes[idx]: float(row[idx])
            for idx in range(len(self._classes))
            if idx != neutral_idx
        }
        best_non_neutral_idx = max(emotion_scores, key=lambda emotion: emotion_scores[emotion])
        rationale = (
            "SVM1 chose "
            f"{path.stages['svm1']}; SVM2 chose {path.stages['svm2']}; "
            f"SVM3 chose {path.stages['svm3']}; SVM4 chose {path.stages['svm4']}; "
            f"final={final_label.value}."
        )
        return TwoStageDecision(
            uid=uid,
            neutral_probability=float(row[neutral_idx]),
            non_neutral_probability=float(max(0.0, 1.0 - row[neutral_idx])),
            stage1_label=stage1_label,
            stage2_label=best_non_neutral_idx,
            final_label=final_label,
            final_probability=final_probability,
            emotion_scores=emotion_scores,
            rationale=rationale,
        )


def _normalize_rows(values: FloatArray, n_classes: int) -> FloatArray:
    proba = np.asarray(values, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != n_classes:
        raise ValueError(f"probability shape must be (n_samples, {n_classes}): {proba.shape}")
    row_sum = proba.sum(axis=1, keepdims=True)
    normalized = np.divide(proba, row_sum, out=np.zeros_like(proba), where=row_sum > 0.0)
    empty = row_sum[:, 0] <= 0.0
    if np.any(empty):
        normalized[empty] = 1.0 / n_classes
    return cast(FloatArray, normalized.astype(np.float64))


def _normalize_vector(values: FloatArray) -> FloatArray:
    row = np.asarray(values, dtype=np.float64)
    total = float(row.sum())
    if total <= 0.0:
        return cast(FloatArray, np.full(row.shape, 1.0 / row.size, dtype=np.float64))
    return cast(FloatArray, (row / total).astype(np.float64))


def _top_margin(scores: FloatArray) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    if finite.size < 2:
        return 0.0
    top_two = np.partition(finite, -2)[-2:]
    return float(top_two[1] - top_two[0])


def _non_neutral_distribution(row: FloatArray, neutral_idx: int) -> FloatArray:
    values = np.asarray(row, dtype=np.float64).copy()
    values[neutral_idx] = 0.0
    total = float(values.sum())
    if total <= 0.0:
        values[:] = 1.0 / max(1, values.size - 1)
        values[neutral_idx] = 0.0
        return values
    return values / total
