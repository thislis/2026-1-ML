"""Persist fitted non-PyTorch classifiers with their experiment wiring."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meld_emotion.config.loader import from_dict
from meld_emotion.config.schema import ExperimentConfig
from meld_emotion.core.protocols import Classifier

ARTIFACT_KIND = "meld_emotion.classifier_artifact.v1"


@dataclass(frozen=True)
class ClassifierArtifact:
    """Loaded classifier artifact and the config needed to rebuild features."""

    classifier: Classifier
    config: ExperimentConfig
    metadata: Mapping[str, str]
    path: Path


def save_classifier_artifact(
    path: str | Path,
    classifier: Classifier,
    config: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> None:
    """Save a fitted classifier plus the experiment config used to train it."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "kind": ARTIFACT_KIND,
        "classifier": classifier,
        "config": dict(config),
        "metadata": dict(metadata),
    }
    with artifact_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_classifier_artifact(path: str | Path) -> ClassifierArtifact:
    """Load a fitted classifier artifact saved by :func:`save_classifier_artifact`."""

    artifact_path = Path(path)
    with artifact_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping) or payload.get("kind") != ARTIFACT_KIND:
        raise ValueError(f"classifier artifact 형식이 아닙니다: {artifact_path}")
    classifier = payload.get("classifier")
    if not isinstance(classifier, Classifier):
        raise ValueError(f"classifier artifact 에 Classifier 가 없습니다: {artifact_path}")
    config_obj = payload.get("config")
    if not isinstance(config_obj, Mapping):
        raise ValueError(f"classifier artifact 에 config 매핑이 없습니다: {artifact_path}")
    metadata_obj = payload.get("metadata", {})
    if not isinstance(metadata_obj, Mapping):
        raise ValueError(f"classifier artifact metadata 형식이 올바르지 않습니다: {artifact_path}")
    return ClassifierArtifact(
        classifier=classifier,
        config=from_dict(config_obj),
        metadata={str(key): str(value) for key, value in metadata_obj.items()},
        path=artifact_path,
    )
