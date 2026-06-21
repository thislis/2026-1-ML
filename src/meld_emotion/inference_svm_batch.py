"""Batch SVM artifact inference with input-unit XAI for MELD splits."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.config.schema import ExtractorConfig
from meld_emotion.core.data import RawSample
from meld_emotion.core.protocols import FeatureExtractor
from meld_emotion.core.types import EMOTION_ORDER, Split
from meld_emotion.data.media import MediaLoader as RawMediaLoader
from meld_emotion.data.meld import MeldDatasetSource
from meld_emotion.inference import InferenceResult, resolve_device, run_inference
from meld_emotion.models.artifact import ClassifierArtifact, load_classifier_artifact

logger = logging.getLogger(__name__)

DEFAULT_SVM_BATCH_PREDICTIONS = "outputs/svm_test_xai_predictions.jsonl"
DEFAULT_SVM_BATCH_SUMMARY = "outputs/svm_test_xai_summary.json"


@dataclass(frozen=True)
class SvmBatchOutputPaths:
    """Output paths for SVM batch inference."""

    predictions: Path
    summary: Path


@dataclass(frozen=True)
class SvmBatchInferenceResult:
    """Completed SVM batch inference paths and summary."""

    paths: SvmBatchOutputPaths
    summary: Mapping[str, Any]


def run_svm_batch_inference(
    *,
    csv_path: str | Path,
    mp4_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "auto",
    predictions_path: str | Path = DEFAULT_SVM_BATCH_PREDICTIONS,
    summary_path: str | Path = DEFAULT_SVM_BATCH_SUMMARY,
    top_k: int = 7,
    xai_top_k: int = 10,
    xai_audio_window_seconds: float = 0.5,
    xai_video_window_seconds: float = 0.5,
    xai_max_units_per_modality: int = 0,
    resume: bool = False,
    limit: int | None = None,
) -> SvmBatchInferenceResult:
    """Run saved SVM artifact inference + unit-level XAI for all rows in a MELD CSV."""

    if top_k <= 0:
        raise ValueError("top_k 는 양수여야 합니다")
    if xai_top_k <= 0:
        raise ValueError("xai_top_k 는 양수여야 합니다")
    if limit is not None and limit <= 0:
        raise ValueError("limit 은 양수여야 합니다")

    paths = SvmBatchOutputPaths(predictions=Path(predictions_path), summary=Path(summary_path))
    for path in (paths.predictions, paths.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and paths.predictions.exists():
        paths.predictions.write_text("", encoding="utf-8")

    resolved_device = resolve_device(device)
    artifact = load_classifier_artifact(checkpoint_path)
    extractors = _build_artifact_extractors(artifact, resolved_device)
    media_loader = _build_artifact_media_loader(artifact)
    samples = _load_samples(csv_path, mp4_dir)
    if limit is not None:
        samples = samples[:limit]
    processed = _read_processed_uids(paths.predictions) if resume else set()
    logger.info(
        "SVM batch inference 시작: samples=%d processed=%d checkpoint=%s",
        len(samples),
        len(processed),
        checkpoint_path,
    )

    with paths.predictions.open("a", encoding="utf-8") as f:
        for index, sample in enumerate(samples, start=1):
            if sample.uid in processed:
                continue
            record = _record_stub(sample, checkpoint_path)
            try:
                mp4_path = _sample_mp4_path(sample)
                result = run_inference(
                    mp4_path=mp4_path,
                    text=sample.text,
                    checkpoint_path=checkpoint_path,
                    device=resolved_device,
                    top_k=top_k,
                    extractors=extractors,
                    media_loader=media_loader,
                    classifier=artifact.classifier,
                    include_xai=True,
                    xai_top_k=xai_top_k,
                    xai_audio_window_seconds=xai_audio_window_seconds,
                    xai_video_window_seconds=xai_video_window_seconds,
                    xai_max_units_per_modality=xai_max_units_per_modality,
                )
                record.update(_prediction_fields(result, sample))
                record["xai_error"] = None
            except Exception as exc:  # pragma: no cover - exercised by real media/model failures
                logger.exception("SVM batch inference 실패: uid=%s", sample.uid)
                record["prediction"] = None
                record["svm_xai"] = None
                record["xai_error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
            f.flush()
            if index == len(samples) or index % 25 == 0:
                logger.info("SVM batch inference 진행: %d/%d", index, len(samples))

    records = _load_prediction_records(paths.predictions)
    summary = _summary(records)
    paths.summary.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("SVM batch inference 완료: predictions=%s summary=%s", paths.predictions, paths.summary)
    return SvmBatchInferenceResult(paths=paths, summary=summary)


def _build_artifact_extractors(
    artifact: ClassifierArtifact,
    device: str,
) -> tuple[FeatureExtractor, ...]:
    from meld_emotion.pipeline.builder import build_extractor

    return tuple(build_extractor(_override_device(config, device)) for config in artifact.config.extractors)


def _override_device(config: ExtractorConfig, device: str) -> ExtractorConfig:
    if not is_dataclass(config) or isinstance(config, type):
        return config
    if any(field.name == "device" for field in fields(config)):
        return replace(config, device=device)  # type: ignore[call-arg]
    return config


def _build_artifact_media_loader(artifact: ClassifierArtifact) -> RawMediaLoader:
    media = artifact.config.media
    return RawMediaLoader(
        audio_sample_rate=media.audio_sample_rate,
        video_max_frames=media.video_max_frames,
        video_frame_size=media.video_frame_size,
        max_audio_seconds=media.max_audio_seconds,
        min_audio_seconds=media.min_audio_seconds,
    )


def _load_samples(csv_path: str | Path, mp4_dir: str | Path) -> list[RawSample]:
    csv = Path(csv_path)
    root = csv.parent
    media_subdir = _relative_or_absolute_subdir(root, Path(mp4_dir))
    source = MeldDatasetSource(
        root=str(root),
        csv_test=csv.name,
        audio_subdir_test=media_subdir,
        video_subdir_test=media_subdir,
    )
    return list(source.load(Split.TEST))


def _relative_or_absolute_subdir(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _sample_mp4_path(sample: RawSample) -> Path:
    if sample.video is None or sample.video.source_path is None:
        raise ValueError(f"sample 에 video source_path 가 없습니다: {sample.uid}")
    return sample.video.source_path


def _record_stub(sample: RawSample, checkpoint_path: str | Path) -> dict[str, object]:
    return {
        "uid": sample.uid,
        "dialogue_id": sample.dialogue_id,
        "utterance_id": sample.utterance_id,
        "speaker": sample.speaker,
        "text": sample.text,
        "gold": sample.emotion.value if sample.emotion is not None else None,
        "checkpoint": str(checkpoint_path),
        "mp4_path": str(_sample_mp4_path(sample)) if sample.video is not None else None,
    }


def _prediction_fields(result: InferenceResult, sample: RawSample) -> dict[str, object]:
    payload = result.to_dict()
    pred = str(payload["label"])
    gold = sample.emotion.value if sample.emotion is not None else None
    return {
        "prediction": payload,
        "pred": pred,
        "correct": pred == gold if gold is not None else None,
        "confidence": payload["probability"],
        "scores": payload["scores"],
        "top_k": payload["top_k"],
        "svm_xai": payload["svm_xai"],
    }


def _read_processed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record["uid"])
        for record in _load_prediction_records(path)
        if isinstance(record.get("uid"), str)
    }


def _load_prediction_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def _summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    completed = [record for record in records if record.get("pred") is not None]
    failed = [record for record in records if record.get("pred") is None]
    metrics = _metrics(completed)
    return {
        "n_records": len(records),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "metrics": metrics,
        "xai": {
            "n_with_svm_xai": sum(1 for record in completed if isinstance(record.get("svm_xai"), Mapping)),
            "top_text_units_available": sum(
                1 for record in completed if _has_units(record, "top_text_units")
            ),
            "top_audio_units_available": sum(
                1 for record in completed if _has_units(record, "top_audio_units")
            ),
            "top_video_units_available": sum(
                1 for record in completed if _has_units(record, "top_video_units")
            ),
        },
        "failures": [
            {"uid": record.get("uid"), "error": record.get("xai_error")}
            for record in failed[:20]
        ],
    }


def _has_units(record: Mapping[str, Any], key: str) -> bool:
    xai = record.get("svm_xai")
    if not isinstance(xai, Mapping):
        return False
    units = xai.get(key)
    return isinstance(units, list) and bool(units)


def _metrics(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    labels = tuple(emotion.value for emotion in EMOTION_ORDER)
    y_true = [str(record["gold"]) for record in records if record.get("gold") in labels]
    y_pred = [str(record["pred"]) for record in records if record.get("gold") in labels]
    total = len(y_true)
    accuracy = (
        sum(1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == pred) / total
        if total
        else 0.0
    )
    per_class = {label: _precision_recall_f1(label, y_true, y_pred) for label in labels}
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()])) if labels else 0.0
    weighted_f1 = (
        sum(per_class[label]["f1"] * per_class[label]["support"] for label in labels) / total
        if total
        else 0.0
    )
    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def _precision_recall_f1(
    label: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Mapping[str, float]:
    tp = sum(
        1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == label and pred == label
    )
    fp = sum(
        1 for gold, pred in zip(y_true, y_pred, strict=True) if gold != label and pred == label
    )
    fn = sum(
        1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == label and pred != label
    )
    support = sum(1 for gold in y_true if gold == label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(support),
    }


def _jsonable(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, tuple | list):
        return [_jsonable(value) for value in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: _jsonable(getattr(obj, field.name)) for field in fields(obj)}
    return obj
