"""Batch inference and XAI analysis for MELD test splits."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.types import EMOTION_ORDER, IntArray, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.media import MediaLoader as RawMediaLoader
from meld_emotion.data.meld import MeldDatasetSource
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.inference import (
    default_extractors,
    default_xai_extractors,
    resolve_device,
)
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier
from meld_emotion.pipeline.cache import NullFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = "outputs/test_batch_xai_predictions.jsonl"
DEFAULT_SUMMARY_PATH = "outputs/test_batch_xai_summary.json"
DEFAULT_REPORT_PATH = "outputs/dialogue_rnn_xai_analysis.md"
DEFAULT_SUITE_PATH = "outputs/all_model_w_all_features.json"

_CLASS_LABELS = tuple(emotion.value for emotion in EMOTION_ORDER)


@dataclass(frozen=True)
class BatchOutputPaths:
    """Output paths for batch inference artifacts."""

    predictions: Path
    summary: Path
    report: Path


@dataclass(frozen=True)
class BatchInferenceResult:
    """Paths and summary from a completed batch inference run."""

    paths: BatchOutputPaths
    summary: Mapping[str, Any]


def run_batch_inference(
    *,
    csv_path: str | Path,
    mp4_dir: str | Path,
    checkpoint_path: str | Path = "outputs/best_model.pt",
    device: str = "auto",
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    xai_steps: int = 32,
    xai_top_k: int = 10,
    resume: bool = False,
    limit: int | None = None,
) -> BatchInferenceResult:
    """Run pooled prediction plus fine-grained XAI for a MELD CSV split.

    The prediction track uses the same pooled feature extractors as ``best_model.pt``.
    The XAI track uses the existing sequence extractors so token/span/frame attributions
    can be produced. The two-track limitation is explicitly recorded in the report.
    """

    if xai_steps <= 0:
        raise ValueError("xai_steps 는 양수여야 합니다")
    if xai_top_k <= 0:
        raise ValueError("xai_top_k 는 양수여야 합니다")
    if limit is not None and limit <= 0:
        raise ValueError("limit 은 양수여야 합니다")

    paths = BatchOutputPaths(
        predictions=Path(predictions_path),
        summary=Path(summary_path),
        report=Path(report_path),
    )
    for path in (paths.predictions, paths.summary, paths.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and paths.predictions.exists():
        paths.predictions.write_text("", encoding="utf-8")

    resolved_device = resolve_device(device)
    samples = _load_test_samples(csv_path, mp4_dir)
    if limit is not None:
        samples = samples[:limit]
    logger.info("batch inference samples loaded: raw=%d", len(samples))

    classifier = TorchDialogueEmotionClassifier.from_checkpoint(checkpoint_path, resolved_device)
    pooled_bundle = _build_pooled_bundle(samples, resolved_device)
    prediction = classifier.predict(pooled_bundle)
    base_records = _base_prediction_records(samples, pooled_bundle, prediction, checkpoint_path)
    logger.info("pooled prediction complete: kept=%d", len(base_records))

    processed = _read_processed_uids(paths.predictions) if resume else set()
    if processed:
        logger.info("resume enabled: already processed=%d", len(processed))

    sample_by_uid = {sample.uid: sample for sample in samples}
    valid_samples = [sample_by_uid[uid] for uid in pooled_bundle.uids if uid in sample_by_uid]
    xai_pipeline = _build_xai_pipeline(valid_samples, resolved_device)
    explainer = DialogueFineGrainedXaiExplainer(
        n_steps=xai_steps,
        top_k=xai_top_k,
        max_targets=max(1, len(valid_samples)),
        target="predicted",
    )
    encoder = EmotionLabelEncoder()

    with paths.predictions.open("a", encoding="utf-8") as f:
        for dialogue_samples in _dialogue_sample_groups(valid_samples):
            if all(sample.uid in processed for sample in dialogue_samples):
                continue
            try:
                seq_bundle = xai_pipeline.transform(dialogue_samples, Split.TEST)
                y_true = _labels_for_bundle(dialogue_samples, seq_bundle, encoder)
                report = explainer.explain(classifier, seq_bundle, y_true)
                by_uid = {item.uid: item for item in report.dialogue_xai}
                for sample in dialogue_samples:
                    if sample.uid in processed:
                        continue
                    record = dict(base_records.get(sample.uid, _sample_stub_record(sample)))
                    record["xai"] = _jsonable(by_uid.get(sample.uid))
                    record["xai_error"] = None if sample.uid in by_uid else "xai_result_missing"
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            except Exception as exc:  # pragma: no cover - exercised with real media/model failures
                logger.exception(
                    "XAI failed for dialogue=%s", dialogue_samples[0].dialogue_id
                )
                for sample in dialogue_samples:
                    if sample.uid in processed:
                        continue
                    record = dict(base_records.get(sample.uid, _sample_stub_record(sample)))
                    record["xai"] = None
                    record["xai_error"] = f"{type(exc).__name__}: {exc}"
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    records, invalid_rows = _load_prediction_records_with_errors(paths.predictions)
    summary = dict(analyze_batch_records(records, suite_path=suite_path))
    summary["jsonl_warnings"] = {
        "invalid_rows_skipped": invalid_rows,
        "source": str(paths.predictions),
    }
    paths.summary.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.report.write_text(render_markdown_report(summary), encoding="utf-8")
    logger.info("batch inference artifacts saved: %s", paths)
    return BatchInferenceResult(paths=paths, summary=summary)


def load_prediction_records(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load JSONL prediction/XAI records."""

    records, _invalid_rows = _load_prediction_records_with_errors(path)
    return records


def _load_prediction_records_with_errors(
    path: str | Path,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    """Load valid JSONL objects and skip interrupted/corrupt rows.

    Long batch XAI runs are resumable and can be interrupted while a row is being
    written. Skipping malformed rows keeps the completed records usable; the caller
    records the skipped count in the summary artifact.
    """

    records: list[Mapping[str, Any]] = []
    invalid_rows = 0
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid_rows += 1
            logger.warning(
                "손상된 JSONL row 건너뜀: path=%s line=%d error=%s",
                path,
                line_no,
                exc,
            )
            continue
        if not isinstance(value, Mapping):
            invalid_rows += 1
            logger.warning(
                "JSONL row 가 object 가 아니어서 건너뜀: path=%s line=%d value=%r",
                path,
                line_no,
                value,
            )
            continue
        records.append(value)
    return tuple(records), invalid_rows


def analyze_batch_records(
    records: Sequence[Mapping[str, Any]],
    *,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
) -> Mapping[str, Any]:
    """Aggregate metrics, XAI patterns, and baseline comparison into one summary."""

    metric_summary = _metric_summary(records)
    xai_summary = _xai_pattern_summary(records)
    suite = _suite_comparison(suite_path)
    return {
        "n_records": len(records),
        "metrics": metric_summary,
        "xai_patterns": xai_summary,
        "suite_comparison": suite,
        "dialogue_rnn_structure": _dialogue_rnn_structure_summary(),
        "answers": _answers(metric_summary, xai_summary, suite),
        "limitations": [
            "Prediction uses pooled EmbeddingGemma/Wav2Vec2/TimeSformer features.",
            "Fine-grained XAI uses token/audio-step/frame sequence extractors with matching dimensions.",
            "Therefore XAI is a diagnostic approximation, not an exact attribution over the pooled "
            "training feature path.",
        ],
    }


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    """Render the integrated analysis report in Markdown."""

    metrics = _as_mapping(summary.get("metrics"))
    xai = _as_mapping(summary.get("xai_patterns"))
    suite = _as_mapping(summary.get("suite_comparison"))
    answers = _as_mapping(summary.get("answers"))
    structure = _as_mapping(summary.get("dialogue_rnn_structure"))

    lines = [
        "# Dialogue RNN Batch XAI Analysis",
        "",
        "## Executive Summary",
        f"- Analyzed records: {summary.get('n_records', 0)}",
        f"- Batch weighted F1: {_fmt(metrics.get('weighted_f1'))}",
        f"- Batch macro F1: {_fmt(metrics.get('macro_f1'))}",
        f"- Existing SVM weighted F1: {_fmt(suite.get('early_svm_weighted_f1'))}",
        f"- Existing dialogue_rnn weighted F1: {_fmt(suite.get('dialogue_rnn_weighted_f1'))}",
        "",
        "## XAI Pattern Summary",
    ]
    pattern_groups = _as_mapping(xai.get("groups"))
    for name in (
        "text_dominant_correct",
        "text_dominant_wrong",
        "audio_video_ignored",
        "context_helpful",
        "context_misleading",
        "low_confidence_ambiguous",
        "rare_class_unstable",
    ):
        group = _as_mapping(pattern_groups.get(name))
        lines.append(
            f"- `{name}`: support={group.get('support', 0)}, "
            f"avg_conf={_fmt(group.get('avg_confidence'))}, "
            f"avg_text_share={_fmt(group.get('avg_text_share'))}, "
            f"avg_audio_share={_fmt(group.get('avg_audio_share'))}, "
            f"avg_video_share={_fmt(group.get('avg_video_share'))}"
        )

    lines.extend(["", "## Representative XAI Cases"])
    for name, group_obj in pattern_groups.items():
        group = _as_mapping(group_obj)
        examples = group.get("examples")
        if not isinstance(examples, list) or not examples:
            continue
        lines.append(f"### {name}")
        for example_obj in examples[:3]:
            example = _as_mapping(example_obj)
            lines.append(
                "- "
                f"{example.get('uid')} gold={example.get('gold')} pred={example.get('pred')} "
                f"conf={_fmt(example.get('confidence'))} text={example.get('top_text', '-')}"
            )

    lines.extend(
        [
            "",
            "## Dialogue RNN Structure",
            f"- {structure.get('overview', '')}",
            f"- Strengths: {', '.join(str(x) for x in structure.get('strengths', []))}",
            f"- Weaknesses: {', '.join(str(x) for x in structure.get('weaknesses', []))}",
            "",
            "## Q1. dialogue_rnn 모델의 구조적 장단점",
            str(answers.get("q1", "")),
            "",
            "## Q2. dialogue_rnn 모델의 성능이 SVM 만도 못한 이유",
            str(answers.get("q2", "")),
            "",
            "## Q3. dialogue_rnn 모델의 성능을 높이기 위해 해야 하는 것",
            str(answers.get("q3", "")),
            "",
            "## Limitations",
        ]
    )
    for item in summary.get("limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _load_test_samples(csv_path: str | Path, mp4_dir: str | Path) -> list[RawSample]:
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


def _build_pooled_bundle(samples: Sequence[RawSample], device: str) -> FeatureBundle:
    pipeline = FeaturePipeline(
        default_extractors(device),
        cache=NullFeatureCache(),
        media_loader=RawMediaLoader(
            audio_sample_rate=16000,
            video_max_frames=8,
            video_frame_size=(224, 224),
            max_audio_seconds=60.0,
            min_audio_seconds=0.025,
        ),
        media_error_policy="drop_sample",
    )
    return pipeline.fit_transform(samples, Split.TEST)


def _build_xai_pipeline(samples: Sequence[RawSample], device: str) -> FeaturePipeline:
    pipeline = FeaturePipeline(
        default_xai_extractors(device),
        cache=NullFeatureCache(),
        media_loader=RawMediaLoader(
            audio_sample_rate=16000,
            video_max_frames=16,
            video_frame_size=(224, 224),
            max_audio_seconds=60.0,
            min_audio_seconds=0.025,
        ),
        media_error_policy="drop_sample",
    )
    pipeline.fit(samples)
    return pipeline


def _base_prediction_records(
    samples: Sequence[RawSample],
    bundle: FeatureBundle,
    prediction: PredictionSet,
    checkpoint_path: str | Path,
) -> dict[str, Mapping[str, Any]]:
    sample_by_uid = {sample.uid: sample for sample in samples}
    result: dict[str, Mapping[str, Any]] = {}
    for row, uid in enumerate(bundle.uids):
        sample = sample_by_uid[uid]
        scores = {
            emotion.value: float(prediction.proba[row, idx])
            for idx, emotion in enumerate(prediction.classes)
        }
        pred = prediction.classes[int(prediction.y_pred[row])]
        gold = sample.emotion
        result[uid] = {
            "uid": uid,
            "dialogue_id": sample.dialogue_id,
            "utterance_id": sample.utterance_id,
            "speaker": sample.speaker,
            "text": sample.text,
            "gold": gold.value if gold is not None else None,
            "pred": pred.value,
            "correct": gold == pred if gold is not None else None,
            "confidence": float(prediction.proba[row, int(prediction.y_pred[row])]),
            "scores": scores,
            "checkpoint": str(checkpoint_path),
            "mp4_path": (
                str(sample.video.source_path)
                if sample.video is not None and sample.video.source_path is not None
                else None
            ),
        }
    return result


def _sample_stub_record(sample: RawSample) -> Mapping[str, Any]:
    return {
        "uid": sample.uid,
        "dialogue_id": sample.dialogue_id,
        "utterance_id": sample.utterance_id,
        "speaker": sample.speaker,
        "text": sample.text,
        "gold": sample.emotion.value if sample.emotion is not None else None,
        "pred": None,
        "correct": None,
        "confidence": None,
    }


def _labels_for_bundle(
    samples: Sequence[RawSample],
    bundle: FeatureBundle,
    encoder: EmotionLabelEncoder,
) -> IntArray:
    sample_by_uid = {sample.uid: sample for sample in samples}
    labels = []
    for uid in bundle.uids:
        emotion = sample_by_uid[uid].emotion
        if emotion is None:
            raise ValueError(f"sample has no emotion label: {uid}")
        labels.append(emotion)
    return encoder.encode(labels)


def _dialogue_sample_groups(samples: Sequence[RawSample]) -> Iterable[tuple[RawSample, ...]]:
    groups: dict[int, list[RawSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.dialogue_id].append(sample)
    for dialogue_id in sorted(groups):
        yield tuple(sorted(groups[dialogue_id], key=lambda sample: sample.utterance_id))


def _read_processed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    for record in load_prediction_records(path):
        uid = record.get("uid")
        if isinstance(uid, str):
            processed.add(uid)
    return processed


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    labels = list(_CLASS_LABELS)
    y_true = [str(record["gold"]) for record in records if record.get("gold") in labels]
    y_pred = [str(record["pred"]) for record in records if record.get("gold") in labels]
    if len(y_true) != len(y_pred):
        raise ValueError("gold/pred lengths differ")
    confusion = {label: dict.fromkeys(labels, 0) for label in labels}
    for gold, pred in zip(y_true, y_pred, strict=True):
        if pred in labels:
            confusion[gold][pred] += 1
    total = len(y_true)
    accuracy = (
        sum(1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == pred) / total
        if total
        else 0.0
    )
    per_class = {
        label: _precision_recall_f1(label, y_true, y_pred)
        for label in labels
    }
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()])) if labels else 0.0
    weighted_f1 = (
        sum(per_class[label]["f1"] * per_class[label]["support"] for label in labels) / total
        if total
        else 0.0
    )
    per_class_recall = {
        label: per_class[label]["recall"]
        for label in labels
    }
    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def _precision_recall_f1(
    label: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Mapping[str, float]:
    tp = sum(1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(y_true, y_pred, strict=True) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(y_true, y_pred, strict=True) if gold == label and pred != label)
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


def _xai_pattern_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    enriched = [_record_xai_features(record) for record in records]
    with_xai = [item for item in enriched if item is not None]
    groups = {
        "text_dominant_correct": [
            item for item in with_xai if item["correct"] is True and item["text_share"] >= 0.6
        ],
        "text_dominant_wrong": [
            item for item in with_xai if item["correct"] is False and item["text_share"] >= 0.6
        ],
        "audio_video_ignored": [
            item for item in with_xai if item["audio_share"] + item["video_share"] <= 0.2
        ],
        "context_helpful": [
            item for item in with_xai if item["correct"] is True and item["context_memory"] >= 0.1
        ],
        "context_misleading": [
            item for item in with_xai if item["correct"] is False and item["context_memory"] >= 0.1
        ],
        "low_confidence_ambiguous": [
            item for item in with_xai if item["confidence"] < 0.45
        ],
        "rare_class_unstable": [
            item
            for item in with_xai
            if (item["gold"] in {"fear", "disgust"} or item["pred"] in {"fear", "disgust"})
            and item["correct"] is False
        ],
    }
    class_patterns = {
        label: _summarize_group([item for item in with_xai if item["gold"] == label])
        for label in _CLASS_LABELS
    }
    correctness_patterns = {
        "correct": _summarize_group([item for item in with_xai if item["correct"] is True]),
        "wrong": _summarize_group([item for item in with_xai if item["correct"] is False]),
    }
    return {
        "n_with_xai": len(with_xai),
        "n_missing_xai": len(records) - len(with_xai),
        "groups": {name: _summarize_group(items) for name, items in groups.items()},
        "by_gold_class": class_patterns,
        "by_correctness": correctness_patterns,
    }


def _record_xai_features(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    xai = record.get("xai")
    if not isinstance(xai, Mapping):
        return None
    modality = _as_list(xai.get("modality"))
    shares = _modality_values(modality, "attribution_share")
    gates = _modality_values(modality, "gate")
    blocks = _as_mapping(xai.get("classifier_blocks"))
    return {
        "uid": str(record.get("uid", "")),
        "text": str(record.get("text", "")),
        "gold": str(record.get("gold", "")),
        "pred": str(record.get("pred", "")),
        "correct": bool(record.get("correct")),
        "confidence": _float(record.get("confidence")),
        "text_share": shares.get("text", 0.0),
        "audio_share": shares.get("audio", 0.0),
        "video_share": shares.get("video", 0.0),
        "text_gate": gates.get("text", 0.0),
        "audio_gate": gates.get("audio", 0.0),
        "video_gate": gates.get("video", 0.0),
        "fused_delta": _float(blocks.get("fused")),
        "context_delta": _float(blocks.get("context")),
        "memory_delta": _float(blocks.get("memory")),
        "context_memory": max(_float(blocks.get("context")), _float(blocks.get("memory"))),
        "top_text": _top_unit_label(xai, "top_text_units"),
        "top_audio": _top_unit_label(xai, "top_audio_units"),
        "top_video": _top_unit_label(xai, "top_video_units"),
    }


def _summarize_group(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    examples = sorted(
        items,
        key=lambda item: (_float(item.get("confidence")), _float(item.get("text_share"))),
        reverse=True,
    )[:5]
    return {
        "support": len(items),
        "avg_confidence": _avg(items, "confidence"),
        "avg_text_share": _avg(items, "text_share"),
        "avg_audio_share": _avg(items, "audio_share"),
        "avg_video_share": _avg(items, "video_share"),
        "avg_text_gate": _avg(items, "text_gate"),
        "avg_audio_gate": _avg(items, "audio_gate"),
        "avg_video_gate": _avg(items, "video_gate"),
        "avg_fused_delta": _avg(items, "fused_delta"),
        "avg_context_delta": _avg(items, "context_delta"),
        "avg_memory_delta": _avg(items, "memory_delta"),
        "examples": [
            {
                "uid": item.get("uid"),
                "gold": item.get("gold"),
                "pred": item.get("pred"),
                "confidence": item.get("confidence"),
                "text": item.get("text"),
                "top_text": item.get("top_text"),
                "top_audio": item.get("top_audio"),
                "top_video": item.get("top_video"),
            }
            for item in examples
        ],
    }


def _avg(items: Sequence[Mapping[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return float(np.mean([_float(item.get(key)) for item in items]))


def _modality_values(modalities: Sequence[Any], key: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in modalities:
        if not isinstance(item, Mapping):
            continue
        modality = item.get("modality")
        if isinstance(modality, str):
            values[modality] = _float(item.get(key))
    return values


def _top_unit_label(xai: Mapping[str, Any], key: str) -> str:
    units = _as_list(xai.get(key))
    if not units:
        return "-"
    first = units[0]
    if not isinstance(first, Mapping):
        return "-"
    return str(first.get("label", "-"))


def _suite_comparison(path: str | Path) -> Mapping[str, Any]:
    suite_path = Path(path)
    if not suite_path.exists():
        return {"available": False}
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return {"available": False}
    values: dict[str, float | bool] = {"available": True}
    for outcome in _as_list(data.get("outcomes")):
        if not isinstance(outcome, Mapping):
            continue
        name = outcome.get("name")
        if name not in {"early_svm", "dialogue_rnn"}:
            continue
        result = _as_mapping(outcome.get("result"))
        evaluation = _as_mapping(result.get("evaluation"))
        metrics = _as_list(evaluation.get("metrics"))
        weighted = _metric_value(metrics, "weighted_f1")
        macro = _metric_value(metrics, "macro_f1")
        accuracy = _metric_value(metrics, "accuracy")
        values[f"{name}_weighted_f1"] = weighted
        values[f"{name}_macro_f1"] = macro
        values[f"{name}_accuracy"] = accuracy
        explanation = _as_mapping(result.get("explanation"))
        for item in _as_list(explanation.get("modality_contributions")):
            if not isinstance(item, Mapping):
                continue
            modality = item.get("modality")
            if isinstance(modality, str):
                values[f"{name}_{modality}_score_drop"] = _float(item.get("score_drop"))
    svm = values.get("early_svm_weighted_f1")
    rnn = values.get("dialogue_rnn_weighted_f1")
    if isinstance(svm, float) and isinstance(rnn, float):
        values["weighted_f1_gap_svm_minus_dialogue_rnn"] = svm - rnn
    return values


def _metric_value(metrics: Sequence[Any], name: str) -> float:
    for metric in metrics:
        if isinstance(metric, Mapping) and metric.get("name") == name:
            return _float(metric.get("value"))
    return 0.0


def _dialogue_rnn_structure_summary() -> Mapping[str, Any]:
    return {
        "overview": (
            "Text/audio/video features are encoded by modality-specific attentive GRU/LSTM "
            "encoders, fused by a missing-modality-aware gate, passed through a speaker-aware "
            "dialogue RNN and causal memory attention, then classified per utterance."
        ),
        "strengths": [
            "modality-specific encoders",
            "missing-modality-aware gated fusion",
            "speaker embeddings",
            "causal memory attention",
            "fine-grained XAI hooks",
        ],
        "weaknesses": [
            "pooled training features reduce each modality encoder to length-1 sequences",
            "many trainable layers on a small and imbalanced dataset",
            "unidirectional dialogue context cannot use future utterances",
            "audio/video gates can collapse or be ignored",
            "rare classes remain fragile",
        ],
    }


def _answers(
    metrics: Mapping[str, Any],
    xai: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> Mapping[str, str]:
    groups = _as_mapping(xai.get("groups"))
    ignored = _as_mapping(groups.get("audio_video_ignored"))
    text_wrong = _as_mapping(groups.get("text_dominant_wrong"))
    svm_gap = _fmt(suite.get("weighted_f1_gap_svm_minus_dialogue_rnn"))
    return {
        "q1": (
            "구조적으로는 멀티모달 인코딩, gate 기반 융합, speaker-aware dialogue context, "
            "causal memory attention 을 모두 갖춘 해석 가능한 모델이라는 장점이 있다. "
            "반면 현재 학습 경로의 pooled feature 에서는 발화 내부 sequence 정보가 거의 "
            "사라지고, 작은 MELD 학습셋에 비해 trainable block 이 많아 SVM보다 분산이 큰 "
            "모델이 된다."
        ),
        "q2": (
            "SVM은 강한 frozen foundation embedding 위에서 margin 기반 결정경계를 안정적으로 "
            "학습한다. dialogue_rnn은 추가한 RNN/fusion/context/memory 파라미터가 항상 이득으로 "
            "이어지지 않으며, 기존 suite 기준 SVM과의 weighted F1 격차는 "
            f"{svm_gap}이다. XAI에서 audio/video ignored 패턴 support={ignored.get('support', 0)}, "
            f"text-dominant wrong support={text_wrong.get('support', 0)}처럼 특정 모달리티 편향이 "
            "반복되면, 복잡한 멀티모달 구조를 갖췄지만 실제 성능은 텍스트 중심의 불안정한 "
            "분류기에 가까워졌다고 해석한다."
        ),
        "q3": (
            "우선 sequence extractor 로 학습한 checkpoint 를 별도로 만들어 XAI와 학습 feature "
            "공간을 일치시켜야 한다. 다음으로 text-only/text+audio/all-modality ablation suite, "
            "class-balanced sampling 또는 focal loss, validation 안정화, LR scheduler와 "
            "hyperparameter search를 수행한다. 그 뒤 bidirectional/Transformer dialogue encoder, "
            "residual text bypass, gate entropy regularization, SVM/LogReg ensemble을 검토한다."
        ),
        "batch_metric_note": (
            f"Batch run weighted_f1={_fmt(metrics.get('weighted_f1'))}, "
            f"macro_f1={_fmt(metrics.get('macro_f1'))}."
        ),
    }


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _float(value: object) -> float:
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    return 0.0


def _fmt(value: object) -> str:
    return f"{_float(value):.4f}"
