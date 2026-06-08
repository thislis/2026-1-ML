"""Batch inference/XAI analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meld_emotion.cli import build_parser
from meld_emotion.inference_batch import (
    DUPLICATE_UID_DROP_ALL,
    _load_test_samples,
    analyze_batch_records,
    load_prediction_records,
    load_prediction_records_with_manifest,
    render_markdown_report,
)


def _record(
    uid: str,
    *,
    gold: str,
    pred: str,
    confidence: float,
    text_share: float,
    audio_share: float,
    video_share: float,
    correct: bool,
) -> dict[str, object]:
    return {
        "uid": uid,
        "dialogue_id": 1,
        "utterance_id": int(uid.rsplit("_", maxsplit=1)[-1]),
        "speaker": "Rachel",
        "text": f"sample {uid}",
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "confidence": confidence,
        "scores": {"neutral": 0.4, pred: confidence},
        "xai": {
            "uid": uid,
            "pred_class": pred,
            "pred_proba": confidence,
            "target_class": pred,
            "modality": [
                {
                    "modality": "text",
                    "available": True,
                    "gate": 0.8,
                    "attribution_share": text_share,
                    "ablation_delta_logit": 1.0,
                },
                {
                    "modality": "audio",
                    "available": True,
                    "gate": 0.1,
                    "attribution_share": audio_share,
                    "ablation_delta_logit": 0.1,
                },
                {
                    "modality": "video",
                    "available": True,
                    "gate": 0.1,
                    "attribution_share": video_share,
                    "ablation_delta_logit": 0.0,
                },
            ],
            "classifier_blocks": {"fused": 0.3, "context": 0.2, "memory": 0.05},
            "top_text_units": [{"label": "happy", "score": 0.9, "index": 1}],
            "top_audio_units": [{"label": "0.0-0.1s", "score": 0.1, "index": 0}],
            "top_video_units": [{"label": "frame_0", "score": 0.1, "index": 0}],
        },
        "xai_error": None,
    }


def test_load_test_samples_maps_csv_rows_to_mp4_dir(tmp_path: Path) -> None:
    root = tmp_path / "MELD.Raw"
    mp4_dir = root / "output_repeated_splits_test"
    mp4_dir.mkdir(parents=True)
    csv = root / "test_sent_emo.csv"
    csv.write_text(
        "\n".join(
            [
                "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,"
                "Season,Episode,StartTime,EndTime",
                '1,"Hello!",Rachel,joy,positive,7,3,1,1,"00:00:00,000","00:00:01,000"',
            ]
        ),
        encoding="utf-8",
    )

    samples = _load_test_samples(csv, mp4_dir)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.uid == "test:7_3"
    assert sample.video is not None
    assert sample.video.source_path == mp4_dir / "dia7_utt3.mp4"
    assert sample.audio is not None
    assert sample.audio.source_path == mp4_dir / "dia7_utt3.mp4"


def test_prediction_jsonl_loading_and_xai_pattern_summary(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    records = [
        _record(
            "test:1_0",
            gold="joy",
            pred="joy",
            confidence=0.82,
            text_share=0.75,
            audio_share=0.10,
            video_share=0.10,
            correct=True,
        ),
        _record(
            "test:1_1",
            gold="fear",
            pred="neutral",
            confidence=0.42,
            text_share=0.80,
            audio_share=0.10,
            video_share=0.05,
            correct=False,
        ),
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    loaded = load_prediction_records(path)
    summary = analyze_batch_records(loaded, suite_path=tmp_path / "missing_suite.json")
    patterns = summary["xai_patterns"]["groups"]

    assert summary["metrics"]["accuracy"] == pytest.approx(0.5)
    assert patterns["text_dominant_correct"]["support"] == 1
    assert patterns["text_dominant_wrong"]["support"] == 1
    assert patterns["audio_video_ignored"]["support"] == 2
    assert patterns["context_helpful"]["support"] == 1
    assert patterns["context_misleading"]["support"] == 1
    assert patterns["low_confidence_ambiguous"]["support"] == 1
    assert patterns["rare_class_unstable"]["support"] == 1

    markdown = render_markdown_report(summary)
    assert "Q1. dialogue_rnn" in markdown
    assert "text_dominant_wrong" in markdown
    assert "Limitations" in markdown


def test_prediction_jsonl_loader_skips_interrupted_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    valid = _record(
        "test:1_0",
        gold="joy",
        pred="joy",
        confidence=0.82,
        text_share=0.75,
        audio_share=0.10,
        video_share=0.10,
        correct=True,
    )
    path.write_text(
        json.dumps(valid, ensure_ascii=False)
        + "\n"
        + '{"uid": "test:broken", "text": "unterminated',
        encoding="utf-8",
    )

    loaded = load_prediction_records(path)

    assert len(loaded) == 1
    assert loaded[0]["uid"] == "test:1_0"


def test_prediction_jsonl_unique_uids_creates_manifest_counts(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    records = [
        _record(
            "test:1_0",
            gold="joy",
            pred="joy",
            confidence=0.9,
            text_share=0.7,
            audio_share=0.1,
            video_share=0.1,
            correct=True,
        ),
        _record(
            "test:1_1",
            gold="sadness",
            pred="neutral",
            confidence=0.6,
            text_share=0.4,
            audio_share=0.3,
            video_share=0.2,
            correct=False,
        ),
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    loaded, manifest = load_prediction_records_with_manifest(path)

    assert len(loaded) == 2
    assert manifest.to_dict() == {
        "raw_lines": 2,
        "valid_json_lines": 2,
        "invalid_rows_skipped": 0,
        "duplicate_uid_count": 0,
        "duplicate_rows": [],
        "evaluated_records": 2,
        "evaluated_uids": ["test:1_0", "test:1_1"],
        "dropped_uid_policy": "fail_fast",
    }


def test_prediction_jsonl_duplicate_uid_default_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    first = _record(
        "test:1_0",
        gold="joy",
        pred="joy",
        confidence=0.9,
        text_share=0.7,
        audio_share=0.1,
        video_share=0.1,
        correct=True,
    )
    second = dict(first)
    second["pred"] = "neutral"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in (first, second)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate uid.*drop_all_rows_with_duplicated_uid"):
        load_prediction_records_with_manifest(path)


def test_prediction_jsonl_duplicate_policy_drops_all_duplicate_uid_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    duplicate = _record(
        "test:1_0",
        gold="joy",
        pred="joy",
        confidence=0.9,
        text_share=0.7,
        audio_share=0.1,
        video_share=0.1,
        correct=True,
    )
    keep = _record(
        "test:1_1",
        gold="sadness",
        pred="sadness",
        confidence=0.8,
        text_share=0.3,
        audio_share=0.4,
        video_share=0.2,
        correct=True,
    )
    path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False) for record in (duplicate, keep, dict(duplicate))
        ),
        encoding="utf-8",
    )

    loaded, manifest = load_prediction_records_with_manifest(
        path,
        duplicate_uid_policy=DUPLICATE_UID_DROP_ALL,
    )

    assert [record["uid"] for record in loaded] == ["test:1_1"]
    assert manifest.duplicate_uid_count == 1
    assert manifest.duplicate_rows == ({"uid": "test:1_0", "count": 2, "line_numbers": (1, 3)},)
    assert manifest.evaluated_records == 1
    assert manifest.evaluated_uids == ("test:1_1",)
    assert manifest.dropped_uid_policy == DUPLICATE_UID_DROP_ALL


def test_prediction_jsonl_invalid_rows_do_not_enter_manifest(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    valid = _record(
        "test:1_0",
        gold="joy",
        pred="joy",
        confidence=0.9,
        text_share=0.7,
        audio_share=0.1,
        video_share=0.1,
        correct=True,
    )
    missing_uid = dict(valid)
    del missing_uid["uid"]
    path.write_text(
        "\n".join(
            (
                json.dumps(valid, ensure_ascii=False),
                '{"uid": "test:broken", "text": "unterminated',
                json.dumps(missing_uid, ensure_ascii=False),
                "[]",
            )
        ),
        encoding="utf-8",
    )

    loaded, manifest = load_prediction_records_with_manifest(path)

    assert [record["uid"] for record in loaded] == ["test:1_0"]
    assert manifest.raw_lines == 4
    assert manifest.valid_json_lines == 3
    assert manifest.invalid_rows_skipped == 3
    assert manifest.evaluated_uids == ("test:1_0",)


def test_batch_summary_filters_to_manifest_evaluated_uids(tmp_path: Path) -> None:
    kept = _record(
        "test:1_0",
        gold="joy",
        pred="joy",
        confidence=0.9,
        text_share=0.7,
        audio_share=0.1,
        video_share=0.1,
        correct=True,
    )
    excluded = _record(
        "test:1_1",
        gold="joy",
        pred="neutral",
        confidence=0.9,
        text_share=0.7,
        audio_share=0.1,
        video_share=0.1,
        correct=False,
    )

    summary = analyze_batch_records(
        [kept, excluded],
        suite_path=tmp_path / "missing_suite.json",
        evaluated_uids=("test:1_0",),
    )

    assert summary["n_records"] == 1
    assert summary["metrics"]["n"] == 1
    assert summary["metrics"]["accuracy"] == pytest.approx(1.0)


def test_infer_batch_help_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["infer-batch", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--csv" in help_text
    assert "--mp4-dir" in help_text
    assert "--resume" in help_text
    assert "--duplicate-uid-policy" in help_text
