"""Batch inference/XAI analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meld_emotion.cli import build_parser
from meld_emotion.inference_batch import (
    _load_test_samples,
    analyze_batch_records,
    load_prediction_records,
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


def test_infer_batch_help_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["infer-batch", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--csv" in help_text
    assert "--mp4-dir" in help_text
    assert "--resume" in help_text
