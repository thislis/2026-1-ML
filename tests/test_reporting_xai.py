"""Fine-grained XAI reporting payloads."""

from __future__ import annotations

import json

from meld_emotion.core.results import (
    DialogueXaiResult,
    EvaluationReport,
    ExperimentResult,
    ExplanationReport,
    MetricResult,
    ModalityXaiSummary,
    UnitAttribution,
    UtteranceAttribution,
)
from meld_emotion.core.types import Emotion, Modality
from meld_emotion.reporting.report import ConsoleReporter, DashboardExporter


def _result() -> ExperimentResult:
    xai = DialogueXaiResult(
        uid="d0/u1",
        dialogue_id=0,
        utterance_id=1,
        speaker="A",
        pred_class=Emotion.ANGER,
        pred_proba=0.8,
        target_class=Emotion.ANGER,
        target_logit=2.5,
        modality=(
            ModalityXaiSummary(Modality.TEXT, True, 0.6, 0.7, 1.2),
            ModalityXaiSummary(Modality.AUDIO, True, 0.3, 0.2, 0.4),
            ModalityXaiSummary(Modality.VIDEO, False, None, 0.0, None),
        ),
        utterances=(
            UtteranceAttribution("d0/u1", 0, 1, "A", 3.0, 0.75, 0.4),
            UtteranceAttribution("d0/u0", 0, 0, "B", 1.0, 0.25, 0.2),
        ),
        classifier_blocks={"fused": 0.4, "context": 0.2, "memory": 0.1},
        top_text_units=(UnitAttribution("d0/u1:angry", 0.5, 1),),
        top_audio_units=(UnitAttribution("d0/u1:0.10-0.20s", 0.3, 2, 0.1, 0.2),),
        top_video_units=(UnitAttribution("d0/u1:frame_3", 0.1, 3),),
        text_dimension_attribution=(UnitAttribution("dim_7", 0.8, 7),),
    )
    return ExperimentResult(
        name="xai",
        evaluation=EvaluationReport(
            scenario="full",
            metrics=(MetricResult("accuracy", 1.0),),
        ),
        explanation=ExplanationReport(dialogue_xai=(xai,)),
    )


def test_console_reporter_includes_finegrained_xai_summary() -> None:
    text = ConsoleReporter().format(_result())
    assert "[Fine-grained dialogue XAI]" in text
    assert "mod=text:0.70" in text
    assert "d0/u1:angry" in text


def test_dashboard_exporter_writes_finegrained_payload(tmp_path) -> None:
    path = tmp_path / "dashboard.json"
    DashboardExporter(str(path)).save(_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload["finegrained_xai"]["targets"][0]
    assert target["uid"] == "d0/u1"
    assert target["modality_panel"][0]["modality"] == "text"
    assert target["dimension_panel"]["text"][0]["label"] == "dim_7"
