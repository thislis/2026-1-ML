"""Single MP4+text inference helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from meld_emotion.cli import build_parser
from meld_emotion.core.data import AudioInput, RawSample, VideoInput
from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.results import (
    DialogueXaiResult,
    ModalityXaiSummary,
    PredictionSet,
    UnitAttribution,
    UtteranceAttribution,
)
from meld_emotion.core.types import (
    EMOTION_ORDER,
    Emotion,
    FeatureKind,
    FloatArray,
    IntArray,
    Modality,
)
from meld_emotion.inference import (
    InferenceResult,
    dashboard_to_json,
    format_inference_result,
    result_to_json,
    result_to_markdown,
    run_inference,
)
from meld_emotion.models.two_stage import TwoStageEmotionClassifier


class _FixedExtractor:
    kind = FeatureKind.EMBEDDING

    def __init__(self, modality: Modality, dim: int) -> None:
        self._modality = modality
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fixed_{self._modality.value}"

    @property
    def modality(self) -> Modality:
        return self._modality

    def fit(self, samples: Sequence[RawSample]) -> _FixedExtractor:
        return self

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        return FeatureMatrix(
            values=np.ones((len(samples), self._dim), dtype=np.float64),
            names=tuple(f"{self.name}_{i}" for i in range(self._dim)),
            modality=self._modality,
            kind=self.kind,
            source=self.name,
        )


class _FakeMediaLoader:
    def load_audio(self, audio: AudioInput) -> AudioInput:
        return replace(audio, waveform=np.ones(16000, dtype=np.float64), sample_rate=16000)

    def load_video(self, video: VideoInput) -> VideoInput:
        frames = np.ones((4, 224, 224, 3), dtype=np.float64)
        return replace(video, frames=frames, fps=4.0)


class _FakeClassifier:
    @property
    def classes(self) -> tuple[Emotion, ...]:
        return EMOTION_ORDER

    def fit(self, bundle: FeatureBundle, y: IntArray) -> _FakeClassifier:
        return self

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        return PredictionSet(
            uids=bundle.uids,
            y_pred=np.argmax(proba, axis=1).astype(np.int64),
            proba=proba,
            classes=self.classes,
        )

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        assert bundle.n_samples == 1
        return np.asarray([[0.05, 0.60, 0.10, 0.08, 0.07, 0.06, 0.04]], dtype=np.float64)


def _extractors() -> tuple[_FixedExtractor, ...]:
    return (
        _FixedExtractor(Modality.TEXT, 2),
        _FixedExtractor(Modality.AUDIO, 3),
        _FixedExtractor(Modality.VIDEO, 4),
    )


def test_run_inference_with_injected_components(tmp_path: Path) -> None:
    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"not a real mp4")

    result = run_inference(
        mp4,
        "I am so happy!",
        checkpoint_path=tmp_path / "fake.pt",
        device="cpu",
        top_k=3,
        extractors=_extractors(),
        media_loader=_FakeMediaLoader(),
        classifier=_FakeClassifier(),
    )

    assert result.label == Emotion.JOY
    assert result.probability == pytest.approx(0.60)
    assert [label for label, _ in result.top_k] == [
        Emotion.JOY,
        Emotion.SADNESS,
        Emotion.ANGER,
    ]
    assert result.to_dict()["label"] == "joy"
    assert '"label": "joy"' in result_to_json(result)
    assert result.two_stage is not None
    assert result.two_stage.stage1_label == "non_neutral"


def test_run_inference_uses_two_stage_final_prediction(tmp_path: Path) -> None:
    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"not a real mp4")
    classifier = TwoStageEmotionClassifier(_FakeClassifier(), neutral_threshold=0.98)

    result = run_inference(
        mp4,
        "I am so happy!",
        checkpoint_path=tmp_path / "fake.pt",
        device="cpu",
        top_k=3,
        extractors=_extractors(),
        media_loader=_FakeMediaLoader(),
        classifier=classifier,
    )

    assert result.label == Emotion.NEUTRAL
    assert result.two_stage is not None
    assert result.two_stage.stage1_label == "neutral"
    assert result.two_stage.stage2_label == Emotion.JOY
    two_stage = cast(dict[str, object], result.to_dict()["two_stage"])
    assert two_stage["final_label"] == "neutral"


def test_svm_inference_xai_includes_text_audio_video_units(tmp_path: Path) -> None:
    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"not a real mp4")

    result = run_inference(
        mp4,
        "I am so happy!",
        checkpoint_path=tmp_path / "fake.pt",
        device="cpu",
        top_k=3,
        extractors=_extractors(),
        media_loader=_FakeMediaLoader(),
        classifier=_FakeClassifier(),
        include_xai=True,
        xai_top_k=2,
        xai_audio_window_seconds=0.5,
        xai_video_window_seconds=0.5,
    )

    assert result.svm_xai is not None
    assert result.xai == ()
    assert [unit.label for unit in result.svm_xai.top_text_units] == ["I", "am"]
    first_text = result.svm_xai.top_text_units[0]
    assert first_text.char_start == 0
    assert first_text.char_end == 1
    assert result.svm_xai.top_audio_units[0].label == "0.00-0.50s"
    assert result.svm_xai.top_audio_units[0].start == pytest.approx(0.0)
    assert result.svm_xai.top_audio_units[0].end == pytest.approx(0.5)
    assert result.svm_xai.top_video_units[0].label == "0.00-0.50s"
    assert result.svm_xai.top_video_units[0].start == pytest.approx(0.0)
    assert result.svm_xai.top_video_units[0].end == pytest.approx(0.5)

    payload = result.to_dict()
    svm_xai = cast(dict[str, object], payload["svm_xai"])
    assert "top_text_units" in svm_xai
    assert "top_audio_units" in svm_xai
    assert "top_video_units" in svm_xai
    assert "Top text units" in result_to_markdown(result)


def test_svm_inference_xai_max_units_caps_scoring(tmp_path: Path) -> None:
    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"not a real mp4")

    result = run_inference(
        mp4,
        "I am so happy!",
        checkpoint_path=tmp_path / "fake.pt",
        device="cpu",
        extractors=_extractors(),
        media_loader=_FakeMediaLoader(),
        classifier=_FakeClassifier(),
        include_xai=True,
        xai_top_k=10,
        xai_audio_window_seconds=0.5,
        xai_video_window_seconds=0.5,
        xai_max_units_per_modality=1,
    )

    assert result.svm_xai is not None
    assert len(result.svm_xai.top_text_units) == 1
    assert len(result.svm_xai.top_audio_units) == 1
    assert len(result.svm_xai.top_video_units) == 1


def test_inference_result_can_include_xai() -> None:
    xai = DialogueXaiResult(
        uid="infer:sample.mp4",
        dialogue_id=0,
        utterance_id=0,
        speaker="unknown",
        pred_class=Emotion.JOY,
        pred_proba=0.6,
        target_class=Emotion.JOY,
        target_logit=1.7,
        modality=(
            ModalityXaiSummary(Modality.TEXT, True, 0.7, 0.8, 1.0),
            ModalityXaiSummary(Modality.AUDIO, True, 0.2, 0.1, 0.2),
            ModalityXaiSummary(Modality.VIDEO, True, 0.1, 0.1, 0.1),
        ),
        utterances=(UtteranceAttribution("infer:sample.mp4", 0, 0, "unknown", 1.0, 1.0, 1.0),),
        classifier_blocks={"fused": 0.3, "context": 0.1, "memory": 0.0},
        top_text_units=(UnitAttribution("infer:sample.mp4:happy", 0.9, 1),),
        top_audio_units=(UnitAttribution("infer:sample.mp4:0.00-0.10s", 0.2, 0, 0.0, 0.1),),
        top_video_units=(UnitAttribution("infer:sample.mp4:frame_0", 0.1, 0),),
    )
    result = InferenceResult(
        label=Emotion.JOY,
        probability=0.6,
        scores={Emotion.JOY: 0.6, Emotion.NEUTRAL: 0.4},
        top_k=((Emotion.JOY, 0.6),),
        checkpoint="best.pt",
        mp4_path="sample.mp4",
        xai=(xai,),
    )
    text = format_inference_result(result)
    assert "xai:" in text
    assert "mod=text:0.80" in text
    payload = result_to_json(result)
    assert '"xai"' in payload
    dashboard = dashboard_to_json(result)
    assert '"finegrained_xai"' in dashboard
    assert '"modality_panel"' in dashboard
    markdown = result_to_markdown(result)
    assert "# Emotion Inference Result" in markdown
    assert "## Fine-Grained XAI" in markdown
    assert "Top text units" in markdown


def test_run_inference_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MP4"):
        run_inference(
            tmp_path / "missing.mp4",
            "hello",
            device="cpu",
            extractors=_extractors(),
            media_loader=_FakeMediaLoader(),
            classifier=_FakeClassifier(),
        )

    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="top_k"):
        run_inference(mp4, "hello", top_k=0, classifier=_FakeClassifier())
    with pytest.raises(ValueError, match="device"):
        run_inference(mp4, "hello", device="bad", classifier=_FakeClassifier())
    with pytest.raises(ValueError, match="xai_audio_window_seconds"):
        run_inference(
            mp4,
            "hello",
            classifier=_FakeClassifier(),
            include_xai=True,
            xai_audio_window_seconds=0.0,
        )
    with pytest.raises(ValueError, match="xai_max_units_per_modality"):
        run_inference(
            mp4,
            "hello",
            classifier=_FakeClassifier(),
            include_xai=True,
            xai_max_units_per_modality=-1,
        )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_inference(
            mp4,
            "hello",
            checkpoint_path=tmp_path / "missing.pt",
            device="cpu",
            extractors=_extractors(),
            media_loader=_FakeMediaLoader(),
        )


def test_infer_help_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["infer", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "MP4" in help_text
    assert "--xai" in help_text
    assert "--xai-audio-window-seconds" in help_text
    assert "--xai-video-window-seconds" in help_text
    assert "--xai-max-units-per-modality" in help_text


def test_infer_svm_batch_help_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["infer-svm-batch", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--csv" in help_text
    assert "--mp4-dir" in help_text
    assert "--checkpoint" in help_text
    assert "--xai-audio-window-seconds" in help_text
