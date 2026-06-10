"""단일 MP4+텍스트 입력에 대한 감정 추론 유틸리티."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.protocols import Classifier, FeatureExtractor
from meld_emotion.core.results import DialogueXaiResult
from meld_emotion.core.types import Emotion, FloatArray, Split
from meld_emotion.data.media import MediaLoader as RawMediaLoader
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.features.audio import (
    Wav2Vec2XlsrAudioExtractor,
    Wav2Vec2XlsrAudioSequenceExtractor,
)
from meld_emotion.features.text import EmbeddingGemmaTextExtractor, TextTokenEmbeddingExtractor
from meld_emotion.features.video import TimeSformerVideoExtractor, VideoFrameEmbeddingExtractor
from meld_emotion.models.two_stage import TwoStageDecision
from meld_emotion.pipeline.cache import NullFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline, MediaLoader

DEFAULT_CHECKPOINT = "outputs/best_model.pt"
_ALLOWED_DEVICES = frozenset(("auto", "cpu", "mps", "cuda"))


@dataclass(frozen=True)
class InferenceResult:
    """단일 발화 추론 결과."""

    label: Emotion
    probability: float
    scores: Mapping[Emotion, float]
    top_k: tuple[tuple[Emotion, float], ...]
    checkpoint: str
    mp4_path: str
    two_stage: TwoStageDecision | None = None
    xai: tuple[DialogueXaiResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label.value,
            "probability": self.probability,
            "scores": {label.value: score for label, score in self.scores.items()},
            "top_k": [
                {"label": label.value, "probability": score} for label, score in self.top_k
            ],
            "checkpoint": self.checkpoint,
            "mp4_path": self.mp4_path,
            "two_stage": _jsonable(self.two_stage),
            "xai": _jsonable(self.xai),
        }


def run_inference(
    mp4_path: str | Path,
    text: str,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "auto",
    top_k: int = 7,
    extractors: Sequence[FeatureExtractor] | None = None,
    media_loader: MediaLoader | None = None,
    classifier: Classifier | None = None,
    include_xai: bool = False,
    xai_n_steps: int = 32,
    xai_top_k: int = 10,
) -> InferenceResult:
    """MP4 경로와 발화 텍스트로 감정 확률을 예측한다.

    ``extractors``/``media_loader``/``classifier`` 인자는 테스트와 특수 실험에서만 쓰는 주입
    지점이다. 일반 사용자는 기본값으로 학습 suite 와 같은 foundation feature 경로를 사용한다.
    """

    if top_k <= 0:
        raise ValueError("top_k 는 양수여야 합니다")
    resolved_device = resolve_device(device)
    mp4 = Path(mp4_path)
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 파일을 찾을 수 없습니다: {mp4}")
    checkpoint = Path(checkpoint_path)
    if classifier is None and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 를 찾을 수 없습니다: {checkpoint}")

    sample = _sample_from_inputs(mp4, text)
    chosen_extractors = (
        extractors
        if extractors is not None
        else default_xai_extractors(resolved_device)
        if include_xai
        else default_extractors(resolved_device)
    )
    chosen_media_loader = (
        media_loader
        if media_loader is not None
        else default_xai_media_loader()
        if include_xai
        else default_media_loader()
    )
    pipeline = FeaturePipeline(
        chosen_extractors,
        cache=NullFeatureCache(),
        media_loader=chosen_media_loader,
        media_error_policy="raise",
    )
    bundle = pipeline.fit_transform((sample,), Split.TEST)
    model = classifier if classifier is not None else _load_classifier(checkpoint, resolved_device)
    prediction = model.predict(bundle)
    probabilities = prediction.proba
    if probabilities.shape[0] != 1:
        raise ValueError(f"추론 결과 행 수가 1이 아닙니다: {probabilities.shape[0]}")
    classes = tuple(model.classes)
    scores = {
        emotion: float(score)
        for emotion, score in zip(classes, probabilities[0].tolist(), strict=True)
    }
    top = tuple(
        sorted(scores.items(), key=lambda item: item[1], reverse=True)[: min(top_k, len(scores))]
    )
    label = classes[int(prediction.y_pred[0])]
    probability = scores[label]
    two_stage = _stage_decision(model, bundle, probabilities)
    xai = _run_xai(model, bundle, xai_n_steps, xai_top_k) if include_xai else ()
    return InferenceResult(
        label=label,
        probability=probability,
        scores=scores,
        top_k=top,
        checkpoint=str(checkpoint),
        mp4_path=str(mp4),
        two_stage=two_stage,
        xai=xai,
    )


def resolve_device(device: str) -> str:
    """``auto`` 또는 명시 장치 문자열을 추론용 장치로 해석한다."""

    if device not in _ALLOWED_DEVICES:
        allowed = ", ".join(sorted(_ALLOWED_DEVICES))
        raise ValueError(f"device 는 {allowed} 중 하나여야 합니다: {device!r}")
    if device != "auto":
        return device

    try:
        torch: Any = import_module("torch")
    except ImportError:
        return "cpu"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        return "cuda"
    return "cpu"


def default_extractors(device: str) -> tuple[FeatureExtractor, ...]:
    """best_model.pt 학습 suite 와 동일한 foundation feature extractor 조합."""

    return (
        EmbeddingGemmaTextExtractor(
            model_name="google/embeddinggemma-300m",
            output_dim=768,
            batch_size=32,
            normalize=True,
            prompt_name="Classification",
            device=device,
        ),
        Wav2Vec2XlsrAudioExtractor(
            model_name="facebook/wav2vec2-xls-r-300m",
            output_dim=1024,
            batch_size=1,
            sampling_rate=16000,
            chunk_seconds=30.0,
            normalize=True,
            device=device,
        ),
        TimeSformerVideoExtractor(
            model_name="facebook/timesformer-base-finetuned-k400",
            output_dim=768,
            batch_size=2,
            num_frames=8,
            frame_size=224,
            normalize=True,
            pooling="cls",
            device=device,
        ),
    )


def default_xai_extractors(device: str) -> tuple[FeatureExtractor, ...]:
    """Fine-grained inference/XAI extractor 조합."""

    return (
        TextTokenEmbeddingExtractor(
            model_name="bert-base-uncased",
            max_tokens=64,
            output_dim=768,
            batch_size=16,
            normalize=True,
            device=device,
        ),
        Wav2Vec2XlsrAudioSequenceExtractor(
            model_name="facebook/wav2vec2-xls-r-300m",
            output_dim=1024,
            batch_size=1,
            sampling_rate=16000,
            max_steps=128,
            normalize=True,
            device=device,
        ),
        VideoFrameEmbeddingExtractor(
            model_name="openai/clip-vit-base-patch32",
            output_dim=768,
            batch_size=8,
            num_frames=16,
            frame_size=224,
            normalize=True,
            device=device,
        ),
    )


def default_media_loader() -> MediaLoader:
    return RawMediaLoader(
        audio_sample_rate=16000,
        video_max_frames=8,
        video_frame_size=(224, 224),
        max_audio_seconds=60.0,
        min_audio_seconds=0.025,
    )


def default_xai_media_loader() -> MediaLoader:
    return RawMediaLoader(
        audio_sample_rate=16000,
        video_max_frames=16,
        video_frame_size=(224, 224),
        max_audio_seconds=60.0,
        min_audio_seconds=0.025,
    )


def format_inference_result(result: InferenceResult) -> str:
    lines = [
        f"prediction: {result.label.value}",
        f"confidence: {result.probability:.6f}",
        f"checkpoint: {result.checkpoint}",
        f"mp4: {result.mp4_path}",
        "top_k:",
    ]
    lines.extend(f"  {label.value}: {score:.6f}" for label, score in result.top_k)
    if result.xai:
        lines.append("xai:")
        for item in result.xai:
            top_modality = max(item.modality, key=lambda modality: modality.attribution_share)
            top_utt = item.utterances[0] if item.utterances else None
            text = item.top_text_units[0].label if item.top_text_units else "-"
            audio = item.top_audio_units[0].label if item.top_audio_units else "-"
            video = item.top_video_units[0].label if item.top_video_units else "-"
            source = (
                f"utt={top_utt.utterance_id} share={top_utt.share:.2f}"
                if top_utt is not None
                else "utt=-"
            )
            lines.append(
                "  "
                f"{item.uid} target={item.target_class.value} "
                f"mod={top_modality.modality.value}:{top_modality.attribution_share:.2f} "
                f"{source}"
            )
            lines.append(f"    text: {text}")
            lines.append(f"    audio: {audio}")
            lines.append(f"    video: {video}")
    if result.two_stage is not None:
        stage2 = result.two_stage.stage2_label.value if result.two_stage.stage2_label else "-"
        lines.extend(
            [
                "two_stage:",
                f"  model1: {result.two_stage.stage1_label}",
                f"  p_neutral: {result.two_stage.neutral_probability:.6f}",
                f"  p_non_neutral: {result.two_stage.non_neutral_probability:.6f}",
                f"  model2: {stage2}",
                f"  rationale: {result.two_stage.rationale}",
            ]
        )
    return "\n".join(lines)


def result_to_json(result: InferenceResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def result_to_markdown(result: InferenceResult) -> str:
    """Human-readable Markdown explanation for a single inference result."""

    lines = [
        "# Emotion Inference Result",
        "",
        f"- Prediction: `{result.label.value}`",
        f"- Confidence: `{result.probability:.6f}`",
        f"- Checkpoint: `{result.checkpoint}`",
        f"- MP4: `{result.mp4_path}`",
        "",
        "## Top Classes",
        "",
    ]
    lines.extend(f"- `{label.value}`: `{score:.6f}`" for label, score in result.top_k)
    if result.two_stage is not None:
        stage2 = result.two_stage.stage2_label.value if result.two_stage.stage2_label else "n/a"
        lines.extend(
            [
                "",
                "## Two-Stage Decision",
                "",
                f"- Model 1: `{result.two_stage.stage1_label}`",
                f"- P(neutral): `{result.two_stage.neutral_probability:.6f}`",
                f"- P(non-neutral): `{result.two_stage.non_neutral_probability:.6f}`",
                f"- Model 2 top emotion: `{stage2}`",
                f"- Rationale: {result.two_stage.rationale}",
            ]
        )
    if result.xai:
        lines.extend(["", "## Fine-Grained XAI", ""])
        for item in result.xai:
            lines.extend(
                [
                    f"### {item.uid}",
                    "",
                    f"- Target class: `{item.target_class.value}`",
                    f"- Predicted class: `{item.pred_class.value}`",
                    f"- Predicted probability: `{item.pred_proba:.6f}`",
                    "",
                    "| Modality | Available | Gate | Attribution Share | Ablation Delta Logit |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for modality in item.modality:
                gate = "-" if modality.gate is None else f"{modality.gate:.6f}"
                delta = (
                    "-"
                    if modality.ablation_delta_logit is None
                    else f"{modality.ablation_delta_logit:.6f}"
                )
                lines.append(
                    f"| {modality.modality.value} | {modality.available} | {gate} | "
                    f"{modality.attribution_share:.6f} | {delta} |"
                )
            lines.extend(["", "Top text units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_text_units)
            lines.extend(["", "Top audio units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_audio_units)
            lines.extend(["", "Top video units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_video_units)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def inference_dashboard_payload(result: InferenceResult) -> dict[str, object]:
    """단일 inference 결과용 dashboard JSON payload."""

    return {
        "prediction": result.to_dict(),
        "finegrained_xai": {
            "targets": [
                {
                    "uid": item.uid,
                    "speaker": item.speaker,
                    "pred_class": item.pred_class.value,
                    "pred_proba": item.pred_proba,
                    "target_class": item.target_class.value,
                    "target_logit": item.target_logit,
                    "modality_panel": _jsonable(item.modality),
                    "dialogue_panel": _jsonable(item.utterances),
                    "block_panel": _jsonable(item.classifier_blocks),
                    "text_panel": _jsonable(item.top_text_units),
                    "audio_panel": _jsonable(item.top_audio_units),
                    "video_panel": _jsonable(item.top_video_units),
                    "dimension_panel": {
                        "text": _jsonable(item.text_dimension_attribution),
                        "audio": _jsonable(item.audio_dimension_attribution),
                        "video": _jsonable(item.video_dimension_attribution),
                    },
                }
                for item in result.xai
            ]
        },
    }


def dashboard_to_json(result: InferenceResult) -> str:
    return json.dumps(inference_dashboard_payload(result), ensure_ascii=False, indent=2)


def _sample_from_inputs(mp4: Path, text: str) -> RawSample:
    return RawSample(
        uid=f"infer:{mp4.name}",
        dialogue_id=0,
        utterance_id=0,
        text=text,
        speaker="unknown",
        split=Split.TEST,
        mask=ModalityMask.full(),
        audio=AudioInput(sample_rate=16000, source_path=mp4),
        video=VideoInput(fps=0.0, source_path=mp4),
        emotion=None,
        metadata={"source": "inference", "mp4_path": str(mp4)},
    )


def _load_classifier(checkpoint: Path, device: str) -> Classifier:
    from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier

    return TorchDialogueEmotionClassifier.from_checkpoint(checkpoint, device=device)


def _run_xai(
    model: Classifier,
    bundle: object,
    n_steps: int,
    top_k: int,
) -> tuple[DialogueXaiResult, ...]:
    from meld_emotion.core.features import FeatureBundle

    if not isinstance(bundle, FeatureBundle):
        raise TypeError("bundle must be FeatureBundle")
    explainer_model = getattr(model, "base", model)
    explainer = DialogueFineGrainedXaiExplainer(
        n_steps=n_steps,
        top_k=top_k,
        max_targets=1,
        target="predicted",
    )
    report = explainer.explain(explainer_model, bundle, _dummy_labels(bundle.n_samples))
    return report.dialogue_xai


def _stage_decision(
    model: Classifier,
    bundle: object,
    proba: FloatArray,
) -> TwoStageDecision | None:
    from meld_emotion.core.features import FeatureBundle

    if not isinstance(bundle, FeatureBundle):
        raise TypeError("bundle must be FeatureBundle")
    stage_outputs = getattr(model, "stage_outputs", None)
    if callable(stage_outputs):
        decisions = stage_outputs(bundle, proba=proba)
        return decisions[0] if decisions else None
    if Emotion.NEUTRAL not in model.classes:
        return None
    neutral_idx = model.classes.index(Emotion.NEUTRAL)
    neutral_probability = float(proba[0, neutral_idx])
    non_neutral_probability = float(max(0.0, 1.0 - neutral_probability))
    emotion_indices = [idx for idx, emotion in enumerate(model.classes) if emotion != Emotion.NEUTRAL]
    if not emotion_indices:
        return None
    emotion_mass = float(proba[0, emotion_indices].sum())
    best_idx = max(emotion_indices, key=lambda idx: float(proba[0, idx]))
    stage2_label = model.classes[best_idx]
    final_idx = int(np.argmax(proba[0]))
    final_label = model.classes[final_idx]
    stage1_label = "neutral" if final_label == Emotion.NEUTRAL else "non_neutral"
    emotion_scores = {
        model.classes[idx]: (
            float(proba[0, idx] / emotion_mass)
            if emotion_mass > 0.0
            else float(proba[0, idx])
        )
        for idx in emotion_indices
    }
    return TwoStageDecision(
        uid=bundle.uids[0],
        neutral_probability=neutral_probability,
        non_neutral_probability=non_neutral_probability,
        stage1_label=stage1_label,
        stage2_label=stage2_label,
        final_label=final_label,
        final_probability=float(proba[0, final_idx]),
        emotion_scores=emotion_scores,
        rationale=(
            f"Derived two-stage trace from seven-way probabilities: "
            f"P(non_neutral)={non_neutral_probability:.6f}, "
            f"top non-neutral emotion={stage2_label.value}."
        ),
    )


def _dummy_labels(n: int) -> Any:
    try:
        numpy: Any = import_module("numpy")
    except ImportError as exc:  # pragma: no cover - numpy is a base dependency
        raise RuntimeError("numpy is required for inference XAI") from exc
    return numpy.zeros(n, dtype=numpy.int64)


def _jsonable(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: _jsonable(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {
            (key.value if isinstance(key, Enum) else str(key)): _jsonable(value)
            for key, value in obj.items()
        }
    if isinstance(obj, tuple | list):
        return [_jsonable(value) for value in obj]
    return obj
