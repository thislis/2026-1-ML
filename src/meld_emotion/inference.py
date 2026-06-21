"""단일 MP4+텍스트 입력에 대한 감정 추론 유틸리티."""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.protocols import Classifier, FeatureExtractor
from meld_emotion.core.results import DialogueXaiResult
from meld_emotion.core.types import Emotion, FeatureKind, FloatArray, Modality, Split
from meld_emotion.data.media import MediaLoader as RawMediaLoader
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.features.audio import (
    Wav2Vec2XlsrAudioExtractor,
    Wav2Vec2XlsrAudioSequenceExtractor,
)
from meld_emotion.features.text import EmbeddingGemmaTextExtractor, TextTokenEmbeddingExtractor
from meld_emotion.features.video import TimeSformerVideoExtractor, VideoFrameEmbeddingExtractor
from meld_emotion.fusion.masking import ModalityScenario, mask_bundle
from meld_emotion.models.artifact import ClassifierArtifact, load_classifier_artifact
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
    svm_xai: SvmXaiResult | None = None

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
            "svm_xai": _jsonable(self.svm_xai),
        }


@dataclass(frozen=True)
class SvmModalityAttribution:
    """Single-sample modality ablation attribution for SVM-style classifiers."""

    modality: Modality
    available: bool
    probability_delta: float
    ablated_probability: float
    ablated_label: Emotion


@dataclass(frozen=True)
class SvmFeatureAttribution:
    """Single feature perturbation attribution for SVM-style classifiers."""

    name: str
    modality: Modality
    kind: FeatureKind
    original_value: float
    probability_delta: float
    perturbed_probability: float


@dataclass(frozen=True)
class SvmUnitAttribution:
    """Input-unit perturbation attribution for SVM-style classifiers."""

    label: str
    modality: Modality
    index: int
    probability_delta: float
    ablated_probability: float
    ablated_label: Emotion
    start: float | None = None
    end: float | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True)
class SvmXaiResult:
    """Model-agnostic XAI payload for saved SVM artifacts."""

    uid: str
    pred_class: Emotion
    pred_proba: float
    modality: tuple[SvmModalityAttribution, ...]
    top_features: tuple[SvmFeatureAttribution, ...]
    top_text_units: tuple[SvmUnitAttribution, ...] = ()
    top_audio_units: tuple[SvmUnitAttribution, ...] = ()
    top_video_units: tuple[SvmUnitAttribution, ...] = ()


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
    xai_audio_window_seconds: float = 0.5,
    xai_video_window_seconds: float = 0.5,
    xai_max_units_per_modality: int = 0,
) -> InferenceResult:
    """MP4 경로와 발화 텍스트로 감정 확률을 예측한다.

    ``extractors``/``media_loader``/``classifier`` 인자는 테스트와 특수 실험에서만 쓰는 주입
    지점이다. 일반 사용자는 기본값으로 학습 suite 와 같은 foundation feature 경로를 사용한다.
    """

    if top_k <= 0:
        raise ValueError("top_k 는 양수여야 합니다")
    if xai_audio_window_seconds <= 0.0:
        raise ValueError("xai_audio_window_seconds 는 양수여야 합니다")
    if xai_video_window_seconds <= 0.0:
        raise ValueError("xai_video_window_seconds 는 양수여야 합니다")
    if xai_max_units_per_modality < 0:
        raise ValueError("xai_max_units_per_modality 는 0 이상이어야 합니다")
    resolved_device = resolve_device(device)
    mp4 = Path(mp4_path)
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 파일을 찾을 수 없습니다: {mp4}")
    checkpoint = Path(checkpoint_path)
    if classifier is None and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 를 찾을 수 없습니다: {checkpoint}")

    sample = _sample_from_inputs(mp4, text)
    artifact = None if classifier is not None else _maybe_load_classifier_artifact(checkpoint)
    chosen_extractors = _choose_extractors(
        extractors=extractors,
        artifact=artifact,
        include_xai=include_xai,
        device=resolved_device,
    )
    chosen_media_loader = _choose_media_loader(
        media_loader=media_loader,
        artifact=artifact,
        include_xai=include_xai,
    )
    pipeline = FeaturePipeline(
        chosen_extractors,
        cache=NullFeatureCache(),
        media_loader=chosen_media_loader,
        media_error_policy="raise",
    )
    bundle = pipeline.fit_transform((sample,), Split.TEST)
    model = (
        classifier
        if classifier is not None
        else artifact.classifier
        if artifact is not None
        else _load_classifier(checkpoint, resolved_device)
    )
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
    xai: tuple[DialogueXaiResult, ...] = ()
    svm_xai: SvmXaiResult | None = None
    if include_xai:
        if _supports_dialogue_xai(model):
            xai = _run_xai(model, bundle, xai_n_steps, xai_top_k)
        else:
            svm_xai = _run_svm_xai(
                model,
                bundle,
                probabilities,
                xai_top_k,
                pipeline,
                sample,
                chosen_media_loader,
                xai_audio_window_seconds=xai_audio_window_seconds,
                xai_video_window_seconds=xai_video_window_seconds,
                xai_max_units_per_modality=xai_max_units_per_modality,
            )
    return InferenceResult(
        label=label,
        probability=probability,
        scores=scores,
        top_k=top,
        checkpoint=str(checkpoint),
        mp4_path=str(mp4),
        two_stage=two_stage,
        xai=xai,
        svm_xai=svm_xai,
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


def _maybe_load_classifier_artifact(checkpoint: Path) -> ClassifierArtifact | None:
    try:
        return load_classifier_artifact(checkpoint)
    except (
        pickle.UnpicklingError,
        ValueError,
        OSError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
    ):
        return None


def _choose_extractors(
    *,
    extractors: Sequence[FeatureExtractor] | None,
    artifact: ClassifierArtifact | None,
    include_xai: bool,
    device: str,
) -> Sequence[FeatureExtractor]:
    if extractors is not None:
        return extractors
    if artifact is not None:
        from meld_emotion.pipeline.builder import build_extractor

        return tuple(
            build_extractor(_override_device(config, device))
            for config in artifact.config.extractors
        )
    return default_xai_extractors(device) if include_xai else default_extractors(device)


def _override_device(config: Any, device: str) -> Any:
    if not is_dataclass(config) or isinstance(config, type):
        return config
    if any(field.name == "device" for field in fields(config)):
        return replace(config, device=device)
    return config


def _choose_media_loader(
    *,
    media_loader: MediaLoader | None,
    artifact: ClassifierArtifact | None,
    include_xai: bool,
) -> MediaLoader:
    if media_loader is not None:
        return media_loader
    if artifact is not None:
        media = artifact.config.media
        return RawMediaLoader(
            audio_sample_rate=media.audio_sample_rate,
            video_max_frames=media.video_max_frames,
            video_frame_size=media.video_frame_size,
            max_audio_seconds=media.max_audio_seconds,
            min_audio_seconds=media.min_audio_seconds,
        )
    return default_xai_media_loader() if include_xai else default_media_loader()


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
            top_dialogue_modality = max(
                item.modality, key=lambda summary: summary.attribution_share
            )
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
                f"mod={top_dialogue_modality.modality.value}:"
                f"{top_dialogue_modality.attribution_share:.2f} "
                f"{source}"
            )
            lines.append(f"    text: {text}")
            lines.append(f"    audio: {audio}")
            lines.append(f"    video: {video}")
    if result.svm_xai is not None:
        lines.append("svm_xai:")
        top_svm_modality = max(
            result.svm_xai.modality,
            key=lambda item: abs(item.probability_delta),
            default=None,
        )
        if top_svm_modality is not None:
            lines.append(
                "  "
                f"modality {top_svm_modality.modality.value}: "
                f"delta={top_svm_modality.probability_delta:.6f} "
                f"ablated_label={top_svm_modality.ablated_label.value}"
            )
        for feature in result.svm_xai.top_features[:3]:
            lines.append(
                "  "
                f"feature {feature.name}: "
                f"delta={feature.probability_delta:.6f} "
                f"value={feature.original_value:.6f}"
            )
        if result.svm_xai.top_text_units:
            lines.append("  text_units:")
            lines.extend(
                f"    {unit.label}: delta={unit.probability_delta:.6f}"
                for unit in result.svm_xai.top_text_units[:3]
            )
        if result.svm_xai.top_audio_units:
            lines.append("  audio_units:")
            lines.extend(
                f"    {unit.label}: delta={unit.probability_delta:.6f}"
                for unit in result.svm_xai.top_audio_units[:3]
            )
        if result.svm_xai.top_video_units:
            lines.append("  video_units:")
            lines.extend(
                f"    {unit.label}: delta={unit.probability_delta:.6f}"
                for unit in result.svm_xai.top_video_units[:3]
            )
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
            for dialogue_modality in item.modality:
                gate = (
                    "-"
                    if dialogue_modality.gate is None
                    else f"{dialogue_modality.gate:.6f}"
                )
                delta = (
                    "-"
                    if dialogue_modality.ablation_delta_logit is None
                    else f"{dialogue_modality.ablation_delta_logit:.6f}"
                )
                lines.append(
                    f"| {dialogue_modality.modality.value} | "
                    f"{dialogue_modality.available} | {gate} | "
                    f"{dialogue_modality.attribution_share:.6f} | {delta} |"
                )
            lines.extend(["", "Top text units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_text_units)
            lines.extend(["", "Top audio units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_audio_units)
            lines.extend(["", "Top video units:"])
            lines.extend(f"- {unit.label}: `{unit.score:.6f}`" for unit in item.top_video_units)
            lines.append("")
    if result.svm_xai is not None:
        lines.extend(
            [
                "",
                "## SVM XAI",
                "",
                "| Modality | Available | Probability Delta | Ablated Probability | Ablated Label |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for svm_modality in result.svm_xai.modality:
            lines.append(
                f"| {svm_modality.modality.value} | {svm_modality.available} | "
                f"{svm_modality.probability_delta:.6f} | "
                f"{svm_modality.ablated_probability:.6f} | "
                f"{svm_modality.ablated_label.value} |"
            )
        lines.extend(["", "Top perturbed features:"])
        for feature in result.svm_xai.top_features:
            lines.append(
                f"- `{feature.name}` ({feature.modality.value}/{feature.kind.value}): "
                f"delta `{feature.probability_delta:.6f}`, value `{feature.original_value:.6f}`"
            )
        lines.extend(["", "Top text units:"])
        lines.extend(_svm_unit_markdown(unit) for unit in result.svm_xai.top_text_units)
        lines.extend(["", "Top audio units:"])
        lines.extend(_svm_unit_markdown(unit) for unit in result.svm_xai.top_audio_units)
        lines.extend(["", "Top video units:"])
        lines.extend(_svm_unit_markdown(unit) for unit in result.svm_xai.top_video_units)
    return "\n".join(lines).rstrip() + "\n"


def _svm_unit_markdown(unit: SvmUnitAttribution) -> str:
    location = ""
    if unit.start is not None and unit.end is not None:
        location = f" `{unit.start:.2f}-{unit.end:.2f}s`"
    elif unit.char_start is not None and unit.char_end is not None:
        location = f" chars `{unit.char_start}:{unit.char_end}`"
    return (
        f"- `{unit.label}`{location}: delta `{unit.probability_delta:.6f}`, "
        f"ablated `{unit.ablated_probability:.6f}` -> `{unit.ablated_label.value}`"
    )


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
        "svm_xai": _jsonable(result.svm_xai),
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


def _supports_dialogue_xai(model: Classifier) -> bool:
    explainer_model = getattr(model, "base", model)
    return callable(getattr(explainer_model, "xai_arrays", None)) and callable(
        getattr(explainer_model, "xai_model", None)
    )


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


def _run_svm_xai(
    model: Classifier,
    bundle: FeatureBundle,
    proba: FloatArray,
    top_k: int,
    pipeline: FeaturePipeline,
    sample: RawSample,
    media_loader: MediaLoader | None,
    *,
    xai_audio_window_seconds: float,
    xai_video_window_seconds: float,
    xai_max_units_per_modality: int,
) -> SvmXaiResult:
    if bundle.n_samples != 1:
        raise ValueError("SVM inference XAI 는 단일 샘플 bundle 만 지원합니다")
    pred_idx = int(np.argmax(proba[0]))
    pred_class = model.classes[pred_idx]
    pred_proba = float(proba[0, pred_idx])
    xai_sample = _load_unit_xai_media(sample, media_loader, bundle.modalities)
    return SvmXaiResult(
        uid=bundle.uids[0],
        pred_class=pred_class,
        pred_proba=pred_proba,
        modality=_svm_modality_attributions(model, bundle, pred_idx, pred_proba),
        top_features=_svm_feature_attributions(model, bundle, pred_idx, pred_proba, top_k),
        top_text_units=_svm_text_unit_attributions(
            model,
            pipeline,
            xai_sample,
            pred_idx,
            pred_proba,
            top_k,
            xai_max_units_per_modality,
        ),
        top_audio_units=_svm_audio_unit_attributions(
            model,
            pipeline,
            xai_sample,
            pred_idx,
            pred_proba,
            top_k,
            xai_audio_window_seconds,
            xai_max_units_per_modality,
        ),
        top_video_units=_svm_video_unit_attributions(
            model,
            pipeline,
            xai_sample,
            pred_idx,
            pred_proba,
            top_k,
            xai_video_window_seconds,
            xai_max_units_per_modality,
        ),
    )


def _load_unit_xai_media(
    sample: RawSample,
    media_loader: MediaLoader | None,
    modalities: Sequence[Modality],
) -> RawSample:
    if media_loader is None:
        return sample
    current = sample
    needs = set(modalities)
    if Modality.AUDIO in needs and current.audio is not None and current.audio.waveform is None:
        current = replace(current, audio=media_loader.load_audio(current.audio))
    if Modality.VIDEO in needs and current.video is not None and current.video.frames is None:
        current = replace(current, video=media_loader.load_video(current.video))
    return current


def _svm_modality_attributions(
    model: Classifier,
    bundle: FeatureBundle,
    pred_idx: int,
    pred_proba: float,
) -> tuple[SvmModalityAttribution, ...]:
    present = set(bundle.modalities)
    values: list[SvmModalityAttribution] = []
    for modality in bundle.modalities:
        scenario = ModalityScenario(
            name=f"no_{modality.value}",
            available=frozenset(present - {modality}),
        )
        ablated = mask_bundle(bundle, scenario)
        ablated_proba = model.predict_proba(ablated)
        ablated_idx = int(np.argmax(ablated_proba[0]))
        available = bool(bundle.availability.get(modality, np.ones(1, dtype=np.bool_))[0])
        values.append(
            SvmModalityAttribution(
                modality=modality,
                available=available,
                probability_delta=pred_proba - float(ablated_proba[0, pred_idx]),
                ablated_probability=float(ablated_proba[0, pred_idx]),
                ablated_label=model.classes[ablated_idx],
            )
        )
    return tuple(values)


def _svm_feature_attributions(
    model: Classifier,
    bundle: FeatureBundle,
    pred_idx: int,
    pred_proba: float,
    top_k: int,
) -> tuple[SvmFeatureAttribution, ...]:
    if top_k <= 0:
        return ()
    scored: list[SvmFeatureAttribution] = []
    for matrix_idx, matrix in enumerate(bundle.matrices):
        for col, name in enumerate(matrix.names):
            original = float(matrix.values[0, col])
            perturbed = _with_feature_value(bundle, matrix_idx, col, 0.0)
            perturbed_proba = model.predict_proba(perturbed)
            after = float(perturbed_proba[0, pred_idx])
            scored.append(
                SvmFeatureAttribution(
                    name=name,
                    modality=matrix.modality,
                    kind=matrix.kind,
                    original_value=original,
                    probability_delta=pred_proba - after,
                    perturbed_probability=after,
                )
            )
    scored.sort(key=lambda item: abs(item.probability_delta), reverse=True)
    return tuple(scored[:top_k])


_WORD_RE = re.compile(r"\b\w+(?:['-]\w+)?\b", re.UNICODE)


def _svm_text_unit_attributions(
    model: Classifier,
    pipeline: FeaturePipeline,
    sample: RawSample,
    pred_idx: int,
    pred_proba: float,
    top_k: int,
    max_units: int,
) -> tuple[SvmUnitAttribution, ...]:
    spans = [
        (match.group(0), match.start(), match.end())
        for match in _WORD_RE.finditer(sample.text)
    ]
    if max_units > 0:
        spans = spans[:max_units]
    scored: list[SvmUnitAttribution] = []
    for index, (word, start, end) in enumerate(spans):
        ablated_text = f"{sample.text[:start]}{' ' * (end - start)}{sample.text[end:]}"
        ablated_sample = replace(sample, text=ablated_text)
        ablated_proba, ablated_label = _predict_perturbed(
            model, pipeline, ablated_sample, pred_idx
        )
        scored.append(
            SvmUnitAttribution(
                label=word,
                modality=Modality.TEXT,
                index=index,
                probability_delta=pred_proba - ablated_proba,
                ablated_probability=ablated_proba,
                ablated_label=ablated_label,
                char_start=start,
                char_end=end,
            )
        )
    return _top_units(scored, top_k)


def _svm_audio_unit_attributions(
    model: Classifier,
    pipeline: FeaturePipeline,
    sample: RawSample,
    pred_idx: int,
    pred_proba: float,
    top_k: int,
    window_seconds: float,
    max_units: int,
) -> tuple[SvmUnitAttribution, ...]:
    if sample.audio is None or sample.audio.waveform is None:
        return ()
    waveform = np.asarray(sample.audio.waveform, dtype=np.float64).reshape(-1)
    sample_rate = int(sample.audio.sample_rate)
    if waveform.size == 0 or sample_rate <= 0:
        return ()
    windows = _time_windows(waveform.size / sample_rate, window_seconds)
    if max_units > 0:
        windows = windows[:max_units]
    scored: list[SvmUnitAttribution] = []
    for index, (start, end) in enumerate(windows):
        start_sample = max(0, min(waveform.size, int(np.floor(start * sample_rate))))
        end_sample = max(start_sample + 1, min(waveform.size, int(np.ceil(end * sample_rate))))
        masked = waveform.copy()
        masked[start_sample:end_sample] = 0.0
        ablated_audio = replace(sample.audio, waveform=masked)
        ablated_sample = replace(sample, audio=ablated_audio)
        ablated_proba, ablated_label = _predict_perturbed(
            model, pipeline, ablated_sample, pred_idx
        )
        scored.append(
            SvmUnitAttribution(
                label=f"{start:.2f}-{end:.2f}s",
                modality=Modality.AUDIO,
                index=index,
                probability_delta=pred_proba - ablated_proba,
                ablated_probability=ablated_proba,
                ablated_label=ablated_label,
                start=start,
                end=end,
            )
        )
    return _top_units(scored, top_k)


def _svm_video_unit_attributions(
    model: Classifier,
    pipeline: FeaturePipeline,
    sample: RawSample,
    pred_idx: int,
    pred_proba: float,
    top_k: int,
    window_seconds: float,
    max_units: int,
) -> tuple[SvmUnitAttribution, ...]:
    if sample.video is None or sample.video.frames is None:
        return ()
    frames = np.asarray(sample.video.frames, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[0] == 0:
        return ()
    candidates = _video_windows(frames.shape[0], float(sample.video.fps), window_seconds)
    if max_units > 0:
        candidates = candidates[:max_units]
    scored: list[SvmUnitAttribution] = []
    for index, candidate in enumerate(candidates):
        start_frame, end_frame, start, end, label = candidate
        masked = frames.copy()
        masked[start_frame:end_frame] = 0.0
        ablated_video = replace(sample.video, frames=masked)
        ablated_sample = replace(sample, video=ablated_video)
        ablated_proba, ablated_label = _predict_perturbed(
            model, pipeline, ablated_sample, pred_idx
        )
        scored.append(
            SvmUnitAttribution(
                label=label,
                modality=Modality.VIDEO,
                index=index,
                probability_delta=pred_proba - ablated_proba,
                ablated_probability=ablated_proba,
                ablated_label=ablated_label,
                start=start,
                end=end,
            )
        )
    return _top_units(scored, top_k)


def _predict_perturbed(
    model: Classifier,
    pipeline: FeaturePipeline,
    sample: RawSample,
    pred_idx: int,
) -> tuple[float, Emotion]:
    bundle = pipeline.transform((sample,), Split.TEST)
    proba = model.predict_proba(bundle)
    if proba.shape[0] != 1:
        raise ValueError(f"perturbed inference 결과 행 수가 1이 아닙니다: {proba.shape[0]}")
    return float(proba[0, pred_idx]), model.classes[int(np.argmax(proba[0]))]


def _top_units(units: Sequence[SvmUnitAttribution], top_k: int) -> tuple[SvmUnitAttribution, ...]:
    scored = list(units)
    scored.sort(key=lambda item: abs(item.probability_delta), reverse=True)
    return tuple(scored[:top_k])


def _time_windows(duration: float, window_seconds: float) -> list[tuple[float, float]]:
    if duration <= 0.0:
        return []
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        windows.append((float(start), float(end)))
        start = end
    return windows


def _video_windows(
    n_frames: int,
    fps: float,
    window_seconds: float,
) -> list[tuple[int, int, float | None, float | None, str]]:
    if n_frames <= 0:
        return []
    if fps <= 0.0:
        return [(idx, idx + 1, None, None, f"frame_{idx}") for idx in range(n_frames)]
    windows: list[tuple[int, int, float | None, float | None, str]] = []
    for start, end in _time_windows(n_frames / fps, window_seconds):
        start_frame = max(0, min(n_frames, int(np.floor(start * fps))))
        end_frame = max(start_frame + 1, min(n_frames, int(np.ceil(end * fps))))
        windows.append((start_frame, end_frame, start, end, f"{start:.2f}-{end:.2f}s"))
    return windows


def _with_feature_value(
    bundle: FeatureBundle,
    matrix_idx: int,
    col: int,
    value: float,
) -> FeatureBundle:
    matrices: list[FeatureMatrix] = []
    for idx, matrix in enumerate(bundle.matrices):
        values = matrix.values.copy()
        if idx == matrix_idx:
            values[:, col] = value
        matrices.append(
            FeatureMatrix(
                values=values,
                names=matrix.names,
                modality=matrix.modality,
                kind=matrix.kind,
                source=matrix.source,
            )
        )
    return FeatureBundle(
        uids=bundle.uids,
        matrices=tuple(matrices),
        sequence_matrices=bundle.sequence_matrices,
        availability=bundle.availability,
        utterances=bundle.utterances,
    )


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
