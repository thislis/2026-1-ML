"""단일 MP4+텍스트 입력에 대한 감정 추론 유틸리티."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.protocols import Classifier, FeatureExtractor
from meld_emotion.core.types import Emotion, Split
from meld_emotion.data.media import MediaLoader as RawMediaLoader
from meld_emotion.features.audio import Wav2Vec2XlsrAudioExtractor
from meld_emotion.features.text import EmbeddingGemmaTextExtractor
from meld_emotion.features.video import TimeSformerVideoExtractor
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
    pipeline = FeaturePipeline(
        extractors if extractors is not None else default_extractors(resolved_device),
        cache=NullFeatureCache(),
        media_loader=media_loader if media_loader is not None else default_media_loader(),
        media_error_policy="raise",
    )
    bundle = pipeline.fit_transform((sample,), Split.TEST)
    model = classifier if classifier is not None else _load_classifier(checkpoint, resolved_device)
    probabilities = model.predict_proba(bundle)
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
    label, probability = top[0]
    return InferenceResult(
        label=label,
        probability=probability,
        scores=scores,
        top_k=top,
        checkpoint=str(checkpoint),
        mp4_path=str(mp4),
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


def default_media_loader() -> MediaLoader:
    return RawMediaLoader(
        audio_sample_rate=16000,
        video_max_frames=8,
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
    return "\n".join(lines)


def result_to_json(result: InferenceResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


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
