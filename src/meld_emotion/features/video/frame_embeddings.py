"""Frame-level video embeddings for fine-grained XAI."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureUnit, SequenceFeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseSequenceFeatureExtractor


def _load_transformers_classes() -> tuple[Any, Any]:
    try:
        module: Any = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "VideoFrameEmbeddingExtractor requires transformers. "
            "Install it with `uv sync --extra video`."
        ) from exc
    try:
        return module.CLIPProcessor, module.CLIPVisionModel
    except AttributeError as exc:
        raise ImportError("transformers does not expose CLIPProcessor/CLIPVisionModel") from exc


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "VideoFrameEmbeddingExtractor requires PyTorch. Install it with `uv sync --extra video`."
        ) from exc


@real
class VideoFrameEmbeddingExtractor(BaseSequenceFeatureExtractor):
    """Sample frames and encode each frame with a CLIP vision model."""

    modality: ClassVar[Modality] = Modality.VIDEO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 768,
        batch_size: int = 8,
        num_frames: int = 16,
        frame_size: int = 224,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim 은 양수여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        if num_frames <= 0:
            raise ValueError("num_frames 는 양수여야 합니다")
        if frame_size <= 0:
            raise ValueError("frame_size 는 양수여야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._frame_size = frame_size
        self._normalize = normalize
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"video_frame_{i}" for i in range(self._output_dim))

    def transform_sequence(self, samples: Sequence[RawSample]) -> SequenceFeatureMatrix:
        values = np.zeros((len(samples), self._num_frames, self._output_dim), dtype=np.float64)
        mask = np.zeros((len(samples), self._num_frames), dtype=bool)
        units: list[tuple[FeatureUnit, ...]] = [() for _ in samples]
        flattened: list[np.ndarray] = []
        positions: list[tuple[int, int]] = []
        for row, sample in enumerate(samples):
            frames = self._frames_or_none(sample)
            if frames is None:
                continue
            sampled, source_indices = self._sample_frames(frames)
            row_units: list[FeatureUnit] = []
            fps = sample.video.fps if sample.video is not None and sample.video.fps > 0 else 0.0
            for local, (frame, source_index) in enumerate(zip(sampled, source_indices, strict=True)):
                flattened.append(frame)
                positions.append((row, local))
                start = float(source_index / fps) if fps > 0 else None
                end = float((source_index + 1) / fps) if fps > 0 else None
                row_units.append(
                    FeatureUnit(
                        label=f"frame_{source_index}",
                        index=local,
                        start=start,
                        end=end,
                    )
                )
            mask[row, : len(row_units)] = True
            units[row] = tuple(row_units)

        for start in range(0, len(flattened), self._batch_size):
            batch = flattened[start : start + self._batch_size]
            embeddings = self._embed_frames(batch)
            for (row, local), embedding in zip(positions[start : start + self._batch_size], embeddings, strict=True):
                values[row, local] = embedding
        return self._sequence_matrix(values, mask, units, self.names)

    def _frames_or_none(self, sample: RawSample) -> np.ndarray | None:
        if sample.video is None or sample.video.frames is None:
            return None
        frames = np.asarray(sample.video.frames, dtype=np.float32)
        if frames.ndim != 4 or frames.size == 0:
            return None
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] >= 3:
            frames = frames[..., :3]
        else:
            return None
        if float(np.nanmax(frames)) > 1.5:
            frames = frames / 255.0
        frames = np.nan_to_num(frames, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return np.asarray(np.clip(frames, 0.0, 1.0), dtype=np.float32)

    def _sample_frames(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.rint(np.linspace(0, frames.shape[0] - 1, self._num_frames)).astype(np.int64)
        return frames[indices], indices

    def _embed_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        processor, model, torch = self._model_parts()
        images = [(frame * 255.0).astype(np.uint8) for frame in frames]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self._device) if self._device is not None and hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }
        with torch.no_grad():
            output = model(**inputs)
        pooled = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
        embeddings = np.asarray(pooled.detach().cpu().numpy(), dtype=np.float64)
        if embeddings.shape[1] < self._output_dim:
            raise ValueError(
                "video frame embedding dim 이 설정값보다 작습니다: "
                f"{embeddings.shape[1]} < {self._output_dim}"
            )
        embeddings = embeddings[:, : self._output_dim]
        if self._normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(
                embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0
            )
        return np.asarray(embeddings, dtype=np.float64)

    def _model_parts(self) -> tuple[Any, Any, Any]:
        if self._processor is None or self._model is None or self._torch is None:
            processor_cls, model_cls = _load_transformers_classes()
            torch = _load_torch_module()
            try:
                processor = processor_cls.from_pretrained(self._model_name)
                model = model_cls.from_pretrained(self._model_name)
                model.eval()
                if self._device is not None:
                    model.to(self._device)
            except Exception as exc:
                raise RuntimeError(
                    "Video frame embedding 모델을 불러오지 못했습니다. "
                    "`uv sync --extra video` 의존성과 모델 접근성을 확인하세요. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._processor = processor
            self._model = model
            self._torch = torch
        return self._processor, self._model, self._torch
