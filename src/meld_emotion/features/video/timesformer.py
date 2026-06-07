"""TimeSformer 기반 비디오 임베딩 추출기.

``facebook/timesformer-base-finetuned-k400`` 을 Hugging Face Transformers 로 lazy-load 해
발화 비디오 프레임을 고정 길이 dense embedding 으로 변환한다. 원 논문/공식 구현의 기본
TimeSformer 설정(8프레임, 224x224, divided space-time attention)을 기본값으로 따르며,
Transformers 의 ``TimesformerModel`` 이 반환하는 hidden state 의 CLS token 또는 평균 pooled
token 을 발화 단위 벡터로 사용한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _load_timesformer_model_class() -> Any:
    try:
        module: Any = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "TimeSformerVideoExtractor requires the 'transformers' package. "
            "Install it with `uv sync --extra video`."
        ) from exc
    try:
        return module.TimesformerModel
    except AttributeError as exc:
        raise ImportError(
            "The 'transformers' package does not expose TimesformerModel. "
            "Install a recent transformers release with `uv sync --extra video`."
        ) from exc


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "TimeSformerVideoExtractor requires PyTorch. Install it with `uv sync --extra video`."
        ) from exc


@real
class TimeSformerVideoExtractor(BaseFeatureExtractor):
    """Facebook TimeSformer utterance-level video embedding extractor."""

    modality: ClassVar[Modality] = Modality.VIDEO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        output_dim: int = 768,
        batch_size: int = 2,
        num_frames: int = 8,
        frame_size: int = 224,
        normalize: bool = True,
        pooling: str = "cls",
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
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling 은 'cls' 또는 'mean' 이어야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._frame_size = frame_size
        self._normalize = normalize
        self._pooling = pooling
        self._device = device
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"timesformer_{i}" for i in range(self._output_dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        if not samples:
            return self._stack_rows((), self.names)

        values = np.zeros((len(samples), self._output_dim), dtype=np.float64)
        valid: list[tuple[int, np.ndarray]] = []
        for idx, sample in enumerate(samples):
            frames = self._frames_or_none(sample)
            if frames is not None:
                valid.append((idx, frames))

        for start in range(0, len(valid), self._batch_size):
            batch = valid[start : start + self._batch_size]
            embeddings = self._embed_batch([frames for _, frames in batch])
            for row, embedding in zip((idx for idx, _ in batch), embeddings, strict=True):
                values[row] = embedding

        return self._matrix(values, self.names)

    def _frames_or_none(self, sample: RawSample) -> np.ndarray | None:
        if sample.video is None or sample.video.frames is None:
            return None
        frames = np.asarray(sample.video.frames, dtype=np.float32)
        if frames.size == 0:
            return None
        if frames.ndim != 4:
            raise ValueError(
                "TimeSformer 입력 frames 는 (T, H, W, C) 형식이어야 합니다: "
                f"ndim={frames.ndim}"
            )
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] >= 3:
            frames = frames[..., :3]
        else:
            raise ValueError(f"TimeSformer 입력 channel 수가 올바르지 않습니다: {frames.shape[-1]}")
        if float(np.nanmax(frames)) > 1.5:
            frames = frames / 255.0
        frames = np.nan_to_num(frames, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        frames = np.clip(frames, 0.0, 1.0)
        frames = self._sample_frames(frames)
        return self._resize_frames(frames)

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == self._num_frames:
            return frames
        indices = np.linspace(0, frames.shape[0] - 1, self._num_frames)
        return frames[np.rint(indices).astype(np.int64)]

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[1] == self._frame_size and frames.shape[2] == self._frame_size:
            return np.asarray(frames, dtype=np.float32)

        try:
            cv2: Any = import_module("cv2")
        except ImportError:
            resized = [self._nearest_resize(frame, self._frame_size) for frame in frames]
        else:
            resized = [
                cv2.resize(
                    frame,
                    (self._frame_size, self._frame_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                for frame in frames
            ]
        return np.asarray(resized, dtype=np.float32)

    @staticmethod
    def _nearest_resize(frame: np.ndarray, size: int) -> np.ndarray:
        y = np.rint(np.linspace(0, frame.shape[0] - 1, size)).astype(np.int64)
        x = np.rint(np.linspace(0, frame.shape[1] - 1, size)).astype(np.int64)
        return frame[y][:, x]

    def _embed_batch(self, frames_batch: Sequence[np.ndarray]) -> np.ndarray:
        model, torch = self._model_parts()
        pixel_values = self._pixel_values(frames_batch)
        tensor = torch.as_tensor(pixel_values, dtype=torch.float32)
        if self._device is not None:
            tensor = tensor.to(self._device)
        with torch.no_grad():
            output = model(pixel_values=tensor)
        hidden = self._last_hidden_state(output)
        pooled = self._pool_hidden(hidden)
        embeddings = np.asarray(pooled.detach().cpu().numpy(), dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"TimeSformer pooled 출력은 2차원이어야 합니다 (got ndim={embeddings.ndim})"
            )
        if embeddings.shape[1] < self._output_dim:
            raise ValueError(
                "TimeSformer 출력 차원이 설정값보다 작습니다: "
                f"{embeddings.shape[1]} < {self._output_dim}"
            )
        if embeddings.shape[1] > self._output_dim:
            embeddings = embeddings[:, : self._output_dim]
        if self._normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(
                embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0
            )
        return embeddings

    def _pixel_values(self, frames_batch: Sequence[np.ndarray]) -> np.ndarray:
        prepared: list[np.ndarray] = []
        for frames in frames_batch:
            normalized = (frames - _IMAGENET_MEAN) / _IMAGENET_STD
            prepared.append(np.transpose(normalized, (0, 3, 1, 2)))
        return np.asarray(prepared, dtype=np.float32)

    def _model_parts(self) -> tuple[Any, Any]:
        if self._model is None or self._torch is None:
            model_cls = _load_timesformer_model_class()
            torch = _load_torch_module()
            try:
                model = model_cls.from_pretrained(self._model_name)
                model.eval()
                if self._device is not None:
                    model.to(self._device)
            except Exception as exc:
                raise RuntimeError(
                    "TimeSformer 모델을 불러오지 못했습니다. "
                    "`uv sync --extra video` 로 transformers/PyTorch 의존성을 설치했고, "
                    "facebook/timesformer-base-finetuned-k400 모델 파일에 접근 가능한지 확인하세요. "
                    "이 extractor 는 (T,H,W,C) RGB 프레임을 8프레임 224x224 입력으로 맞춥니다. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._model = model
            self._torch = torch
        return self._model, self._torch

    @staticmethod
    def _last_hidden_state(output: object) -> Any:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, tuple | list) and output:
            return output[0]
        raise ValueError("TimeSformer 출력에서 last_hidden_state 를 찾을 수 없습니다")

    def _pool_hidden(self, hidden: Any) -> Any:
        if hidden.ndim != 3:
            raise ValueError(f"TimeSformer hidden state 는 3차원이어야 합니다: ndim={hidden.ndim}")
        if self._pooling == "cls":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)
