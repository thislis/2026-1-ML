"""VideoPrism 기반 비디오 임베딩 추출기.

``google/videoprism-base-f16r288`` 을 Google DeepMind VideoPrism JAX/Flax 구현으로
lazy-load 해 발화 비디오 프레임을 고정 길이 dense embedding 으로 변환한다. VideoPrism 의
출력은 spatiotemporal patch token 묶음이므로, 이 추출기는 모든 token 을 평균 풀링해 발화 단위
벡터를 만든 뒤 필요하면 앞 ``output_dim`` 차원으로 잘라 쓴다.
"""

from __future__ import annotations

import sys
from builtins import open as builtin_open
from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor


def _load_videoprism_modules() -> tuple[Any, Any]:
    try:
        jax = import_module("jax")
        _ensure_tensorflow_gfile_shim()
        videoprism_models = import_module("videoprism.models")
    except ImportError as exc:
        raise ImportError(
            "VideoPrismVideoExtractor requires JAX and the Google DeepMind VideoPrism "
            "package. Install them with `uv sync --extra video`, or install VideoPrism "
            "from https://github.com/google-deepmind/videoprism."
        ) from exc
    return jax, videoprism_models


def _ensure_tensorflow_gfile_shim() -> None:
    try:
        tensorflow_module = import_module("tensorflow")
        io_module = import_module("tensorflow.io")
    except ImportError:
        tensorflow_module = ModuleType("tensorflow")
        io_module = ModuleType("tensorflow.io")
        tensorflow_module.__dict__["io"] = io_module
        sys.modules.setdefault("tensorflow", tensorflow_module)
        sys.modules.setdefault("tensorflow.io", io_module)

    if not hasattr(io_module, "gfile"):
        io_module.__dict__["gfile"] = SimpleNamespace(GFile=builtin_open)
    if not hasattr(tensorflow_module, "io"):
        tensorflow_module.__dict__["io"] = io_module
    for name in ("Tensor", "Variable", "RaggedTensor"):
        if not hasattr(tensorflow_module, name):
            tensorflow_module.__dict__[name] = type(name, (), {})


@real
class VideoPrismVideoExtractor(BaseFeatureExtractor):
    """Google VideoPrism utterance-level video embedding extractor."""

    modality: ClassVar[Modality] = Modality.VIDEO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "google/videoprism-base-f16r288",
        output_dim: int = 768,
        num_frames: int | None = 16,
        frame_size: int = 288,
        normalize: bool = True,
        prefer_batched_input: bool = True,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim 은 양수여야 합니다")
        if num_frames is not None and num_frames <= 0:
            raise ValueError("num_frames 는 양수이거나 None 이어야 합니다")
        if frame_size <= 0:
            raise ValueError("frame_size 는 양수여야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._num_frames = num_frames
        self._frame_size = frame_size
        self._normalize = normalize
        self._prefer_batched_input = prefer_batched_input
        self._jax: Any | None = None
        self._model: Any | None = None
        self._state: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"videoprism_{i}" for i in range(self._output_dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        if not samples:
            return self._stack_rows((), self.names)

        values = np.zeros((len(samples), self._output_dim), dtype=np.float64)
        for idx, sample in enumerate(samples):
            frames = self._frames_or_none(sample)
            if frames is not None:
                values[idx] = self._embed_frames(frames)
        return self._matrix(values, self.names)

    def _frames_or_none(self, sample: RawSample) -> np.ndarray | None:
        if sample.video is None or sample.video.frames is None:
            return None
        frames = np.asarray(sample.video.frames, dtype=np.float32)
        if frames.size == 0:
            return None
        if frames.ndim != 4:
            raise ValueError(
                "VideoPrism 입력 frames 는 (T, H, W, C) 형식이어야 합니다: "
                f"ndim={frames.ndim}"
            )
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] >= 3:
            frames = frames[..., :3]
        else:
            raise ValueError(f"VideoPrism 입력 channel 수가 올바르지 않습니다: {frames.shape[-1]}")
        if float(np.nanmax(frames)) > 1.5:
            frames = frames / 255.0
        frames = np.nan_to_num(frames, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        frames = np.clip(frames, 0.0, 1.0)
        frames = self._sample_frames(frames)
        return self._resize_frames(frames)

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        if self._num_frames is None or frames.shape[0] == self._num_frames:
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

    def _embed_frames(self, frames: np.ndarray) -> np.ndarray:
        model, state, jax = self._model_parts()
        primary = frames[np.newaxis, ...] if self._prefer_batched_input else frames
        try:
            output = model.apply(state, jax.device_put(primary), train=False)
        except Exception as primary_exc:
            if not self._prefer_batched_input:
                raise
            try:
                output = model.apply(state, jax.device_put(frames), train=False)
            except Exception as fallback_exc:
                raise primary_exc from fallback_exc
        embedding = self._pool_output(output)
        if embedding.shape[0] < self._output_dim:
            raise ValueError(
                "VideoPrism 출력 차원이 설정값보다 작습니다: "
                f"{embedding.shape[0]} < {self._output_dim}"
            )
        if embedding.shape[0] > self._output_dim:
            embedding = embedding[: self._output_dim]
        if self._normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0.0:
                embedding = embedding / norm
        return np.asarray(embedding, dtype=np.float64)

    def _model_parts(self) -> tuple[Any, Any, Any]:
        if self._model is None or self._state is None or self._jax is None:
            jax, vp = _load_videoprism_modules()
            try:
                self._model = vp.get_model(self._model_name)
                self._state = vp.load_pretrained_weights(self._model_name)
            except Exception as exc:
                raise RuntimeError(
                    "VideoPrism 모델을 불러오지 못했습니다. "
                    "`uv sync --extra video` 로 JAX/Flax/VideoPrism 의존성을 설치했고, "
                    "google/videoprism-base-f16r288 checkpoint 에 접근 가능한지 확인하세요. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._jax = jax
        return self._model, self._state, self._jax

    def _pool_output(self, output: object) -> np.ndarray:
        value = self._first_tensor(output)
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("VideoPrism 출력은 최소 1차원이어야 합니다")
        if array.ndim >= 3 and array.shape[0] == 1:
            array = array[0]
        pooled = array if array.ndim == 1 else array.mean(axis=tuple(range(array.ndim - 1)))
        return np.asarray(pooled, dtype=np.float64).reshape(-1)

    def _first_tensor(self, output: object) -> object:
        if isinstance(output, Mapping):
            for key in (
                "video_embeddings",
                "video_embedding",
                "embeddings",
                "embedding",
                "features",
            ):
                if key in output:
                    return output[key]
            values = list(output.values())
            if values:
                return values[0]
        if isinstance(output, tuple | list):
            if output:
                return output[0]
            raise ValueError("VideoPrism 출력 tuple/list 가 비어 있습니다")
        return output
