"""Jina Embeddings v5 Omni 기반 fused multimodal embedding 추출기."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor

_SUPPORTED_OUTPUT_DIMS = frozenset((32, 64, 128, 256, 512, 768, 1024))
_SUPPORTED_DEVICES = frozenset(("cpu", "mps", "gpu"))


class _SentenceTransformerModel(Protocol):
    def encode(self, sentences: Sequence[object], **kwargs: object) -> object: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(self, model_name: str, **kwargs: object) -> _SentenceTransformerModel: ...


def _load_sentence_transformer_class() -> _SentenceTransformerFactory:
    try:
        module = import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            "JinaOmniMultimodalExtractor requires sentence-transformers. "
            "Install it with `uv sync --extra text --extra audio --extra video --extra deep`."
        ) from exc
    try:
        factory = module.__dict__["SentenceTransformer"]
    except KeyError as exc:
        raise ImportError(
            "The sentence_transformers package does not expose SentenceTransformer."
        ) from exc
    return cast(_SentenceTransformerFactory, factory)


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "JinaOmniMultimodalExtractor requires PyTorch. Install it with `uv sync --extra deep`."
        ) from exc


def _require_peft() -> None:
    try:
        import_module("peft")
    except ImportError as exc:
        raise ImportError(
            "Jina Omni remote modeling code requires peft. "
            "Install it with `uv sync --extra deep`."
        ) from exc


def _require_media_processor_dependencies() -> None:
    missing: list[str] = []
    for package in ("PIL", "torchvision"):
        try:
            import_module(package)
        except ImportError:
            missing.append("pillow" if package == "PIL" else package)
    if missing:
        packages = ", ".join(missing)
        raise ImportError(
            "Jina Omni image/video processor dependencies are missing: "
            f"{packages}. Install them with `uv sync --extra video`."
        )


def _load_soundfile_module() -> Any:
    try:
        return import_module("soundfile")
    except ImportError as exc:
        raise ImportError(
            "JinaOmniMultimodalExtractor needs soundfile to pass loaded audio segments to Jina. "
            "Install it with `uv sync --extra audio`."
        ) from exc


def _mapped_device(device: str) -> str:
    name = device.lower()
    if name not in _SUPPORTED_DEVICES:
        allowed = ", ".join(sorted(_SUPPORTED_DEVICES))
        raise ValueError(f"device 는 {allowed} 중 하나여야 합니다")
    return "cuda" if name == "gpu" else name


def _validate_device_available(mapped: str) -> None:
    if mapped == "cpu":
        return
    torch = _load_torch_module()
    if mapped == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("device='gpu' 를 요청했지만 CUDA 를 사용할 수 없습니다")
    if mapped == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        available = bool(mps is not None and mps.is_available())
        if not available:
            raise RuntimeError("device='mps' 를 요청했지만 PyTorch MPS 를 사용할 수 없습니다")


@real
class JinaOmniMultimodalExtractor(BaseFeatureExtractor):
    """Text/audio/video 를 한 번에 입력해 fused Jina Omni embedding 을 계산한다."""

    modality: ClassVar[Modality] = Modality.MULTIMODAL
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING
    required_modalities: ClassVar[tuple[Modality, ...]] = (
        Modality.AUDIO,
        Modality.VIDEO,
    )

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v5-omni-small",
        output_dim: int = 1024,
        batch_size: int = 4,
        task: str = "classification",
        device: str = "cpu",
        max_video_frames: int = 8,
    ) -> None:
        if output_dim not in _SUPPORTED_OUTPUT_DIMS:
            allowed = ", ".join(str(dim) for dim in sorted(_SUPPORTED_OUTPUT_DIMS))
            raise ValueError(f"output_dim 은 {allowed} 중 하나여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        if max_video_frames <= 0:
            raise ValueError("max_video_frames 는 양수여야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._task = task
        self._device = _mapped_device(device)
        self._max_video_frames = max_video_frames
        self._model: _SentenceTransformerModel | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"jina_omni_{i}" for i in range(self._output_dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        if not samples:
            return self._stack_rows((), self.names)

        rows: list[np.ndarray] = []
        with tempfile.TemporaryDirectory(prefix="meld_jina_omni_") as tmp_dir:
            for start in range(0, len(samples), self._batch_size):
                batch = samples[start : start + self._batch_size]
                inputs = [self._sample_input(sample, Path(tmp_dir), start + idx) for idx, sample in enumerate(batch)]
                try:
                    encoded = self._model_instance().encode(
                        inputs,
                        batch_size=self._batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        truncate_dim=self._output_dim,
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if "out of memory" in message.lower() or "MPS backend out of memory" in message:
                        raise RuntimeError(
                            "Jina Omni embedding 중 device 메모리가 부족합니다. "
                            "batch_size 를 1로 낮추고, max_video_frames/media.video_max_frames 를 줄이거나 "
                            "device: cpu 를 사용하세요. MPS에서는 max_video_frames: 8 이하를 권장합니다."
                        ) from exc
                    raise
                finally:
                    self._clear_device_cache()
                values = self._coerce_embeddings(encoded, len(batch))
                rows.extend(np.asarray(row, dtype=np.float64) for row in values)
        return self._stack_rows(rows, self.names)

    def _model_instance(self) -> _SentenceTransformerModel:
        if self._model is None:
            _validate_device_available(self._device)
            _require_peft()
            _require_media_processor_dependencies()
            factory = _load_sentence_transformer_class()
            try:
                self._model = factory(
                    self._model_name,
                    trust_remote_code=True,
                    device=self._device,
                    model_kwargs={"default_task": self._task, "modality": "omni"},
                )
            except Exception as exc:
                raise RuntimeError(
                    "Jina Omni 모델을 불러오지 못했습니다. "
                    "`uv sync --extra text --extra audio --extra video --extra deep` 로 "
                    "sentence-transformers/transformers/torch/torchvision/peft 의존성을 설치했고, "
                    "jinaai/jina-embeddings-v5-omni-small 모델 파일에 접근 가능한지 확인하세요. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                first_module = self._model[0]  # type: ignore[index]
            except (TypeError, AttributeError):
                first_module = None
            if first_module is not None and getattr(first_module, "processor", None) is None:
                raise RuntimeError(
                    "Jina Omni processor 를 로드하지 못했습니다. "
                    "`uv sync --extra video` 로 pillow/torchvision 을 설치한 뒤 다시 실행하세요."
                )
        return self._model

    def _sample_input(self, sample: RawSample, tmp_dir: Path, index: int) -> object:
        parts: list[object] = []
        if sample.has(Modality.TEXT) and sample.text:
            parts.append(sample.text)
        if sample.has(Modality.AUDIO):
            audio = self._audio_input(sample, tmp_dir, index)
            if audio is not None:
                parts.append(audio)
        if sample.has(Modality.VIDEO):
            video = self._video_input(sample)
            if video is not None:
                parts.append(video)
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return tuple(parts)

    def _audio_input(self, sample: RawSample, tmp_dir: Path, index: int) -> str | None:
        if sample.audio is None:
            return None
        if sample.audio.waveform is not None:
            wave = np.asarray(sample.audio.waveform, dtype=np.float32).reshape(-1)
            if wave.size == 0:
                return None
            path = tmp_dir / f"audio_{index}.wav"
            sf = _load_soundfile_module()
            sf.write(str(path), wave, sample.audio.sample_rate)
            return str(path)
        if sample.audio.source_path is not None:
            return str(sample.audio.source_path)
        return None

    def _video_input(self, sample: RawSample) -> np.ndarray | str | None:
        if sample.video is None:
            return None
        if sample.video.frames is not None:
            frames = np.asarray(sample.video.frames)
            if frames.size == 0:
                return None
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] >= 3:
                frames = frames[..., :3]
            else:
                return None
            if frames.dtype != np.uint8:
                if float(np.nanmax(frames)) <= 1.5:
                    frames = frames * 255.0
                frames = np.clip(np.nan_to_num(frames), 0.0, 255.0).astype(np.uint8)
            return self._sample_video_frames(frames)
        if sample.video.source_path is not None:
            return str(sample.video.source_path)
        return None

    def _sample_video_frames(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] <= self._max_video_frames:
            return frames
        indices = np.linspace(0, frames.shape[0] - 1, self._max_video_frames)
        return np.asarray(frames[np.rint(indices).astype(np.int64)], dtype=np.uint8)

    def _clear_device_cache(self) -> None:
        if self._device == "cpu":
            return
        torch = _load_torch_module()
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _coerce_embeddings(self, encoded: object, n_samples: int) -> np.ndarray:
        values = np.asarray(encoded, dtype=np.float64)
        if values.ndim == 1 and n_samples == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"Jina Omni 출력은 2차원이어야 합니다 (got ndim={values.ndim})")
        if values.shape[0] != n_samples:
            raise ValueError(
                f"Jina Omni 출력 행 수가 sample 수와 일치하지 않습니다: {values.shape[0]} != {n_samples}"
            )
        if values.shape[1] < self._output_dim:
            raise ValueError(
                "Jina Omni 출력 차원이 설정값보다 작습니다: "
                f"{values.shape[1]} < {self._output_dim}"
            )
        if values.shape[1] > self._output_dim:
            values = values[:, : self._output_dim]
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        return np.asarray(values, dtype=np.float64)
