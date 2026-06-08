"""Wav2Vec2 hidden-step audio embeddings for fine-grained XAI."""

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
            "Wav2Vec2XlsrAudioSequenceExtractor requires transformers. "
            "Install it with `uv sync --extra audio`."
        ) from exc
    try:
        return module.AutoFeatureExtractor, module.Wav2Vec2Model
    except AttributeError as exc:
        raise ImportError("transformers does not expose Wav2Vec2 classes") from exc


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2XlsrAudioSequenceExtractor requires PyTorch. "
            "Install it with `uv sync --extra audio`."
        ) from exc


@real
class Wav2Vec2XlsrAudioSequenceExtractor(BaseSequenceFeatureExtractor):
    """Wav2Vec2 XLS-R sequence extractor with time-span metadata."""

    modality: ClassVar[Modality] = Modality.AUDIO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        output_dim: int = 1024,
        batch_size: int = 4,
        sampling_rate: int = 16000,
        max_seconds: float | None = None,
        max_steps: int = 128,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim 은 양수여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate 는 양수여야 합니다")
        if max_steps <= 0:
            raise ValueError("max_steps 는 양수여야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._sampling_rate = sampling_rate
        self._max_seconds = max_seconds
        self._max_steps = max_steps
        self._normalize = normalize
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"wav2vec2_step_{i}" for i in range(self._output_dim))

    def transform_sequence(self, samples: Sequence[RawSample]) -> SequenceFeatureMatrix:
        values = np.zeros((len(samples), self._max_steps, self._output_dim), dtype=np.float64)
        mask = np.zeros((len(samples), self._max_steps), dtype=bool)
        units: list[tuple[FeatureUnit, ...]] = [() for _ in samples]
        valid: list[tuple[int, np.ndarray]] = []
        for idx, sample in enumerate(samples):
            waveform = self._waveform_or_none(sample)
            if waveform is not None:
                valid.append((idx, waveform))
        for start in range(0, len(valid), self._batch_size):
            batch = valid[start : start + self._batch_size]
            seqs = self._embed_batch([waveform for _, waveform in batch])
            for (row, waveform), sequence in zip(batch, seqs, strict=True):
                selected, selected_units = self._select_steps(sequence, waveform.size)
                length = selected.shape[0]
                values[row, :length] = selected
                mask[row, :length] = True
                units[row] = selected_units
        return self._sequence_matrix(values, mask, units, self.names)

    def _waveform_or_none(self, sample: RawSample) -> np.ndarray | None:
        if sample.audio is None or sample.audio.waveform is None:
            return None
        if sample.audio.sample_rate != self._sampling_rate:
            raise ValueError(
                "Wav2Vec2 sequence 입력 sample_rate 가 설정값과 다릅니다: "
                f"{sample.audio.sample_rate} != {self._sampling_rate}"
            )
        wave = np.asarray(sample.audio.waveform, dtype=np.float32).reshape(-1)
        if self._max_seconds is not None:
            max_samples = max(1, round(self._sampling_rate * self._max_seconds))
            wave = wave[:max_samples]
        return wave if wave.size > 0 else None

    def _embed_batch(self, waveforms: Sequence[np.ndarray]) -> list[np.ndarray]:
        processor, model, torch = self._model_parts()
        inputs = processor(
            list(waveforms),
            sampling_rate=self._sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device) if self._device is not None and hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }
        with torch.no_grad():
            output = model(**inputs)
        hidden = output.last_hidden_state
        feature_mask = self._feature_mask(hidden, inputs.get("attention_mask"))
        hidden_np = np.asarray(hidden.detach().cpu().numpy(), dtype=np.float64)
        mask_np = np.asarray(feature_mask.detach().cpu().numpy(), dtype=bool)
        hidden_np = self._coerce_dim(hidden_np)
        return [hidden_np[i, mask_np[i]] for i in range(hidden_np.shape[0])]

    def _feature_mask(self, hidden: Any, attention_mask: Any | None) -> Any:
        if attention_mask is None:
            torch = self._torch if self._torch is not None else _load_torch_module()
            return torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        projector = getattr(self._model, "_get_feature_vector_attention_mask", None)
        if callable(projector):
            return projector(hidden.shape[1], attention_mask).to(hidden.device)
        if attention_mask.shape[1] == hidden.shape[1]:
            return attention_mask.to(hidden.device).to(dtype=hidden.dtype) > 0
        torch = self._torch if self._torch is not None else _load_torch_module()
        return torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)

    def _coerce_dim(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.shape[-1] < self._output_dim:
            raise ValueError(
                "Wav2Vec2 hidden dim 이 설정값보다 작습니다: "
                f"{hidden.shape[-1]} < {self._output_dim}"
            )
        hidden = hidden[..., : self._output_dim]
        if self._normalize:
            norms = np.linalg.norm(hidden, axis=-1, keepdims=True)
            hidden = np.divide(hidden, norms, out=np.zeros_like(hidden), where=norms > 0)
        return np.asarray(hidden, dtype=np.float64)

    def _select_steps(self, sequence: np.ndarray, n_samples: int) -> tuple[np.ndarray, tuple[FeatureUnit, ...]]:
        if sequence.shape[0] == 0:
            return (
                np.zeros((0, self._output_dim), dtype=np.float64),
                (),
            )
        if sequence.shape[0] > self._max_steps:
            indices = np.rint(np.linspace(0, sequence.shape[0] - 1, self._max_steps)).astype(np.int64)
            selected = sequence[indices]
        else:
            indices = np.arange(sequence.shape[0], dtype=np.int64)
            selected = sequence
        duration = float(n_samples) / float(self._sampling_rate)
        step_seconds = duration / float(max(sequence.shape[0], 1))
        units = tuple(
            FeatureUnit(
                label=f"{idx * step_seconds:.2f}-{(idx + 1) * step_seconds:.2f}s",
                index=local,
                start=float(idx * step_seconds),
                end=float((idx + 1) * step_seconds),
            )
            for local, idx in enumerate(indices.tolist())
        )
        return np.asarray(selected, dtype=np.float64), units

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
                    "Wav2Vec2 sequence 모델을 불러오지 못했습니다. "
                    "`uv sync --extra audio` 의존성과 모델 접근성을 확인하세요. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._processor = processor
            self._model = model
            self._torch = torch
        return self._processor, self._model, self._torch
