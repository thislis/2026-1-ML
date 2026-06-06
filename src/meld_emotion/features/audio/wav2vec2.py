"""Wav2Vec2 XLS-R 기반 오디오 임베딩 추출기.

``facebook/wav2vec2-xls-r-300m`` 을 Hugging Face Transformers 로 lazy-load 해 mono waveform 을
발화 단위 dense embedding 으로 변환한다. 이 base checkpoint 는 ASR tokenizer 가 없으므로
``AutoFeatureExtractor`` + ``Wav2Vec2Model`` 경로를 사용한다. 학습 상태가 필요 없어 ``fit`` 은
no-op 이며, 출력은 예측용 ``FeatureKind.EMBEDDING`` 행렬이다.
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


def _load_transformers_classes() -> tuple[Any, Any]:
    try:
        module: Any = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2XlsrAudioExtractor requires the 'transformers' package. "
            "Install it with `uv sync --extra audio`."
        ) from exc
    try:
        processor_cls = module.AutoFeatureExtractor
        model_cls = module.Wav2Vec2Model
    except AttributeError as exc:
        raise ImportError(
            "The 'transformers' package does not expose Wav2Vec2 feature-extraction classes."
        ) from exc
    return processor_cls, model_cls


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2XlsrAudioExtractor requires PyTorch. "
            "Install it with `uv sync --extra audio`."
        ) from exc


@real
class Wav2Vec2XlsrAudioExtractor(BaseFeatureExtractor):
    """Facebook Wav2Vec2 XLS-R utterance embedding extractor."""

    modality: ClassVar[Modality] = Modality.AUDIO
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        output_dim: int = 1024,
        batch_size: int = 4,
        sampling_rate: int = 16000,
        max_seconds: float | None = None,
        chunk_seconds: float | None = 30.0,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim 은 양수여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate 는 양수여야 합니다")
        if max_seconds is not None and max_seconds <= 0.0:
            raise ValueError("max_seconds 는 양수이거나 None 이어야 합니다")
        if chunk_seconds is not None and chunk_seconds <= 0.0:
            raise ValueError("chunk_seconds 는 양수이거나 None 이어야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._sampling_rate = sampling_rate
        self._max_seconds = max_seconds
        self._chunk_seconds = chunk_seconds
        self._normalize = normalize
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"wav2vec2_xlsr_{i}" for i in range(self._output_dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        if not samples:
            return self._stack_rows((), self.names)

        values = np.zeros((len(samples), self._output_dim), dtype=np.float64)
        valid: list[tuple[int, np.ndarray]] = []
        for idx, sample in enumerate(samples):
            waveform = self._waveform_or_none(sample)
            if waveform is not None:
                valid.append((idx, waveform))

        for row, waveform in valid:
            values[row] = self._embed_waveform(waveform)

        return self._matrix(values, self.names)

    def _waveform_or_none(self, sample: RawSample) -> np.ndarray | None:
        if sample.audio is None or sample.audio.waveform is None:
            return None
        if sample.audio.sample_rate != self._sampling_rate:
            raise ValueError(
                "Wav2Vec2 XLS-R 입력 sample_rate 가 설정값과 다릅니다: "
                f"{sample.audio.sample_rate} != {self._sampling_rate}. "
                "MediaConfig.audio_sample_rate 를 16000으로 맞추거나 resampling 후 사용하세요."
            )
        wave = np.asarray(sample.audio.waveform, dtype=np.float32).reshape(-1)
        if self._max_seconds is not None:
            max_samples = max(1, int(round(self._sampling_rate * self._max_seconds)))
            if wave.size > max_samples:
                wave = wave[:max_samples]
        return wave if wave.size > 0 else None

    def _embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        chunks = self._chunks(waveform)
        embeddings: list[np.ndarray] = []
        weights: list[int] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            embeddings.extend(self._embed_batch(batch))
            weights.extend(chunk.size for chunk in batch)
        if not embeddings:
            return np.zeros(self._output_dim, dtype=np.float64)
        stacked = np.vstack(embeddings)
        averaged = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float64))
        if self._normalize:
            norm = np.linalg.norm(averaged)
            if norm > 0:
                averaged = averaged / norm
        return np.asarray(averaged, dtype=np.float64)

    def _chunks(self, waveform: np.ndarray) -> list[np.ndarray]:
        if self._chunk_seconds is None:
            return [waveform]
        chunk_size = max(1, int(round(self._sampling_rate * self._chunk_seconds)))
        if waveform.size <= chunk_size:
            return [waveform]
        return [waveform[start : start + chunk_size] for start in range(0, waveform.size, chunk_size)]

    def _embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
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
        pooled = self._mean_pool(output.last_hidden_state, inputs.get("attention_mask"))
        embeddings = np.asarray(pooled.detach().cpu().numpy(), dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Wav2Vec2 출력은 2차원이어야 합니다 (got ndim={embeddings.ndim})"
            )
        if embeddings.shape[1] < self._output_dim:
            raise ValueError(
                "Wav2Vec2 출력 차원이 설정값보다 작습니다: "
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
                    "Wav2Vec2 XLS-R 모델을 불러오지 못했습니다. "
                    "`uv sync --extra audio` 로 transformers/PyTorch 의존성을 설치했고, "
                    "facebook/wav2vec2-xls-r-300m 모델 파일에 접근 가능한지 확인하세요. "
                    "이 extractor 는 16kHz mono waveform 입력을 기대합니다. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._processor = processor
            self._model = model
            self._torch = torch
        return self._processor, self._model, self._torch

    def _mean_pool(self, hidden: Any, attention_mask: Any | None) -> Any:
        if attention_mask is None:
            return hidden.mean(dim=1)

        feature_mask = attention_mask
        projector = getattr(self._model, "_get_feature_vector_attention_mask", None)
        if callable(projector):
            feature_mask = projector(hidden.shape[1], attention_mask)
        elif feature_mask.shape[1] != hidden.shape[1]:
            return hidden.mean(dim=1)

        feature_mask = feature_mask.to(hidden.device).unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * feature_mask).sum(dim=1)
        counts = feature_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
