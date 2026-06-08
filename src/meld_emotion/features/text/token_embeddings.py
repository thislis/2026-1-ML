"""Token-level text embeddings for fine-grained dialogue XAI."""

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
            "TextTokenEmbeddingExtractor requires transformers. "
            "Install it with `uv sync --extra text`."
        ) from exc
    try:
        return module.AutoTokenizer, module.AutoModel
    except AttributeError as exc:
        raise ImportError("transformers does not expose AutoTokenizer/AutoModel") from exc


def _load_torch_module() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "TextTokenEmbeddingExtractor requires PyTorch. Install it with `uv sync --extra deep`."
        ) from exc


@real
class TextTokenEmbeddingExtractor(BaseSequenceFeatureExtractor):
    """HF token embeddings with token labels and character spans."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 64,
        output_dim: int = 768,
        batch_size: int = 16,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 는 양수여야 합니다")
        if output_dim <= 0:
            raise ValueError("output_dim 은 양수여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._normalize = normalize
        self._device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"text_token_{i}" for i in range(self._output_dim))

    def transform_sequence(self, samples: Sequence[RawSample]) -> SequenceFeatureMatrix:
        values = np.zeros((len(samples), self._max_tokens, self._output_dim), dtype=np.float64)
        mask = np.zeros((len(samples), self._max_tokens), dtype=bool)
        units: list[tuple[FeatureUnit, ...]] = [() for _ in samples]
        if not samples:
            return self._sequence_matrix(values, mask, units, self.names)

        tokenizer, model, torch = self._model_parts()
        for start in range(0, len(samples), self._batch_size):
            batch = samples[start : start + self._batch_size]
            texts = [sample.text for sample in batch]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self._max_tokens,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = np.asarray(encoded.pop("offset_mapping").cpu().numpy(), dtype=np.int64)
            inputs = {
                key: value.to(self._device) if self._device is not None and hasattr(value, "to") else value
                for key, value in dict(encoded).items()
            }
            with torch.no_grad():
                output = model(**inputs)
            hidden = np.asarray(output.last_hidden_state.detach().cpu().numpy(), dtype=np.float64)
            hidden = self._coerce_dim(hidden)
            attention = np.asarray(inputs["attention_mask"].detach().cpu().numpy(), dtype=bool)
            tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"].cpu().tolist()]
            for local, _sample in enumerate(batch):
                row = start + local
                values[row] = hidden[local]
                mask[row] = attention[local]
                row_units: list[FeatureUnit] = []
                for idx, valid in enumerate(attention[local]):
                    if not valid:
                        continue
                    char_start, char_end = int(offsets[local, idx, 0]), int(offsets[local, idx, 1])
                    row_units.append(
                        FeatureUnit(
                            label=str(tokens[local][idx]),
                            index=idx,
                            char_start=char_start if char_end > char_start else None,
                            char_end=char_end if char_end > char_start else None,
                        )
                    )
                units[row] = tuple(row_units)
        return self._sequence_matrix(values, mask, units, self.names)

    def _coerce_dim(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.shape[-1] < self._output_dim:
            raise ValueError(
                "text token embedding dim 이 설정값보다 작습니다: "
                f"{hidden.shape[-1]} < {self._output_dim}"
            )
        hidden = hidden[..., : self._output_dim]
        if self._normalize:
            norms = np.linalg.norm(hidden, axis=-1, keepdims=True)
            hidden = np.divide(hidden, norms, out=np.zeros_like(hidden), where=norms > 0)
        return np.asarray(hidden, dtype=np.float64)

    def _model_parts(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is None or self._model is None or self._torch is None:
            tokenizer_cls, model_cls = _load_transformers_classes()
            torch = _load_torch_module()
            try:
                tokenizer = tokenizer_cls.from_pretrained(self._model_name, use_fast=True)
                model = model_cls.from_pretrained(self._model_name)
                model.eval()
                if self._device is not None:
                    model.to(self._device)
            except Exception as exc:
                raise RuntimeError(
                    "Token embedding 모델을 불러오지 못했습니다. "
                    "`uv sync --extra text --extra deep` 의존성과 모델 접근성을 확인하세요. "
                    f"원인: {type(exc).__name__}: {exc}"
                ) from exc
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
        return self._tokenizer, self._model, self._torch
