"""EmbeddingGemma 기반 텍스트 임베딩 추출기.

``google/embeddinggemma-300m`` 을 Sentence Transformers 로 lazy-load 해 발화 텍스트를
고정 차원 dense embedding 으로 변환한다. BoW 와 마찬가지로 학습 상태가 필요 없어 ``fit`` 은
no-op 이며, 출력은 예측용 ``FeatureKind.EMBEDDING`` 행렬이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import ClassVar, Protocol, cast

import numpy as np

from meld_emotion.core.data import RawSample
from meld_emotion.core.features import FeatureMatrix
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.features.base import BaseFeatureExtractor

_SUPPORTED_OUTPUT_DIMS = frozenset((128, 256, 512, 768))


class _SentenceTransformerModel(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(self, model_name: str, **kwargs: object) -> _SentenceTransformerModel: ...


def _load_sentence_transformer_class() -> _SentenceTransformerFactory:
    try:
        module = import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            "EmbeddingGemmaTextExtractor requires the 'sentence-transformers' package. "
            "Install it with `uv sync --extra text`."
        ) from exc
    try:
        factory = module.__dict__["SentenceTransformer"]
    except KeyError as exc:
        raise ImportError(
            "The 'sentence_transformers' package does not expose SentenceTransformer."
        ) from exc
    return cast(_SentenceTransformerFactory, factory)


@real
class EmbeddingGemmaTextExtractor(BaseFeatureExtractor):
    """Google EmbeddingGemma sentence embedding extractor."""

    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        output_dim: int = 768,
        batch_size: int = 32,
        normalize: bool = True,
        prompt_name: str | None = "classification",
        device: str | None = None,
    ) -> None:
        if output_dim not in _SUPPORTED_OUTPUT_DIMS:
            allowed = ", ".join(str(dim) for dim in sorted(_SUPPORTED_OUTPUT_DIMS))
            raise ValueError(f"output_dim 은 {allowed} 중 하나여야 합니다")
        if batch_size <= 0:
            raise ValueError("batch_size 는 양수여야 합니다")
        self._model_name = model_name
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._normalize = normalize
        self._prompt_name = prompt_name
        self._device = device
        self._model: _SentenceTransformerModel | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"embeddinggemma_{i}" for i in range(self._output_dim))

    def transform(self, samples: Sequence[RawSample]) -> FeatureMatrix:
        if not samples:
            return self._stack_rows((), self.names)

        texts = [sample.text for sample in samples]
        encode_kwargs: dict[str, object] = {
            "batch_size": self._batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self._normalize,
            "show_progress_bar": False,
        }
        if self._prompt_name is not None:
            encode_kwargs["prompt_name"] = self._prompt_name

        encoded = self._model_instance().encode(texts, **encode_kwargs)
        values = self._coerce_embeddings(encoded, len(samples))
        return self._matrix(values, self.names)

    def _model_instance(self) -> _SentenceTransformerModel:
        if self._model is None:
            factory = _load_sentence_transformer_class()
            kwargs: dict[str, object] = {}
            if self._device is not None:
                kwargs["device"] = self._device
            try:
                self._model = factory(self._model_name, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "EmbeddingGemma 모델을 불러오지 못했습니다. "
                    "Hugging Face에서 google/embeddinggemma-300m 라이선스에 동의했고, "
                    "`uv sync --extra text` 로 호환 의존성을 설치했는지 확인하세요."
                ) from exc
        return self._model

    def _coerce_embeddings(self, encoded: object, n_samples: int) -> np.ndarray:
        values = np.asarray(encoded, dtype=np.float64)
        if values.ndim == 1 and n_samples == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(
                f"EmbeddingGemma 출력은 2차원이어야 합니다 (got ndim={values.ndim})"
            )
        if values.shape[0] != n_samples:
            raise ValueError(
                "EmbeddingGemma 출력 행 수가 sample 수와 일치하지 않습니다: "
                f"{values.shape[0]} != {n_samples}"
            )
        if values.shape[1] < self._output_dim:
            raise ValueError(
                "EmbeddingGemma 출력 차원이 설정값보다 작습니다: "
                f"{values.shape[1]} < {self._output_dim}"
            )
        if values.shape[1] > self._output_dim:
            values = values[:, : self._output_dim]
            if self._normalize:
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        return np.asarray(values, dtype=np.float64)
