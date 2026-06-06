"""EmbeddingGemma 텍스트 임베딩 extractor 계약 검증."""

from __future__ import annotations

import numpy as np

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import EmbeddingGemmaTextConfig, ExperimentConfig
from meld_emotion.core.data import ModalityMask, RawSample
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.text import EmbeddingGemmaTextExtractor
from meld_emotion.features.text import embeddinggemma as embeddinggemma_module
from meld_emotion.pipeline.builder import build_extractor


def _sample(text: str, uid: str = "x") -> RawSample:
    return RawSample(
        uid=uid,
        dialogue_id=0,
        utterance_id=0,
        text=text,
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.TEXT),
    )


class _FakeSentenceTransformer:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, sentences, **kwargs):
        texts = list(sentences)
        self.calls.append((texts, kwargs))
        values = np.arange(len(texts) * self.dim, dtype=np.float64).reshape(len(texts), self.dim)
        return values + 1.0


def test_embeddinggemma_transform_uses_sentence_transformer(monkeypatch) -> None:
    fake = _FakeSentenceTransformer()
    created: list[tuple[str, dict[str, object]]] = []

    def factory(model_name: str, **kwargs):
        created.append((model_name, kwargs))
        return fake

    monkeypatch.setattr(embeddinggemma_module, "_load_sentence_transformer_class", lambda: factory)

    extractor = EmbeddingGemmaTextExtractor(output_dim=128, batch_size=4, device="cpu")
    matrix = extractor.transform([_sample("hello", "a"), _sample("world", "b")])

    assert created == [("google/embeddinggemma-300m", {"device": "cpu"})]
    assert fake.calls[0][0] == ["hello", "world"]
    assert fake.calls[0][1]["batch_size"] == 4
    assert fake.calls[0][1]["prompt_name"] == "classification"
    assert fake.calls[0][1]["normalize_embeddings"] is True
    assert matrix.values.shape == (2, 128)
    assert matrix.names == tuple(f"embeddinggemma_{i}" for i in range(128))
    assert matrix.modality == Modality.TEXT
    assert matrix.kind == FeatureKind.EMBEDDING
    assert matrix.source == "EmbeddingGemmaTextExtractor"
    assert np.allclose(np.linalg.norm(matrix.values, axis=1), 1.0)


def test_embeddinggemma_empty_transform_does_not_load_model(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("empty transform should not load the model")

    monkeypatch.setattr(embeddinggemma_module, "_load_sentence_transformer_class", fail_load)

    matrix = EmbeddingGemmaTextExtractor(output_dim=256).transform([])

    assert matrix.values.shape == (0, 256)
    assert matrix.names == tuple(f"embeddinggemma_{i}" for i in range(256))


def test_embeddinggemma_config_roundtrip_and_builder() -> None:
    config = ExperimentConfig(
        extractors=(
            EmbeddingGemmaTextConfig(
                output_dim=512,
                batch_size=8,
                normalize=False,
                prompt_name=None,
                device="mps",
            ),
        )
    )
    assert from_dict(to_dict(config)) == config

    extractor_config = from_dict(
        {"extractors": [{"type": "text_embeddinggemma", "output_dim": 128}]}
    ).extractors[0]
    extractor = build_extractor(extractor_config)
    assert isinstance(extractor, EmbeddingGemmaTextExtractor)
