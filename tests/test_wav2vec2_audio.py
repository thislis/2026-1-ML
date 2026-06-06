"""Wav2Vec2 XLS-R 오디오 임베딩 extractor 계약 검증."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import ExperimentConfig, Wav2Vec2XlsrAudioConfig
from meld_emotion.core.data import AudioInput, ModalityMask, RawSample
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.audio import Wav2Vec2XlsrAudioExtractor
from meld_emotion.features.audio import wav2vec2 as wav2vec2_module
from meld_emotion.pipeline.builder import build_extractor


def _sample(waveform: np.ndarray | None, uid: str = "x", sample_rate: int = 16000) -> RawSample:
    audio = AudioInput(sample_rate=sample_rate, waveform=waveform) if waveform is not None else None
    return RawSample(
        uid=uid,
        dialogue_id=0,
        utterance_id=0,
        text="",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.AUDIO),
        audio=audio,
    )


class _FakeProcessor:
    created: ClassVar[list[str]] = []
    seen_lengths: ClassVar[list[int]] = []

    @classmethod
    def from_pretrained(cls, model_name: str):
        cls.created.append(model_name)
        return cls()

    def __call__(self, waveforms, **kwargs):
        torch = pytest.importorskip("torch")
        waves = list(waveforms)
        self.seen_lengths.extend(len(wave) for wave in waves)
        max_len = max((len(wave) for wave in waves), default=0)
        input_values = torch.zeros((len(waves), max_len), dtype=torch.float32)
        attention_mask = torch.zeros((len(waves), max_len), dtype=torch.long)
        for row, wave in enumerate(waves):
            input_values[row, : len(wave)] = torch.as_tensor(wave, dtype=torch.float32)
            attention_mask[row, : len(wave)] = 1
        self.kwargs = kwargs
        return {"input_values": input_values, "attention_mask": attention_mask}


class _FakeModel:
    created: ClassVar[list[str]] = []

    @classmethod
    def from_pretrained(cls, model_name: str):
        cls.created.append(model_name)
        return cls()

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device: str):
        self.device = device
        return self

    def __call__(self, **inputs):
        torch = pytest.importorskip("torch")
        n = int(inputs["input_values"].shape[0])
        values = torch.arange(n * 2 * 1024, dtype=torch.float32).reshape(n, 2, 1024) + 1.0
        return SimpleNamespace(last_hidden_state=values)

    def _get_feature_vector_attention_mask(self, length: int, attention_mask):
        torch = pytest.importorskip("torch")
        masks = torch.ones((attention_mask.shape[0], length), dtype=torch.long)
        if masks.shape[0] > 1:
            masks[1, 1:] = 0
        return masks


def test_wav2vec2_loads_transformers_lazy_exports(monkeypatch) -> None:
    class _LazyTransformersModule:
        def __getattr__(self, name: str):
            if name == "AutoFeatureExtractor":
                return _FakeProcessor
            if name == "Wav2Vec2Model":
                return _FakeModel
            raise AttributeError(name)

    def fake_import_module(name: str):
        assert name == "transformers"
        return _LazyTransformersModule()

    monkeypatch.setattr(wav2vec2_module, "import_module", fake_import_module)

    assert wav2vec2_module._load_transformers_classes() == (_FakeProcessor, _FakeModel)


def test_wav2vec2_transform_uses_transformers(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _FakeProcessor.created = []
    _FakeProcessor.seen_lengths = []
    _FakeModel.created = []
    monkeypatch.setattr(
        wav2vec2_module,
        "_load_transformers_classes",
        lambda: (_FakeProcessor, _FakeModel),
    )
    monkeypatch.setattr(wav2vec2_module, "_load_torch_module", lambda: torch)

    extractor = Wav2Vec2XlsrAudioExtractor(batch_size=2, device="cpu")
    matrix = extractor.transform(
        [
            _sample(np.ones(4, dtype=np.float64), "a"),
            _sample(np.ones(2, dtype=np.float64), "b"),
            _sample(None, "missing"),
            _sample(np.zeros(0, dtype=np.float64), "empty"),
        ]
    )

    assert _FakeProcessor.created == ["facebook/wav2vec2-xls-r-300m"]
    assert _FakeModel.created == ["facebook/wav2vec2-xls-r-300m"]
    assert matrix.values.shape == (4, 1024)
    assert matrix.names == tuple(f"wav2vec2_xlsr_{i}" for i in range(1024))
    assert matrix.modality == Modality.AUDIO
    assert matrix.kind == FeatureKind.EMBEDDING
    assert matrix.source == "Wav2Vec2XlsrAudioExtractor"
    assert np.allclose(np.linalg.norm(matrix.values[:2], axis=1), 1.0)
    assert np.allclose(matrix.values[2:], 0.0)


def test_wav2vec2_truncates_overlong_waveforms(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _FakeProcessor.created = []
    _FakeProcessor.seen_lengths = []
    _FakeModel.created = []
    monkeypatch.setattr(
        wav2vec2_module,
        "_load_transformers_classes",
        lambda: (_FakeProcessor, _FakeModel),
    )
    monkeypatch.setattr(wav2vec2_module, "_load_torch_module", lambda: torch)

    extractor = Wav2Vec2XlsrAudioExtractor(
        batch_size=1,
        sampling_rate=10,
        max_seconds=0.5,
        device="cpu",
    )
    matrix = extractor.transform([_sample(np.ones(20, dtype=np.float64), "long", sample_rate=10)])

    assert matrix.values.shape == (1, 1024)
    assert _FakeProcessor.seen_lengths == [5]


def test_wav2vec2_chunks_long_waveforms_without_dropping_tail(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _FakeProcessor.created = []
    _FakeProcessor.seen_lengths = []
    _FakeModel.created = []
    monkeypatch.setattr(
        wav2vec2_module,
        "_load_transformers_classes",
        lambda: (_FakeProcessor, _FakeModel),
    )
    monkeypatch.setattr(wav2vec2_module, "_load_torch_module", lambda: torch)

    extractor = Wav2Vec2XlsrAudioExtractor(
        batch_size=2,
        sampling_rate=10,
        chunk_seconds=0.5,
        device="cpu",
    )
    matrix = extractor.transform([_sample(np.ones(12, dtype=np.float64), "long", sample_rate=10)])

    assert matrix.values.shape == (1, 1024)
    assert _FakeProcessor.seen_lengths == [5, 5, 2]


def test_wav2vec2_empty_transform_does_not_load_model(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("empty transform should not load the model")

    monkeypatch.setattr(wav2vec2_module, "_load_transformers_classes", fail_load)

    matrix = Wav2Vec2XlsrAudioExtractor().transform([])

    assert matrix.values.shape == (0, 1024)
    assert matrix.names == tuple(f"wav2vec2_xlsr_{i}" for i in range(1024))


def test_wav2vec2_rejects_wrong_sample_rate() -> None:
    extractor = Wav2Vec2XlsrAudioExtractor()
    with pytest.raises(ValueError, match="sample_rate"):
        extractor.transform([_sample(np.ones(4), sample_rate=8000)])


def test_wav2vec2_config_roundtrip_and_builder() -> None:
    config = ExperimentConfig(
        extractors=(
            Wav2Vec2XlsrAudioConfig(
                output_dim=512,
                batch_size=2,
                sampling_rate=16000,
                max_seconds=30.0,
                chunk_seconds=15.0,
                normalize=False,
                device="mps",
            ),
        )
    )
    assert from_dict(to_dict(config)) == config

    extractor_config = from_dict(
        {"extractors": [{"type": "audio_wav2vec2_xlsr", "output_dim": 1024}]}
    ).extractors[0]
    extractor = build_extractor(extractor_config)
    assert isinstance(extractor, Wav2Vec2XlsrAudioExtractor)
