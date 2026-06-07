"""TimeSformer 비디오 임베딩 extractor 계약 검증."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import ExperimentConfig, TimeSformerVideoConfig
from meld_emotion.core.data import ModalityMask, RawSample, VideoInput
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.video import TimeSformerVideoExtractor
from meld_emotion.features.video import timesformer as timesformer_module
from meld_emotion.pipeline.builder import build_extractor


def _sample(frames: np.ndarray | None, uid: str = "x") -> RawSample:
    video = VideoInput(fps=25.0, frames=frames) if frames is not None else None
    return RawSample(
        uid=uid,
        dialogue_id=0,
        utterance_id=0,
        text="",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.VIDEO),
        video=video,
    )


class _FakeModel:
    created: ClassVar[list[str]] = []
    seen_shapes: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def from_pretrained(cls, model_name: str) -> _FakeModel:
        cls.created.append(model_name)
        return cls()

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def __call__(self, *, pixel_values):
        torch = pytest.importorskip("torch")
        _FakeModel.seen_shapes.append(tuple(int(dim) for dim in pixel_values.shape))
        n = int(pixel_values.shape[0])
        values = torch.arange(n * 3 * 8, dtype=torch.float32).reshape(n, 3, 8) + 1.0
        return SimpleNamespace(last_hidden_state=values)


def test_timesformer_transform_uses_model_and_cls_embedding(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _FakeModel.created = []
    _FakeModel.seen_shapes = []
    monkeypatch.setattr(timesformer_module, "_load_timesformer_model_class", lambda: _FakeModel)
    monkeypatch.setattr(timesformer_module, "_load_torch_module", lambda: torch)

    frames = np.arange(4 * 2 * 2 * 3, dtype=np.float64).reshape(4, 2, 2, 3)
    extractor = TimeSformerVideoExtractor(output_dim=4, batch_size=2, num_frames=2, frame_size=2)
    matrix = extractor.transform([_sample(frames, "a"), _sample(None, "missing")])

    assert _FakeModel.created == ["facebook/timesformer-base-finetuned-k400"]
    assert _FakeModel.seen_shapes == [(1, 2, 3, 2, 2)]
    assert matrix.values.shape == (2, 4)
    assert matrix.names == tuple(f"timesformer_{i}" for i in range(4))
    assert matrix.modality == Modality.VIDEO
    assert matrix.kind == FeatureKind.EMBEDDING
    assert matrix.source == "TimeSformerVideoExtractor"
    assert np.isclose(np.linalg.norm(matrix.values[0]), 1.0)
    assert np.allclose(matrix.values[1], 0.0)


def test_timesformer_mean_pooling(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(timesformer_module, "_load_timesformer_model_class", lambda: _FakeModel)
    monkeypatch.setattr(timesformer_module, "_load_torch_module", lambda: torch)

    frames = np.ones((2, 2, 2, 3), dtype=np.float64)
    extractor = TimeSformerVideoExtractor(
        output_dim=8,
        num_frames=2,
        frame_size=2,
        normalize=False,
        pooling="mean",
    )
    matrix = extractor.transform([_sample(frames)])

    hidden = np.arange(3 * 8, dtype=np.float64).reshape(3, 8) + 1.0
    assert np.allclose(matrix.values[0], hidden.mean(axis=0))


def test_timesformer_empty_transform_does_not_load_model(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("empty transform should not load the model")

    monkeypatch.setattr(timesformer_module, "_load_timesformer_model_class", fail_load)

    matrix = TimeSformerVideoExtractor(output_dim=8).transform([])

    assert matrix.values.shape == (0, 8)
    assert matrix.names == tuple(f"timesformer_{i}" for i in range(8))


def test_timesformer_rejects_invalid_pooling() -> None:
    with pytest.raises(ValueError, match="pooling"):
        TimeSformerVideoExtractor(pooling="bad")


def test_timesformer_config_roundtrip_and_builder() -> None:
    config = ExperimentConfig(
        extractors=(
            TimeSformerVideoConfig(
                output_dim=256,
                batch_size=1,
                num_frames=8,
                frame_size=224,
                normalize=False,
                pooling="mean",
                device="mps",
            ),
        )
    )
    assert from_dict(to_dict(config)) == config

    extractor_config = from_dict(
        {"extractors": [{"type": "video_timesformer", "output_dim": 128}]}
    ).extractors[0]
    extractor = build_extractor(extractor_config)
    assert isinstance(extractor, TimeSformerVideoExtractor)
