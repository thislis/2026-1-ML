"""VideoPrism 비디오 임베딩 extractor 계약 검증."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pytest

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import ExperimentConfig, VideoPrismConfig
from meld_emotion.core.data import ModalityMask, RawSample, VideoInput
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.video import VideoPrismVideoExtractor
from meld_emotion.features.video import videoprism as videoprism_module
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


class _FakeJax:
    seen_shapes: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def device_put(cls, value):
        array = np.asarray(value)
        cls.seen_shapes.append(tuple(int(dim) for dim in array.shape))
        return array


class _FakeModel:
    def apply(self, state: dict[str, str], inputs: Any, train: bool = False) -> np.ndarray:
        assert state == {"state": "ok"}
        assert train is False
        array = np.asarray(inputs)
        assert array.shape == (1, 2, 2, 2, 3)
        return np.arange(1, 1 + 1 * 3 * 8, dtype=np.float64).reshape(1, 3, 8)


class _FakeVideoPrism:
    requested_models: ClassVar[list[str]] = []
    requested_weights: ClassVar[list[str]] = []

    @classmethod
    def get_model(cls, model_name: str) -> _FakeModel:
        cls.requested_models.append(model_name)
        return _FakeModel()

    @classmethod
    def load_pretrained_weights(cls, model_name: str) -> dict[str, str]:
        cls.requested_weights.append(model_name)
        return {"state": "ok"}


def test_tensorflow_shim_exposes_variable_for_einops(monkeypatch) -> None:
    tensorflow = ModuleType("tensorflow")
    tensorflow_io = ModuleType("tensorflow.io")
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setitem(sys.modules, "tensorflow.io", tensorflow_io)

    videoprism_module._ensure_tensorflow_gfile_shim()

    assert hasattr(tensorflow, "Tensor")
    assert hasattr(tensorflow, "Variable")
    assert hasattr(tensorflow, "RaggedTensor")
    assert hasattr(tensorflow.io, "gfile")


def test_videoprism_transform_uses_model_and_pools_tokens(monkeypatch) -> None:
    _FakeJax.seen_shapes = []
    _FakeVideoPrism.requested_models = []
    _FakeVideoPrism.requested_weights = []
    monkeypatch.setattr(
        videoprism_module,
        "_load_videoprism_modules",
        lambda: (_FakeJax, _FakeVideoPrism),
    )

    frames = np.arange(4 * 2 * 2 * 3, dtype=np.float64).reshape(4, 2, 2, 3)
    extractor = VideoPrismVideoExtractor(output_dim=4, num_frames=2, frame_size=2)
    matrix = extractor.transform([_sample(frames, "a"), _sample(None, "missing")])

    assert _FakeVideoPrism.requested_models == ["google/videoprism-base-f16r288"]
    assert _FakeVideoPrism.requested_weights == ["google/videoprism-base-f16r288"]
    assert _FakeJax.seen_shapes == [(1, 2, 2, 2, 3)]
    assert matrix.values.shape == (2, 4)
    assert matrix.names == tuple(f"videoprism_{i}" for i in range(4))
    assert matrix.modality == Modality.VIDEO
    assert matrix.kind == FeatureKind.EMBEDDING
    assert matrix.source == "VideoPrismVideoExtractor"
    assert np.isclose(np.linalg.norm(matrix.values[0]), 1.0)
    assert np.allclose(matrix.values[1], 0.0)


def test_videoprism_empty_transform_does_not_load_model(monkeypatch) -> None:
    def fail_load():
        raise AssertionError("empty transform should not load the model")

    monkeypatch.setattr(videoprism_module, "_load_videoprism_modules", fail_load)

    matrix = VideoPrismVideoExtractor(output_dim=8).transform([])

    assert matrix.values.shape == (0, 8)
    assert matrix.names == tuple(f"videoprism_{i}" for i in range(8))


def test_videoprism_fallback_does_not_hide_primary_error(monkeypatch) -> None:
    class BrokenModel:
        def apply(self, state: dict[str, str], inputs: Any, train: bool = False) -> np.ndarray:
            array = np.asarray(inputs)
            if array.ndim == 5:
                raise AttributeError("primary backend error")
            raise ValueError("fallback shape error")

    class BrokenVideoPrism:
        @classmethod
        def get_model(cls, model_name: str) -> BrokenModel:
            return BrokenModel()

        @classmethod
        def load_pretrained_weights(cls, model_name: str) -> dict[str, str]:
            return {"state": "ok"}

    monkeypatch.setattr(
        videoprism_module,
        "_load_videoprism_modules",
        lambda: (_FakeJax, BrokenVideoPrism),
    )

    frames = np.ones((2, 2, 2, 3), dtype=np.float32)
    extractor = VideoPrismVideoExtractor(output_dim=4, num_frames=2, frame_size=2)

    with pytest.raises(AttributeError, match="primary backend error"):
        extractor.transform([_sample(frames)])


def test_videoprism_config_roundtrip_and_builder() -> None:
    config = ExperimentConfig(
        extractors=(
            VideoPrismConfig(
                output_dim=256,
                num_frames=8,
                frame_size=288,
                normalize=False,
                prefer_batched_input=False,
            ),
        )
    )
    assert from_dict(to_dict(config)) == config

    extractor_config = from_dict(
        {"extractors": [{"type": "video_videoprism", "output_dim": 128}]}
    ).extractors[0]
    extractor = build_extractor(extractor_config)
    assert isinstance(extractor, VideoPrismVideoExtractor)
