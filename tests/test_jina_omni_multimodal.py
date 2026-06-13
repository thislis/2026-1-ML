"""Jina Omni fused multimodal extractor and scenario plumbing tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from meld_emotion.config.loader import from_dict, to_dict
from meld_emotion.config.schema import ExperimentConfig, JinaOmniMultimodalConfig
from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.types import FeatureKind, Modality, Split
from meld_emotion.features.base import BaseFeatureExtractor
from meld_emotion.features.multimodal import JinaOmniMultimodalExtractor
from meld_emotion.features.multimodal import jina_omni as jina_module
from meld_emotion.fusion.masking import get_scenario
from meld_emotion.pipeline.builder import build_extractor
from meld_emotion.pipeline.cache import InMemoryFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline
from meld_emotion.pipeline.runner import _augment_samples_for_scenarios, _samples_for_scenario


def _sample(uid: str = "x", mask: ModalityMask | None = None) -> RawSample:
    return RawSample(
        uid=uid,
        dialogue_id=1,
        utterance_id=2,
        text="hello world",
        speaker="s",
        split=Split.TRAIN,
        mask=mask if mask is not None else ModalityMask.full(),
        audio=AudioInput(sample_rate=16000, source_path=Path("audio.wav")),
        video=VideoInput(fps=25.0, source_path=Path("video.mp4")),
    )


class _FakeSentenceTransformer:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[tuple[list[object], dict[str, object]]] = []

    def encode(self, sentences, **kwargs):
        values = []
        inputs = list(sentences)
        self.calls.append((inputs, kwargs))
        for item in inputs:
            width = len(item) if isinstance(item, tuple) else 1
            values.append(np.full(self.dim, float(width), dtype=np.float64))
        return np.vstack(values)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def empty_cache() -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()


def test_jina_omni_transform_uses_sentence_transformer_and_masks(monkeypatch) -> None:
    fake = _FakeSentenceTransformer()
    created: list[tuple[str, dict[str, object]]] = []

    def factory(model_name: str, **kwargs):
        created.append((model_name, kwargs))
        return fake

    monkeypatch.setattr(jina_module, "_load_sentence_transformer_class", lambda: factory)
    monkeypatch.setattr(jina_module, "_load_torch_module", lambda: _FakeTorch)
    monkeypatch.setattr(jina_module, "_require_peft", lambda: None)
    monkeypatch.setattr(jina_module, "_require_media_processor_dependencies", lambda: None)

    extractor = JinaOmniMultimodalExtractor(output_dim=128, batch_size=2, device="gpu")
    matrix = extractor.transform(
        [
            _sample("full"),
            _sample("no_audio", ModalityMask.of(Modality.TEXT, Modality.VIDEO)),
        ]
    )

    assert created == [
        (
            "jinaai/jina-embeddings-v5-omni-small",
            {
                "trust_remote_code": True,
                "device": "cuda",
                "model_kwargs": {"default_task": "classification", "modality": "omni"},
            },
        )
    ]
    assert fake.calls[0][1]["truncate_dim"] == 128
    assert isinstance(fake.calls[0][0][0], tuple)
    assert len(fake.calls[0][0][0]) == 3
    assert isinstance(fake.calls[0][0][1], tuple)
    assert len(fake.calls[0][0][1]) == 2
    assert matrix.values.shape == (2, 128)
    assert matrix.names == tuple(f"jina_omni_{i}" for i in range(128))
    assert matrix.modality == Modality.MULTIMODAL
    assert matrix.kind == FeatureKind.EMBEDDING


def test_jina_omni_rejects_invalid_device() -> None:
    try:
        JinaOmniMultimodalExtractor(device="tpu")
    except ValueError as exc:
        assert "device" in str(exc)
    else:
        raise AssertionError("invalid device should fail")


def test_jina_omni_caps_loaded_video_frames() -> None:
    frames = np.zeros((20, 4, 4, 3), dtype=np.float64)
    sample = RawSample(
        uid="video",
        dialogue_id=0,
        utterance_id=0,
        text="hello",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.full(),
        audio=None,
        video=VideoInput(fps=25.0, frames=frames),
    )
    video = JinaOmniMultimodalExtractor(max_video_frames=5)._video_input(sample)

    assert isinstance(video, np.ndarray)
    assert video.shape == (5, 4, 4, 3)
    assert video.dtype == np.uint8


def test_jina_omni_config_roundtrip_and_builder() -> None:
    config = ExperimentConfig(
        extractors=(
            JinaOmniMultimodalConfig(
                output_dim=512,
                batch_size=3,
                device="mps",
                max_video_frames=6,
            ),
        )
    )
    assert from_dict(to_dict(config)) == config

    extractor_config = from_dict(
        {"extractors": [{"type": "jina_omni_multimodal", "output_dim": 256}]}
    ).extractors[0]
    extractor = build_extractor(extractor_config)
    assert isinstance(extractor, JinaOmniMultimodalExtractor)


class _MaskCountingExtractor(BaseFeatureExtractor):
    modality = Modality.MULTIMODAL
    kind = FeatureKind.EMBEDDING
    feature_names = ("available_count",)

    def transform(self, samples):
        rows = [
            np.array(
                [
                    float(
                        sample.has(Modality.TEXT)
                        + sample.has(Modality.AUDIO)
                        + sample.has(Modality.VIDEO)
                    )
                ],
                dtype=np.float64,
            )
            for sample in samples
        ]
        return self._stack_rows(rows, self.feature_names)


def test_feature_cache_separates_same_uid_with_different_masks() -> None:
    cache = InMemoryFeatureCache()
    pipeline = FeaturePipeline([_MaskCountingExtractor()], cache=cache)
    sample = _sample("same")
    full = pipeline.transform([sample], Split.TEST)
    no_audio = pipeline.transform(
        _samples_for_scenario([sample], get_scenario("no_audio")),
        Split.TEST,
    )

    assert full.matrices[0].values.tolist() == [[3.0]]
    assert no_audio.matrices[0].values.tolist() == [[2.0]]


def test_training_augmentation_rewrites_uid_and_dialogue_id() -> None:
    samples = [_sample("a"), _sample("b")]
    augmented = _augment_samples_for_scenarios(samples, ("no_audio", "no_video"))

    assert len(augmented) == 6
    assert augmented[2].uid == "no_audio:a"
    assert augmented[2].dialogue_id != samples[0].dialogue_id
    assert not augmented[2].has(Modality.AUDIO)
    assert augmented[4].uid == "no_video:a"
    assert not augmented[4].has(Modality.VIDEO)
