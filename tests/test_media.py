"""Raw media loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.types import Modality, Split
from meld_emotion.data.media import MediaLoader
from meld_emotion.features.text import TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


def _video_sample(path: Path, frames: np.ndarray | None = None) -> RawSample:
    return RawSample(
        uid="v",
        dialogue_id=1,
        utterance_id=2,
        text="hello",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.TEXT, Modality.VIDEO),
        video=VideoInput(fps=25.0, frames=frames, source_path=path),
    )


def _write_mp4(path: Path, n_frames: int = 10) -> None:
    cv2 = pytest.importorskip("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (20, 12))
    if not bool(writer.isOpened()):
        pytest.skip("OpenCV VideoWriter cannot create mp4 in this environment")
    try:
        for i in range(n_frames):
            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 10
            frame[:, :, 1] = 255 - i * 10
            frame[:, :, 2] = 20
            writer.write(frame)
    finally:
        writer.release()


def test_load_video_defaults_to_64_square_frames(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_mp4(path)

    loaded = MediaLoader().load_video(VideoInput(fps=25.0, source_path=path))

    assert loaded.frames is not None
    assert loaded.frames.shape == (10, 64, 64, 3)
    assert loaded.frames.dtype == np.float64
    assert float(np.min(loaded.frames)) >= 0.0
    assert float(np.max(loaded.frames)) <= 1.0


def test_load_video_respects_custom_frame_size_and_max_frames(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_mp4(path, n_frames=12)

    loaded = MediaLoader(video_max_frames=4, video_frame_size=(32, 48)).load_video(
        VideoInput(fps=25.0, source_path=path)
    )

    assert loaded.frames is not None
    assert loaded.frames.shape == (4, 32, 48, 3)


def test_media_loader_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        MediaLoader(video_frame_size=(0, 64))
    with pytest.raises(ValueError):
        MediaLoader(video_max_frames=0)
    with pytest.raises(ValueError):
        MediaLoader().load_video(VideoInput(fps=25.0))
    with pytest.raises(FileNotFoundError):
        MediaLoader().load_video(VideoInput(fps=25.0, source_path=tmp_path / "missing.mp4"))
    with pytest.raises(NotImplementedError):
        MediaLoader().load_audio(AudioInput(sample_rate=16000))


class _FakeVideoLoader:
    def __init__(self) -> None:
        self.calls = 0

    def load_video(self, video: VideoInput) -> VideoInput:
        self.calls += 1
        frames = np.ones((2, 4, 4, 3), dtype=np.float64)
        return replace(video, frames=frames)


def test_feature_pipeline_lazy_loads_before_video_extractors(tmp_path: Path) -> None:
    loader = _FakeVideoLoader()
    pipeline = FeaturePipeline([VideoConceptExtractor()], media_loader=loader)

    bundle = pipeline.fit_transform([_video_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.calls == 1
    assert bundle.matrices[0].values.shape == (1, 5)


def test_feature_pipeline_skips_loader_without_video_extractors(tmp_path: Path) -> None:
    loader = _FakeVideoLoader()
    pipeline = FeaturePipeline([TextConceptExtractor()], media_loader=loader)

    pipeline.fit_transform([_video_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.calls == 0


def test_feature_pipeline_does_not_reload_loaded_frames(tmp_path: Path) -> None:
    loader = _FakeVideoLoader()
    frames = np.ones((2, 4, 4, 3), dtype=np.float64)
    pipeline = FeaturePipeline([VideoConceptExtractor()], media_loader=loader)

    pipeline.fit_transform([_video_sample(tmp_path / "clip.mp4", frames=frames)], Split.TRAIN)

    assert loader.calls == 0
