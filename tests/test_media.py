"""Raw media loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.types import Modality, Split
from meld_emotion.data.media import MediaLoader
from meld_emotion.features.audio import AudioConceptExtractor
from meld_emotion.features.text import TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline


def _media_sample(
    path: Path,
    waveform: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    uid: str = "v",
) -> RawSample:
    return RawSample(
        uid=uid,
        dialogue_id=1,
        utterance_id=2,
        text="hello",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.TEXT, Modality.AUDIO, Modality.VIDEO),
        audio=AudioInput(sample_rate=16000, waveform=waveform, source_path=path),
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


def _write_wav(path: Path, sample_rate: int = 8000) -> None:
    soundfile = pytest.importorskip("soundfile")
    t = np.linspace(0.0, 0.1, num=sample_rate // 10, endpoint=False)
    waveform = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    soundfile.write(path, waveform, sample_rate)


def _write_mp4_with_audio(
    path: Path, sample_rate: int = 16000, freq: float = 440.0, seconds: float = 0.5
) -> None:
    """av 로 AAC 오디오 트랙을 가진 mp4 를 인코딩(MELD 컨테이너 구조를 모사). 시스템 ffmpeg 불필요."""
    av = pytest.importorskip("av")
    frame_size = 1024  # AAC 인코더 프레임 크기
    total = int(sample_rate * seconds)
    total += (-total) % frame_size  # frame_size 배수로 패딩
    t = np.arange(total, dtype=np.float32) / sample_rate
    mono = (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        pts = 0
        for start in range(0, mono.size, frame_size):
            chunk = mono[start : start + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="fltp", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = pts
            pts += chunk.size
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):  # flush
            container.mux(packet)


def test_load_audio_returns_mono_float64_waveform(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "clip.wav"
    _write_wav(path)

    loaded = MediaLoader(audio_sample_rate=16000).load_audio(
        AudioInput(sample_rate=8000, source_path=path)
    )

    assert loaded.waveform is not None
    assert loaded.sample_rate == 16000
    assert loaded.waveform.ndim == 1
    assert loaded.waveform.dtype == np.float64
    assert loaded.waveform.size > 0


def test_load_audio_decodes_mp4_aac_track(tmp_path: Path) -> None:
    """MELD 처럼 AAC 오디오를 담은 mp4 를 시스템 ffmpeg 없이 av 로 디코딩한다."""
    pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    _write_mp4_with_audio(path)

    loaded = MediaLoader(audio_sample_rate=16000).load_audio(
        AudioInput(sample_rate=16000, source_path=path)
    )

    assert loaded.waveform is not None
    assert loaded.sample_rate == 16000
    assert loaded.waveform.ndim == 1
    assert loaded.waveform.dtype == np.float64
    assert loaded.waveform.size > 0


def test_load_audio_uses_relative_segment_bounds(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    _write_mp4_with_audio(path, seconds=2.0)

    loaded = MediaLoader(audio_sample_rate=16000).load_audio(
        AudioInput(
            sample_rate=16000,
            source_path=path,
            segment_start=0.25,
            segment_end=0.75,
        )
    )

    assert loaded.waveform is not None
    assert loaded.waveform.size == 8000


def test_load_audio_keeps_pretrimmed_absolute_meld_segment(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    _write_mp4_with_audio(path, seconds=0.5)

    loaded = MediaLoader(audio_sample_rate=16000).load_audio(
        AudioInput(
            sample_rate=16000,
            source_path=path,
            segment_start=811.060,
            segment_end=811.560,
        )
    )

    assert loaded.waveform is not None
    assert 0.45 <= loaded.waveform.size / 16000 <= 0.65


def test_load_audio_rejects_overlong_source_media(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "long.mp4"
    _write_mp4_with_audio(path, seconds=0.5)

    with pytest.raises(ValueError, match="허용 한도"):
        MediaLoader(audio_sample_rate=16000, max_audio_seconds=0.1).load_audio(
            AudioInput(sample_rate=16000, source_path=path)
        )


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
        MediaLoader(audio_sample_rate=0)
    with pytest.raises(ValueError):
        MediaLoader(video_frame_size=(0, 64))
    with pytest.raises(ValueError):
        MediaLoader(video_max_frames=0)
    with pytest.raises(ValueError):
        MediaLoader().load_audio(AudioInput(sample_rate=16000))
    with pytest.raises(FileNotFoundError):
        MediaLoader().load_audio(AudioInput(sample_rate=16000, source_path=tmp_path / "missing.wav"))
    with pytest.raises(ValueError):
        MediaLoader().load_video(VideoInput(fps=25.0))
    with pytest.raises(FileNotFoundError):
        MediaLoader().load_video(VideoInput(fps=25.0, source_path=tmp_path / "missing.mp4"))


class _FakeMediaLoader:
    def __init__(self) -> None:
        self.audio_calls = 0
        self.video_calls = 0

    def load_audio(self, audio: AudioInput) -> AudioInput:
        self.audio_calls += 1
        waveform = np.ones(8, dtype=np.float64)
        return replace(audio, waveform=waveform)

    def load_video(self, video: VideoInput) -> VideoInput:
        self.video_calls += 1
        frames = np.ones((2, 4, 4, 3), dtype=np.float64)
        return replace(video, frames=frames)


def test_feature_pipeline_lazy_loads_before_audio_extractors(tmp_path: Path) -> None:
    loader = _FakeMediaLoader()
    pipeline = FeaturePipeline([AudioConceptExtractor()], media_loader=loader)

    bundle = pipeline.fit_transform([_media_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.audio_calls == 1
    assert loader.video_calls == 0
    assert bundle.matrices[0].values.shape == (1, 6)


def test_feature_pipeline_lazy_loads_before_video_extractors(tmp_path: Path) -> None:
    loader = _FakeMediaLoader()
    pipeline = FeaturePipeline([VideoConceptExtractor()], media_loader=loader)

    bundle = pipeline.fit_transform([_media_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.audio_calls == 0
    assert loader.video_calls == 1
    assert bundle.matrices[0].values.shape == (1, 5)


def test_feature_pipeline_skips_loader_without_media_extractors(tmp_path: Path) -> None:
    loader = _FakeMediaLoader()
    pipeline = FeaturePipeline([TextConceptExtractor()], media_loader=loader)

    pipeline.fit_transform([_media_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.audio_calls == 0
    assert loader.video_calls == 0


def test_feature_pipeline_does_not_reload_loaded_media(tmp_path: Path) -> None:
    loader = _FakeMediaLoader()
    waveform = np.ones(8, dtype=np.float64)
    frames = np.ones((2, 4, 4, 3), dtype=np.float64)
    pipeline = FeaturePipeline(
        [AudioConceptExtractor(), VideoConceptExtractor()],
        media_loader=loader,
    )

    pipeline.fit_transform(
        [_media_sample(tmp_path / "clip.mp4", waveform=waveform, frames=frames)],
        Split.TRAIN,
    )

    assert loader.audio_calls == 0
    assert loader.video_calls == 0


def test_feature_pipeline_loads_audio_and_video_when_both_extractors_need_them(
    tmp_path: Path,
) -> None:
    loader = _FakeMediaLoader()
    pipeline = FeaturePipeline(
        [AudioConceptExtractor(), VideoConceptExtractor()],
        media_loader=loader,
    )

    bundle = pipeline.fit_transform([_media_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.audio_calls == 1
    assert loader.video_calls == 1
    assert bundle.matrices[0].values.shape == (1, 6)
    assert bundle.matrices[1].values.shape == (1, 5)


class _FailingAudioLoader(_FakeMediaLoader):
    def load_audio(self, audio: AudioInput) -> AudioInput:
        self.audio_calls += 1
        raise ValueError("broken audio")


def test_feature_pipeline_can_drop_failed_audio_modality(tmp_path: Path) -> None:
    loader = _FailingAudioLoader()
    pipeline = FeaturePipeline(
        [AudioConceptExtractor()],
        media_loader=loader,
        media_error_policy="drop_modality",
    )

    with pytest.warns(RuntimeWarning, match="누락 처리"):
        bundle = pipeline.fit_transform([_media_sample(tmp_path / "clip.mp4")], Split.TRAIN)

    assert loader.audio_calls == 1
    assert bundle.availability[Modality.AUDIO].tolist() == [False]
    assert bundle.matrices[0].values.shape == (1, 6)
    assert np.allclose(bundle.matrices[0].values, 0.0)


def test_feature_pipeline_can_drop_failed_media_sample(tmp_path: Path) -> None:
    loader = _FailingAudioLoader()
    pipeline = FeaturePipeline(
        [AudioConceptExtractor()],
        media_loader=loader,
        media_error_policy="drop_sample",
    )
    broken = _media_sample(tmp_path / "broken.mp4")
    good = _media_sample(tmp_path / "good.mp4", waveform=np.ones(8, dtype=np.float64))

    with pytest.warns(RuntimeWarning, match="샘플 전체를 제외"):
        bundle = pipeline.fit_transform([broken, good], Split.TRAIN)

    assert loader.audio_calls == 1
    assert bundle.uids == ("v",)
    assert bundle.n_samples == 1
    assert bundle.availability[Modality.AUDIO].tolist() == [True]


def test_feature_pipeline_drops_overlong_audio_sample_and_text_features(
    tmp_path: Path,
) -> None:
    pytest.importorskip("av")
    long_path = tmp_path / "long.mp4"
    good_path = tmp_path / "good.mp4"
    _write_mp4_with_audio(long_path, seconds=0.5)
    _write_mp4_with_audio(good_path, seconds=0.05)
    pipeline = FeaturePipeline(
        [TextConceptExtractor(), AudioConceptExtractor()],
        media_loader=MediaLoader(audio_sample_rate=16000, max_audio_seconds=0.1),
        media_error_policy="drop_sample",
    )

    with pytest.warns(RuntimeWarning, match="샘플 전체를 제외"):
        bundle = pipeline.fit_transform(
            [
                _media_sample(long_path, uid="long"),
                _media_sample(good_path, uid="good"),
            ],
            Split.TRAIN,
        )

    assert bundle.uids == ("good",)
    assert bundle.n_samples == 1
    assert bundle.matrices[0].modality == Modality.TEXT
    assert bundle.matrices[0].values.shape[0] == 1
    assert bundle.matrices[1].modality == Modality.AUDIO
    assert bundle.matrices[1].values.shape[0] == 1
