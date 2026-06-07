"""Raw media 적재 경계.

MP4 파일에서 필요한 스트림만 lazy-load 하는 얇은 경계이다. 오디오 로딩은 비디오 프레임을
읽지 않고 waveform 만 적재하며, 비디오 로딩은 오디오를 추출하지 않고 프레임만 적재한다.
두 경로 모두 ffmpeg 를 휠에 번들한 PyAV(``av``) 로 **in-process** 디코딩한다.
시스템 ffmpeg·subprocess 가 필요 없다. 무거운 의존성은
실제 로딩 시점에만 import 해서, 텍스트/합성 데이터 실험은 기본 환경에서도 그대로 동작한다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from meld_emotion.core.data import AudioInput, VideoInput
from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray


@real
class MediaLoader:
    """오디오/비디오 파일을 배열로 적재하는 경계.

    오디오와 비디오 모두 MP4 컨테이너에서 필요한 스트림만 읽는다. 로딩 결과는
    :class:`AudioInput`/ :class:`VideoInput` 에 채워져 특징 추출기에서 사용된다.
    """

    def __init__(
        self,
        audio_sample_rate: int = 16000,
        video_max_frames: int = 32,
        video_frame_size: tuple[int, int] = (64, 64),
        max_audio_seconds: float | None = None,
        min_audio_seconds: float | None = None,
    ) -> None:
        if audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate 는 양의 정수여야 합니다")
        if video_max_frames <= 0:
            raise ValueError("video_max_frames 는 양의 정수여야 합니다")
        if max_audio_seconds is not None and max_audio_seconds <= 0.0:
            raise ValueError("max_audio_seconds 는 양수이거나 None 이어야 합니다")
        if min_audio_seconds is not None and min_audio_seconds <= 0.0:
            raise ValueError("min_audio_seconds 는 양수이거나 None 이어야 합니다")
        if (
            max_audio_seconds is not None
            and min_audio_seconds is not None
            and min_audio_seconds > max_audio_seconds
        ):
            raise ValueError("min_audio_seconds 는 max_audio_seconds 보다 클 수 없습니다")
        self._audio_sample_rate = audio_sample_rate
        self._video_max_frames = video_max_frames
        self._video_frame_size = _validate_frame_size(video_frame_size)
        self._max_audio_seconds = max_audio_seconds
        self._min_audio_seconds = min_audio_seconds

    def load_audio(self, audio: AudioInput) -> AudioInput:
        if audio.waveform is not None:
            return audio
        if audio.source_path is None:
            raise ValueError("AudioInput.source_path 가 없어 오디오를 적재할 수 없습니다")

        path = audio.source_path
        if not path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {path}")

        av = _require_av()
        duration = _audio_duration_seconds(av, str(path))
        if self._max_audio_seconds is not None and duration > self._max_audio_seconds:
            raise ValueError(
                "오디오 파일 길이가 허용 한도를 초과합니다: "
                f"{path} ({duration:.3f}s > {self._max_audio_seconds:.3f}s)"
            )
        try:
            wave = _decode_audio(av, str(path), self._audio_sample_rate)
        except Exception as exc:
            raise ValueError(f"오디오 파일을 읽을 수 없습니다: {path}") from exc
        wave = _select_segment(
            wave,
            self._audio_sample_rate,
            audio.segment_start,
            audio.segment_end,
        )

        if wave.size == 0:
            raise ValueError(f"오디오 waveform 이 비어 있습니다: {path}")
        selected_duration = wave.size / self._audio_sample_rate
        if self._min_audio_seconds is not None and selected_duration < self._min_audio_seconds:
            raise ValueError(
                "오디오 구간 길이가 허용 하한보다 짧습니다: "
                f"{path} ({selected_duration:.6f}s < {self._min_audio_seconds:.6f}s)"
            )
        return replace(audio, sample_rate=self._audio_sample_rate, waveform=wave)

    def load_video(self, video: VideoInput) -> VideoInput:
        if video.frames is not None:
            return video
        if video.source_path is None:
            raise ValueError("VideoInput.source_path 가 없어 비디오를 적재할 수 없습니다")

        path = video.source_path
        if not path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {path}")

        try:
            frames, fps = self._decode_video(_require_av(), str(path), video.fps)
        except Exception as exc:
            raise ValueError(f"비디오 파일을 읽을 수 없습니다: {path}") from exc

        if not frames:
            raise ValueError(f"비디오 프레임을 읽지 못했습니다: {path}")
        return replace(video, frames=np.stack(frames, axis=0), fps=fps)

    def _decode_video(
        self, av: Any, path: str, fallback_fps: float
    ) -> tuple[list[FloatArray], float]:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError("비디오 스트림이 없습니다")
            stream = container.streams.video[0]
            fps = _video_fps(stream, fallback_fps)
            total = int(getattr(stream, "frames", 0) or 0)
            indices = _sample_indices(total, self._video_max_frames)
            frames = (
                self._read_indexed_video_frames(container, stream, indices)
                if indices is not None
                else self._read_stream_video_frames(container, stream)
            )
        return frames, fps

    def _read_indexed_video_frames(
        self,
        container: Any,
        stream: Any,
        indices: tuple[int, ...],
    ) -> list[FloatArray]:
        targets = set(indices)
        frames: list[FloatArray] = []
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in targets:
                continue
            frames.append(self._convert_frame(frame))
            if len(frames) == len(targets):
                break
        return frames

    def _read_stream_video_frames(self, container: Any, stream: Any) -> list[FloatArray]:
        frames: list[FloatArray] = []
        for frame in container.decode(stream):
            frames.append(self._convert_frame(frame))
            if len(frames) >= self._video_max_frames:
                break
        return frames

    def _convert_frame(self, frame: Any) -> FloatArray:
        height, width = self._video_frame_size
        rgb = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
        return np.asarray(rgb, dtype=np.float64) / 255.0


def _validate_frame_size(frame_size: tuple[int, int]) -> tuple[int, int]:
    if len(frame_size) != 2:
        raise ValueError("video_frame_size 는 (height, width) 두 값이어야 합니다")
    height, width = frame_size
    if height <= 0 or width <= 0:
        raise ValueError("video_frame_size 값은 모두 양의 정수여야 합니다")
    return (height, width)


def _sample_indices(total_frames: int, max_frames: int) -> tuple[int, ...] | None:
    if total_frames <= 0:
        return None
    count = min(total_frames, max_frames)
    return tuple(int(i) for i in np.linspace(0, total_frames - 1, num=count, dtype=np.int64))


def _video_fps(stream: Any, fallback: float) -> float:
    for attr in ("average_rate", "base_rate"):
        value = getattr(stream, attr, None)
        if value is not None:
            fps = float(value)
            if fps > 0:
                return fps
    return fallback


def _decode_audio(av: Any, path: str, target_sr: int) -> FloatArray:
    """av 로 오디오 스트림만 in-process 디코딩 → mono float64 waveform.

    ``AudioResampler`` 로 단일 채널·목표 샘플레이트로 맞추고, packed float(``flt``)로 받아
    1차원 배열로 이어 붙인다. 마지막에 리샘플러를 flush 해 버퍼 잔여 샘플까지 회수한다.
    """
    chunks: list[FloatArray] = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("오디오 스트림이 없습니다")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(np.asarray(out.to_ndarray(), dtype=np.float64).reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(np.asarray(out.to_ndarray(), dtype=np.float64).reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def _audio_duration_seconds(av: Any, path: str) -> float:
    with av.open(path) as container:
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0
        if not container.streams.audio:
            raise ValueError("오디오 스트림이 없습니다")
        stream = container.streams.audio[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
    return 0.0


def _select_segment(
    wave: FloatArray,
    sample_rate: int,
    start: float | None,
    end: float | None,
) -> FloatArray:
    if start is None or end is None:
        return wave
    if end <= start:
        raise ValueError(f"오디오 구간 end 는 start 보다 커야 합니다: {start} >= {end}")

    duration = wave.size / sample_rate
    segment_duration = end - start
    tolerance = max(0.25, min(1.0, segment_duration * 0.05))

    if end <= duration + tolerance:
        start_index = max(0, round(start * sample_rate))
        end_index = min(wave.size, round(end * sample_rate))
        return wave[start_index:end_index]

    if abs(duration - segment_duration) <= tolerance:
        return wave

    segment_samples = round(segment_duration * sample_rate)
    if 0 < segment_samples <= wave.size:
        return wave[:segment_samples]
    return wave


def _require_av() -> Any:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "PyAV 가 필요합니다. `uv sync --extra audio` (또는 --extra all) 로 설치하세요."
        ) from exc
    return av
