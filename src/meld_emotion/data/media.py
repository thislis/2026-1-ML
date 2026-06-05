"""Raw media 적재 경계.

MP4 파일에서 필요한 스트림만 lazy-load 하는 얇은 경계이다. 오디오 로딩은 비디오 프레임을
읽지 않고 waveform 만 적재하며, 비디오 로딩은 오디오를 추출하지 않고 프레임만 적재한다.
무거운 의존성(librosa/OpenCV)은 실제 로딩 시점에만 import 해서, 텍스트/합성 데이터 실험은
기본 환경에서도 그대로 동작하게 둔다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
    ) -> None:
        if audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate 는 양의 정수여야 합니다")
        if video_max_frames <= 0:
            raise ValueError("video_max_frames 는 양의 정수여야 합니다")
        self._audio_sample_rate = audio_sample_rate
        self._video_max_frames = video_max_frames
        self._video_frame_size = _validate_frame_size(video_frame_size)

    def load_audio(self, audio: AudioInput) -> AudioInput:
        if audio.waveform is not None:
            return audio
        if audio.source_path is None:
            raise ValueError("AudioInput.source_path 가 없어 오디오를 적재할 수 없습니다")

        path = audio.source_path
        if not path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {path}")

        librosa = _require_librosa()
        try:
            waveform, sample_rate = librosa.load(
                str(path),
                sr=self._audio_sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise ValueError(f"오디오 파일을 읽을 수 없습니다: {path}") from exc

        wave = np.asarray(waveform, dtype=np.float64).reshape(-1)
        if wave.size == 0:
            raise ValueError(f"오디오 waveform 이 비어 있습니다: {path}")
        return replace(audio, sample_rate=int(sample_rate), waveform=wave)

    def load_video(self, video: VideoInput) -> VideoInput:
        if video.frames is not None:
            return video
        if video.source_path is None:
            raise ValueError("VideoInput.source_path 가 없어 비디오를 적재할 수 없습니다")

        path = video.source_path
        if not path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {path}")

        cv2 = _require_cv2()
        capture = cv2.VideoCapture(str(path))
        try:
            if not bool(capture.isOpened()):
                raise ValueError(f"비디오 파일을 열 수 없습니다: {path}")

            fps = _fps(capture, cv2, video.fps)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = _sample_indices(total, self._video_max_frames)
            frames = (
                self._read_indexed_frames(capture, cv2, indices, path)
                if indices is not None
                else self._read_stream_frames(capture, cv2)
            )
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"비디오 프레임을 읽지 못했습니다: {path}")
        return replace(video, frames=np.stack(frames, axis=0), fps=fps)

    def _read_indexed_frames(
        self,
        capture: Any,
        cv2: Any,
        indices: tuple[int, ...],
        path: Path,
    ) -> list[FloatArray]:
        frames: list[FloatArray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not bool(ok) or frame is None:
                continue
            frames.append(self._convert_frame(frame, cv2))
        if not frames:
            raise ValueError(f"샘플링된 인덱스에서 비디오 프레임을 읽지 못했습니다: {path}")
        return frames

    def _read_stream_frames(self, capture: Any, cv2: Any) -> list[FloatArray]:
        frames: list[FloatArray] = []
        while len(frames) < self._video_max_frames:
            ok, frame = capture.read()
            if not bool(ok) or frame is None:
                break
            frames.append(self._convert_frame(frame, cv2))
        return frames

    def _convert_frame(self, frame: Any, cv2: Any) -> FloatArray:
        height, width = self._video_frame_size
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
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


def _fps(capture: Any, cv2: Any, fallback: float) -> float:
    value = float(capture.get(cv2.CAP_PROP_FPS))
    return value if value > 0 else fallback


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "OpenCV 가 필요합니다. `uv sync --extra video` (또는 --extra all) 로 설치하세요."
        ) from exc
    return cv2


def _require_librosa() -> Any:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "librosa 가 필요합니다. `uv sync --extra audio` (또는 --extra all) 로 설치하세요."
        ) from exc
    return librosa
