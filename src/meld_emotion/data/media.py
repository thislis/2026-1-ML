"""Raw media 적재 경계.

MP4 파일에서 오디오는 건드리지 않고 비디오 프레임만 lazy-load 하는 얇은 경계이다. 무거운
비디오 의존성(OpenCV)은 실제 로딩 시점에만 import 해서, 텍스트/합성 데이터 실험은 기본
환경에서도 그대로 동작하게 둔다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.data import AudioInput, VideoInput
from meld_emotion.core.status import raise_unimplemented, unimplemented
from meld_emotion.core.types import FloatArray


@unimplemented("오디오 디코딩은 미구현입니다. MP4 비디오 프레임 로딩만 지원합니다.")
class MediaLoader:
    """오디오/비디오 파일을 배열로 적재하는 경계.

    현재 구현 범위는 MP4 비디오 프레임 로딩이다. 오디오 디코딩은 실험 오염을 피하기 위해
    명시적으로 지원하지 않는다.
    """

    def __init__(
        self,
        video_max_frames: int = 32,
        video_frame_size: tuple[int, int] = (64, 64),
    ) -> None:
        if video_max_frames <= 0:
            raise ValueError("video_max_frames 는 양의 정수여야 합니다")
        self._video_max_frames = video_max_frames
        self._video_frame_size = _validate_frame_size(video_frame_size)

    def load_audio(self, audio: AudioInput) -> AudioInput:
        raise_unimplemented(self)

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
