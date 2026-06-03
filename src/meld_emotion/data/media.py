"""Raw media 적재 경계."""

from __future__ import annotations

from meld_emotion.core.data import AudioInput, VideoInput
from meld_emotion.core.status import raise_unimplemented, unimplemented


@unimplemented("raw MELD 오디오/비디오 디코딩은 baseline 범위 밖입니다.")
class MediaLoader:
    """오디오/비디오 파일을 배열로 적재하는 인터페이스 자리."""

    def load_audio(self, audio: AudioInput) -> AudioInput:
        raise_unimplemented(self)

    def load_video(self, video: VideoInput) -> VideoInput:
        raise_unimplemented(self)
