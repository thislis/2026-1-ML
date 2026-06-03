"""합성 데이터셋 소스."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.status import real
from meld_emotion.core.types import EMOTION_ORDER, Emotion, Modality, Sentiment, Split

_POSITIVE = {
    Emotion.JOY: "happy wonderful love awesome",
    Emotion.SURPRISE: "wow amazing excited",
    Emotion.NEUTRAL: "okay fine ordinary",
}
_NEGATIVE = {
    Emotion.SADNESS: "sad sorry lonely cry",
    Emotion.ANGER: "angry furious mad hate",
    Emotion.FEAR: "scared afraid terrified nervous",
    Emotion.DISGUST: "disgusting gross awful ugh",
}


@real
class SyntheticDatasetSource:
    """테스트와 예제 실행을 위한 결정적 합성 MELD 형태 데이터."""

    def __init__(
        self,
        n_train: int = 240,
        n_dev: int = 60,
        n_test: int = 60,
        seed: int = 0,
        with_audio: bool = True,
        with_video: bool = True,
        missing_rate: float = 0.0,
    ) -> None:
        if min(n_train, n_dev, n_test) < 0:
            raise ValueError("분할 크기는 음수일 수 없습니다")
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate 는 0..1 범위여야 합니다")
        self._sizes = {Split.TRAIN: n_train, Split.DEV: n_dev, Split.TEST: n_test}
        self._seed = seed
        self._with_audio = with_audio
        self._with_video = with_video
        self._missing_rate = missing_rate

    def load(self, split: Split) -> Iterable[RawSample]:
        split = Split(split)
        rng = np.random.default_rng(self._seed + _split_offset(split))
        for i in range(self._sizes[split]):
            emotion = EMOTION_ORDER[i % len(EMOTION_ORDER)]
            dialogue_id = i // 10
            utterance_id = i % 10
            available = {Modality.TEXT}
            audio = None
            video = None
            if self._with_audio and rng.random() >= self._missing_rate:
                available.add(Modality.AUDIO)
                audio = AudioInput(sample_rate=16000, waveform=_waveform(emotion, rng))
            if self._with_video and rng.random() >= self._missing_rate:
                available.add(Modality.VIDEO)
                video = VideoInput(fps=25.0, frames=_frames(emotion, rng))
            yield RawSample(
                uid=f"{split.value}:{dialogue_id}_{utterance_id}",
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                text=_text(emotion, i),
                speaker=f"speaker_{i % 5}",
                split=split,
                mask=ModalityMask(available=frozenset(available)),
                audio=audio,
                video=video,
                emotion=emotion,
                sentiment=_sentiment(emotion),
                metadata={"source": "synthetic"},
            )


def _split_offset(split: Split) -> int:
    return {Split.TRAIN: 0, Split.DEV: 10_000, Split.TEST: 20_000}[split]


def _text(emotion: Emotion, index: int) -> str:
    words = _POSITIVE.get(emotion) or _NEGATIVE.get(emotion) or "ordinary"
    punctuation = "!" if emotion in (Emotion.JOY, Emotion.SURPRISE, Emotion.ANGER) else "."
    return f"{words} sample {index} {emotion.value}{punctuation}"


def _sentiment(emotion: Emotion) -> Sentiment:
    if emotion in (Emotion.JOY, Emotion.SURPRISE):
        return Sentiment.POSITIVE
    if emotion in (Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.DISGUST):
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL


def _waveform(emotion: Emotion, rng: np.random.Generator) -> np.ndarray:
    base = float(EMOTION_ORDER.index(emotion) + 1) / float(len(EMOTION_ORDER))
    t = np.linspace(0.0, 1.0, 160, dtype=np.float64)
    wave = base * np.sin(2.0 * np.pi * (1.0 + base * 4.0) * t)
    return (wave + rng.normal(0.0, 0.01, size=t.shape)).astype(np.float64)


def _frames(emotion: Emotion, rng: np.random.Generator) -> np.ndarray:
    base = float(EMOTION_ORDER.index(emotion) + 1) / float(len(EMOTION_ORDER))
    frames = np.full((4, 8, 8, 3), base, dtype=np.float64)
    frames += rng.normal(0.0, 0.005, size=frames.shape)
    return np.clip(frames, 0.0, 1.0).astype(np.float64)
