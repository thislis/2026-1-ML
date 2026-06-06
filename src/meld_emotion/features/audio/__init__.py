"""오디오 특징 추출기."""

from __future__ import annotations

from meld_emotion.features.audio.acoustic import MfccAcousticExtractor
from meld_emotion.features.audio.concepts import AudioConceptExtractor
from meld_emotion.features.audio.wav2vec2 import Wav2Vec2XlsrAudioExtractor

__all__ = ["AudioConceptExtractor", "MfccAcousticExtractor", "Wav2Vec2XlsrAudioExtractor"]
