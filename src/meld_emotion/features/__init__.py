"""특징 추출기 모음 (텍스트/오디오/비디오 x 임베딩/개념).

새 추출기 추가법은 ``features/README.md`` 참고. 모든 추출기는
:class:`~meld_emotion.features.base.BaseFeatureExtractor` 를 상속하고
:class:`~meld_emotion.core.protocols.FeatureExtractor` 를 만족한다.
"""

from __future__ import annotations

from meld_emotion.features.audio import (
    AudioConceptExtractor,
    MfccAcousticExtractor,
    Wav2Vec2XlsrAudioExtractor,
)
from meld_emotion.features.base import BaseFeatureExtractor
from meld_emotion.features.text import (
    BowTextExtractor,
    EmbeddingGemmaTextExtractor,
    SentenceEmbeddingExtractor,
    TextConceptExtractor,
    TfidfTextExtractor,
)
from meld_emotion.features.video import VideoConceptExtractor, VisualCueExtractor

__all__ = [
    "BaseFeatureExtractor",
    # text
    "TextConceptExtractor",
    "BowTextExtractor",
    "TfidfTextExtractor",
    "SentenceEmbeddingExtractor",
    "EmbeddingGemmaTextExtractor",
    # audio
    "AudioConceptExtractor",
    "MfccAcousticExtractor",
    "Wav2Vec2XlsrAudioExtractor",
    # video
    "VideoConceptExtractor",
    "VisualCueExtractor",
]
