"""비디오 특징 추출기."""

from __future__ import annotations

from meld_emotion.features.video.concepts import VideoConceptExtractor
from meld_emotion.features.video.frame_embeddings import VideoFrameEmbeddingExtractor
from meld_emotion.features.video.timesformer import TimeSformerVideoExtractor
from meld_emotion.features.video.videoprism import VideoPrismVideoExtractor
from meld_emotion.features.video.visual import VisualCueExtractor

__all__ = [
    "VideoConceptExtractor",
    "VideoFrameEmbeddingExtractor",
    "TimeSformerVideoExtractor",
    "VideoPrismVideoExtractor",
    "VisualCueExtractor",
]
