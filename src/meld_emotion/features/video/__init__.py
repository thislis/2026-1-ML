"""비디오 특징 추출기."""

from __future__ import annotations

from meld_emotion.features.video.concepts import VideoConceptExtractor
from meld_emotion.features.video.visual import VisualCueExtractor

__all__ = ["VideoConceptExtractor", "VisualCueExtractor"]
