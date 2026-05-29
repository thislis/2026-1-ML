"""상태 마커 동작: 미구현은 예외, 임시는 경고."""

from __future__ import annotations

import numpy as np
import pytest

from meld_emotion.core.data import ModalityMask, RawSample
from meld_emotion.core.status import ComponentStatus, status_of
from meld_emotion.core.types import Modality, Split
from meld_emotion.data.media import MediaLoader
from meld_emotion.features.text.tfidf import TfidfTextExtractor
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.models.sklearn_estimators import SvmEstimator


def _text_sample() -> RawSample:
    return RawSample(
        uid="x",
        dialogue_id=0,
        utterance_id=0,
        text="hello world",
        speaker="s",
        split=Split.TRAIN,
        mask=ModalityMask.of(Modality.TEXT),
    )


def test_unimplemented_estimator_raises() -> None:
    with pytest.raises(NotImplementedError):
        SvmEstimator().fit(np.zeros((2, 2)), np.zeros(2, dtype=np.int64))


def test_unimplemented_media_loader_raises() -> None:
    from meld_emotion.core.data import AudioInput

    with pytest.raises(NotImplementedError):
        MediaLoader().load_audio(AudioInput(sample_rate=16000))


def test_placeholder_warns_and_returns_valid_matrix() -> None:
    with pytest.warns(RuntimeWarning):
        matrix = TfidfTextExtractor().transform([_text_sample()])
    assert matrix.values.shape[0] == 1
    assert matrix.values.shape[1] == len(matrix.names)


def test_status_tags() -> None:
    assert status_of(TfidfTextExtractor) == ComponentStatus.PLACEHOLDER
    assert status_of(SvmEstimator) == ComponentStatus.UNIMPLEMENTED
    assert status_of(EarlyFusionClassifier) == ComponentStatus.REAL
