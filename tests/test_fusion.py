"""Early/Late fusion 과 모달리티 마스킹."""

from __future__ import annotations

import numpy as np

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.types import Modality
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.fusion.combiners import MeanCombiner, WeightedCombiner
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.late import LateFusionClassifier
from meld_emotion.fusion.masking import get_scenario, mask_bundle
from meld_emotion.models.baselines import NearestCentroidEstimator


def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float((y_pred == y_true).mean())


def test_early_fusion_learns(train_bundle, test_bundle, y_train, y_test) -> None:
    clf = EarlyFusionClassifier(NearestCentroidEstimator, EmotionLabelEncoder().classes)
    clf.fit(train_bundle, y_train)
    prediction = clf.predict(test_bundle)
    assert prediction.proba.shape == (test_bundle.n_samples, 7)
    assert prediction.classes == EmotionLabelEncoder().classes
    assert _accuracy(prediction.y_pred, y_test) > 0.5  # 무작위(≈0.14) 대비 학습됨


def test_late_fusion_learns(train_bundle, test_bundle, y_train, y_test) -> None:
    clf = LateFusionClassifier(
        NearestCentroidEstimator, MeanCombiner(), EmotionLabelEncoder().classes
    )
    clf.fit(train_bundle, y_train)
    prediction = clf.predict(test_bundle)
    assert _accuracy(prediction.y_pred, y_test) > 0.5


def test_mask_bundle_zeros_absent_modality(test_bundle: FeatureBundle) -> None:
    masked = mask_bundle(test_bundle, get_scenario("text_only"))
    for matrix in masked.matrices:
        if matrix.modality != Modality.TEXT:
            assert not matrix.values.any()
            assert not masked.availability[matrix.modality].any()
    # 텍스트는 보존
    text_matrix = masked.by_modality(Modality.TEXT)[0]
    assert text_matrix.values.any()


def test_weighted_combiner_respects_availability() -> None:
    n = 4
    per_modality = {
        Modality.TEXT: np.tile([0.8, 0.1, 0.1], (n, 1)),
        Modality.AUDIO: np.tile([0.1, 0.8, 0.1], (n, 1)),
    }
    availability = {
        Modality.TEXT: np.ones(n, dtype=bool),
        Modality.AUDIO: np.zeros(n, dtype=bool),  # 오디오 전부 누락
    }
    combined = WeightedCombiner().combine(per_modality, availability)
    # 오디오가 누락되었으므로 텍스트 분포만 반영되어 argmax=0.
    assert np.argmax(combined, axis=1).tolist() == [0, 0, 0, 0]
