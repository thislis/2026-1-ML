"""기초 학습기의 클래스 수(K) 처리 — 분할에 소수 클래스가 누락돼도 형상이 어긋나지 않음."""

from __future__ import annotations

import numpy as np

from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.types import FeatureKind, Modality
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.models.baselines import (
    LinearRegressionEstimator,
    MajorityClassEstimator,
    NearestCentroidEstimator,
    RandomEstimator,
)

_N_EMOTIONS = 7


def _xy_missing_top_classes() -> tuple[np.ndarray, np.ndarray]:
    # 7 클래스 중 0..4 만 등장(fear/disgust 누락) → y.max()+1 == 5.
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    x = rng.normal(size=(y.size, 6)).astype(np.float64)
    return x, y


def test_majority_proba_width_uses_injected_n_classes() -> None:
    x, y = _xy_missing_top_classes()
    est = MajorityClassEstimator(n_classes=_N_EMOTIONS).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (y.size, _N_EMOTIONS)  # 누락 클래스 열도 존재


def test_random_proba_width_uses_injected_n_classes() -> None:
    x, y = _xy_missing_top_classes()
    est = RandomEstimator(n_classes=_N_EMOTIONS).fit(x, y)
    assert est.predict_proba(x).shape == (y.size, _N_EMOTIONS)


def test_centroid_proba_width_uses_injected_n_classes() -> None:
    x, y = _xy_missing_top_classes()
    est = NearestCentroidEstimator(n_classes=_N_EMOTIONS).fit(x, y)
    assert est.predict_proba(x).shape == (y.size, _N_EMOTIONS)


def test_linear_regression_proba_width_uses_injected_n_classes() -> None:
    x, y = _xy_missing_top_classes()
    est = LinearRegressionEstimator(n_classes=_N_EMOTIONS).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (y.size, _N_EMOTIONS)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_linear_regression_predict_matches_proba_argmax() -> None:
    x, y = _xy_missing_top_classes()
    est = LinearRegressionEstimator(n_classes=_N_EMOTIONS).fit(x, y)
    assert np.array_equal(est.predict(x), np.argmax(est.predict_proba(x), axis=1))


def test_injected_n_classes_grows_if_data_has_more() -> None:
    # 안전장치: 데이터가 더 많은 클래스를 담으면 그에 맞춘다(주입값이 과소여도 깨지지 않음).
    x, y = _xy_missing_top_classes()
    est = MajorityClassEstimator(n_classes=2).fit(x, y)  # 실제론 5개 등장
    assert est.predict_proba(x).shape[1] == 5


def _bundle(x: np.ndarray, uids: tuple[str, ...]) -> FeatureBundle:
    matrix = FeatureMatrix(
        values=x,
        names=tuple(f"f{i}" for i in range(x.shape[1])),
        modality=Modality.TEXT,
        kind=FeatureKind.EMBEDDING,
    )
    return FeatureBundle(uids=uids, matrices=(matrix,))


def test_early_fusion_predicts_full_width_when_train_misses_classes() -> None:
    """회귀 테스트: 학습 분할이 소수 클래스를 누락해도 PredictionSet 형상 검증이 통과한다."""

    x, y = _xy_missing_top_classes()
    classes = EmotionLabelEncoder().classes
    uids = tuple(str(i) for i in range(y.size))

    clf = EarlyFusionClassifier(NearestCentroidEstimator, classes).fit(_bundle(x, uids), y)
    prediction = clf.predict(_bundle(x, uids))

    # proba 열 수가 전체 감정 수(7)와 일치해야 한다(누락 클래스 포함).
    assert prediction.proba.shape == (y.size, _N_EMOTIONS)
    assert len(prediction.classes) == _N_EMOTIONS
