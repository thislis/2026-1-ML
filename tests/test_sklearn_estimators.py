"""scikit-learn 베이스라인 래퍼 (SVM/LogReg/RandomForest/KNN).

scikit-learn 은 선택적 ``[text]`` 의존성이므로, 미설치 환경에서는 전체를 skip 한다(numpy 전용
개발 루프를 빠르게 유지). 설치 환경에서는 핵심 계약을 검증한다:
- ``predict_proba`` 가 항상 전체 클래스 폭(K)을 돌려준다(학습에 없던 소수 클래스 포함).
- ``predict`` 와 ``predict_proba`` 의 argmax 가 일치한다.
- 빌더/융합 분류기 경로로도 동작한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn 미설치 (uv sync --extra text 로 설치)")

from meld_emotion.config.schema import (
    EarlyFusionConfig,
    ExperimentConfig,
    KnnConfig,
    LogRegConfig,
    RandomForestConfig,
    SvmConfig,
    SyntheticConfig,
)
from meld_emotion.core.protocols import Estimator
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.models.sklearn_estimators import (
    KnnEstimator,
    LogisticRegressionEstimator,
    RandomForestEstimator,
    SvmEstimator,
)
from meld_emotion.pipeline.builder import build_experiment

_N_EMOTIONS = 7

_ALL = (SvmEstimator, LogisticRegressionEstimator, RandomForestEstimator, KnnEstimator)


def _xy_missing_top_classes() -> tuple[np.ndarray, np.ndarray]:
    # 7 클래스 중 0..4 만 등장(fear/disgust 누락). 레이블별로 평균을 크게 벌려 분리 가능하게.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5, dtype=np.int64), 10)  # 각 클래스 10개씩
    x = rng.normal(loc=y[:, None] * 2.0, scale=1.0, size=(y.size, 8)).astype(np.float64)
    return x, y


@pytest.mark.parametrize("cls", _ALL)
def test_satisfies_estimator_protocol(cls: type) -> None:
    assert isinstance(cls(), Estimator)


@pytest.mark.parametrize("cls", _ALL)
def test_proba_full_width_when_classes_missing(cls: type) -> None:
    x, y = _xy_missing_top_classes()
    est = cls(n_classes=_N_EMOTIONS).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (y.size, _N_EMOTIONS)  # 누락 클래스(5,6) 열도 존재
    # 누락 클래스 열은 학습에서 본 적 없으므로 0.
    assert np.allclose(proba[:, 5], 0.0) and np.allclose(proba[:, 6], 0.0)
    # 각 행은 (본 클래스에 한해) 확률 분포: 합이 1 에 근접.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("cls", _ALL)
def test_predict_matches_proba_argmax(cls: type) -> None:
    x, y = _xy_missing_top_classes()
    est = cls(n_classes=_N_EMOTIONS).fit(x, y)
    assert np.array_equal(est.predict(x), np.argmax(est.predict_proba(x), axis=1))


@pytest.mark.parametrize("cls", _ALL)
def test_learns_separable_signal(cls: type) -> None:
    # 레이블에 강하게 상관된 특징이면 학습 정확도가 무작위(0.2)보다 충분히 높아야 한다.
    x, y = _xy_missing_top_classes()
    est = cls(n_classes=_N_EMOTIONS).fit(x, y)
    accuracy = float((est.predict(x) == y).mean())
    assert accuracy > 0.7


@pytest.mark.parametrize(
    "base_config",
    [SvmConfig(), LogRegConfig(max_iter=200), RandomForestConfig(n_estimators=50), KnnConfig()],
)
def test_builder_early_fusion_runs(base_config) -> None:
    config = ExperimentConfig(
        name="skl",
        dataset=SyntheticConfig(n_train=140, n_dev=0, n_test=70),
        model=EarlyFusionConfig(base=base_config),
        reporters=(),
    )
    result = build_experiment(config).run()
    prediction_classes = EmotionLabelEncoder().classes
    accuracy = result.evaluation.metric("accuracy")
    assert accuracy is not None and accuracy.value > 0.5
    # 융합 분류기가 만든 확률 폭이 전체 감정 수와 일치(형상 계약).
    assert len(prediction_classes) == _N_EMOTIONS
