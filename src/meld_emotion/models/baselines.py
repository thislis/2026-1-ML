"""numpy 전용 기초 학습기 (완전 구현).

무거운 라이브러리 없이 전체 파이프라인을 학습/평가/설명까지 실행하기 위한 학습기들이다.
모두 :class:`~meld_emotion.core.protocols.Estimator` 를 만족한다.

레이블은 0..K-1 의 연속 정수라고 가정한다(:class:`EmotionLabelEncoder` 가 보장).
``predict_proba`` 는 (n_samples, K) 행렬을 반환하며 열 index 가 곧 클래스 index 이다.

**클래스 수(K)는 학습 데이터가 아니라 전체 레이블 공간에서 와야 한다.** 한 분할에 소수 클래스
(예: fear/disgust)가 누락되면 ``y.max()+1`` 로 추론한 K 가 7 보다 작아져 ``predict_proba`` 의
열 수가 어긋난다(융합 분류기가 만드는 :class:`PredictionSet` 의 형상 검증 실패). 따라서 각
학습기는 생성자 첫 인자로 ``n_classes`` 를 받아 이를 우선 사용하고, 주어지지 않은 경우에만
데이터에서 추론한다(빌더는 항상 인코더의 클래스 수를 주입한다).
"""

from __future__ import annotations

from typing import Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray


def _infer_n_classes(y: IntArray) -> int:
    return int(y.max()) + 1 if y.size else 0


def _resolve_n_classes(n_classes: int | None, y: IntArray) -> int:
    """주입된 ``n_classes`` 를 우선하되, 데이터가 더 많은 클래스를 담고 있으면 그에 맞춘다."""

    inferred = _infer_n_classes(y)
    if n_classes is None:
        return inferred
    return max(n_classes, inferred)


def _as_float(values: object) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _as_int(values: object) -> IntArray:
    return np.asarray(values, dtype=np.int64)


@real
class MajorityClassEstimator:
    """항상 최빈 클래스를 예측. 다른 학습기 성능 비교의 하한선."""

    def __init__(self, n_classes: int | None = None) -> None:
        self._n_classes = n_classes
        self._proba: FloatArray = np.zeros(0, dtype=np.float64)
        self._majority: int = 0

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        k = _resolve_n_classes(self._n_classes, y)
        counts = np.bincount(y, minlength=k).astype(np.float64)
        total = counts.sum()
        self._proba = counts / total if total > 0 else counts
        self._majority = int(np.argmax(counts)) if counts.size else 0
        return self

    def predict(self, x: FloatArray) -> IntArray:
        return np.full(x.shape[0], self._majority, dtype=np.int64)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return _as_float(np.tile(self._proba, (x.shape[0], 1)))


@real
class RandomEstimator:
    """시드 기반 무작위 예측. 정상성/형상 검증용 기준선."""

    def __init__(self, n_classes: int | None = None, seed: int = 0) -> None:
        self._n_classes = n_classes
        self._seed = seed
        self._k = 0

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        self._k = _resolve_n_classes(self._n_classes, y)
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        rng = np.random.default_rng(self._seed)
        raw = rng.random((x.shape[0], max(self._k, 1)))
        return _as_float(raw / raw.sum(axis=1, keepdims=True))

    def predict(self, x: FloatArray) -> IntArray:
        return _as_int(np.argmax(self.predict_proba(x), axis=1))


@real
class NearestCentroidEstimator:
    """클래스별 중심(centroid)과의 거리로 분류하는 학습기.

    특징을 z-점수로 표준화한 뒤 클래스 평균 벡터를 학습하고, 예측 시 가장 가까운 중심의
    클래스를 고른다. 확률은 음의 거리에 softmax 를 적용해 산출한다. 순수 numpy 로 구현되어
    합성 데이터에서도 실제 학습 신호를 보여 준다.
    """

    def __init__(self, n_classes: int | None = None, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature 는 양수여야 합니다")
        self._n_classes = n_classes
        self._temperature = temperature
        self._mean: FloatArray = np.zeros(0)
        self._std: FloatArray = np.ones(0)
        self._centroids: FloatArray = np.zeros((0, 0))
        self._k = 0

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        x = np.asarray(x, dtype=np.float64)
        self._k = _resolve_n_classes(self._n_classes, y)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        self._std[self._std == 0.0] = 1.0
        xs = (x - self._mean) / self._std
        n_features = x.shape[1]
        centroids = np.zeros((self._k, n_features), dtype=np.float64)
        global_mean = xs.mean(axis=0)
        for c in range(self._k):
            mask = y == c
            centroids[c] = xs[mask].mean(axis=0) if np.any(mask) else global_mean
        self._centroids = centroids
        return self

    def _distances(self, x: FloatArray) -> FloatArray:
        xs = (np.asarray(x, dtype=np.float64) - self._mean) / self._std
        # (n, 1, d) - (1, k, d) -> (n, k)
        diff = xs[:, None, :] - self._centroids[None, :, :]
        return _as_float(np.linalg.norm(diff, axis=2))

    def predict(self, x: FloatArray) -> IntArray:
        return _as_int(np.argmin(self._distances(x), axis=1))

    def predict_proba(self, x: FloatArray) -> FloatArray:
        distances = self._distances(x)
        logits = -distances / self._temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return _as_float(exp / exp.sum(axis=1, keepdims=True))
