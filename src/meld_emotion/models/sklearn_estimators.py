"""scikit-learn 기반 기초 학습기 (완전 구현 — 프로젝트의 명시된 베이스라인).

교수님 피드백에서 지정한 SVM·Logistic Regression 을 비롯해, 비교용 RandomForest·KNN 까지
얇은 래퍼로 제공한다. 모두 :class:`~meld_emotion.core.protocols.Estimator` 를 만족한다.

설계 요점:

- **선택적 의존성**: scikit-learn 은 ``[text]`` extra 에만 있다. 미설치 시 ``fit`` 단계에서
  명확한 :class:`ImportError` 를 던진다(가짜 결과로 실험을 오염시키지 않는다).
- **전체 클래스 폭 보존**: sklearn 의 ``predict_proba`` 는 **학습에서 본 클래스 열만** 돌려준다.
  한 분할에 소수 클래스(fear/disgust 등)가 없으면 열 수가 K 보다 작아진다. 그래서 주입된
  ``n_classes`` 와 모델의 ``classes_`` 를 이용해 항상 ``(n_samples, K)`` 로 **확장**한다
  (누락 클래스 열은 0). 이로써 융합 분류기의 :class:`PredictionSet` 형상 검증이 깨지지 않는다.
- **전처리 캡슐화**: 스케일에 민감한 모델(SVM/LogReg/KNN)은 ``StandardScaler`` 를 파이프라인
  앞단에 두고, 트리 기반(RandomForest)은 스케일 없이 둔다 — 각 학습기가 필요한 전처리를
  스스로 소유한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray


def _require_sklearn() -> Any:
    """scikit-learn 을 지연 import 한다(미설치 시 설치 안내와 함께 ImportError)."""

    try:
        import sklearn
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "scikit-learn 이 필요합니다. `uv sync --extra text` (또는 --extra all) 로 설치하세요."
        ) from exc
    return sklearn


class _SklearnProbaEstimator(ABC):
    """sklearn 분류기를 감싸 전체 클래스 폭의 확률을 보장하는 공통 기반.

    하위 클래스는 :meth:`_make_model` 에서 (필요하면 전처리를 포함한) sklearn 추정기를
    돌려주기만 하면 된다.
    """

    def __init__(self, n_classes: int | None = None) -> None:
        self._n_classes = n_classes
        self._model: Any | None = None
        self._seen: IntArray = np.zeros(0, dtype=np.int64)

    @abstractmethod
    def _make_model(self) -> Any:
        """학습할 sklearn 추정기를 생성한다(매 호출 새 인스턴스)."""

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        _require_sklearn()
        model = self._make_model()
        model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
        self._model = model
        self._seen = np.asarray(model.classes_, dtype=np.int64)
        return self

    def _require(self) -> Any:
        if self._model is None:
            raise RuntimeError("학습되지 않은 학습기입니다. 먼저 fit 을 호출하세요.")
        return self._model

    def _full_k(self) -> int:
        seen_max = int(self._seen.max()) + 1 if self._seen.size else 0
        return max(self._n_classes or 0, seen_max)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        model = self._require()
        raw = np.asarray(model.predict_proba(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        full = np.zeros((raw.shape[0], self._full_k()), dtype=np.float64)
        full[:, self._seen] = raw  # 본 클래스 열에만 채우고 나머지는 0
        return full

    def _expand_seen_scores(self, raw: FloatArray, fill_value: float = 0.0) -> FloatArray:
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim == 1:
            if len(self._seen) == 2:
                values = np.stack([-values, values], axis=1)
            else:
                values = values.reshape(-1, 1)
        full = np.full((values.shape[0], self._full_k()), fill_value, dtype=np.float64)
        full[:, self._seen] = values
        return full

    def predict(self, x: FloatArray) -> IntArray:
        # 확장된 확률에서 argmax → 전역 클래스 인덱스(누락 클래스는 0 확률이라 선택되지 않음).
        return np.asarray(np.argmax(self.predict_proba(x), axis=1), dtype=np.int64)


@real
class SvmEstimator(_SklearnProbaEstimator):
    """SVM 분류기 (베이스라인). StandardScaler + SVC(probability=True)."""

    def __init__(self, n_classes: int | None = None, C: float = 1.0, kernel: str = "rbf") -> None:
        super().__init__(n_classes)
        self.C = C
        self.kernel = kernel

    def _make_model(self) -> Any:
        _require_sklearn()
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(),
            SVC(
                C=self.C,
                kernel=self.kernel,
                probability=True,
                decision_function_shape="ovr",
                random_state=0,
            ),
        )

    def decision_scores(self, x: FloatArray) -> FloatArray:
        """Return full-width one-vs-rest SVM decision scores for margin routing."""

        model = self._require()
        raw = np.asarray(
            model.decision_function(np.asarray(x, dtype=np.float64)),
            dtype=np.float64,
        )
        return self._expand_seen_scores(raw, fill_value=-np.inf)


@real
class LogisticRegressionEstimator(_SklearnProbaEstimator):
    """로지스틱 회귀 분류기 (베이스라인). StandardScaler + LogisticRegression."""

    def __init__(self, n_classes: int | None = None, C: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__(n_classes)
        self.C = C
        self.max_iter = max_iter

    def _make_model(self) -> Any:
        _require_sklearn()
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=self.max_iter),
        )


@real
class RandomForestEstimator(_SklearnProbaEstimator):
    """랜덤 포레스트 분류기 (비교 베이스라인). 트리 기반이라 스케일 불필요."""

    def __init__(
        self, n_classes: int | None = None, n_estimators: int = 200, max_depth: int | None = None
    ) -> None:
        super().__init__(n_classes)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _make_model(self) -> Any:
        _require_sklearn()
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=0
        )


@real
class KnnEstimator(_SklearnProbaEstimator):
    """k-최근접 이웃 분류기 (비교 베이스라인). 거리 기반이라 StandardScaler 포함."""

    def __init__(self, n_classes: int | None = None, n_neighbors: int = 5) -> None:
        super().__init__(n_classes)
        self.n_neighbors = n_neighbors

    def _make_model(self) -> Any:
        _require_sklearn()
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=self.n_neighbors),
        )
