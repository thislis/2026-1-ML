"""XGBoost 기반 baseline estimator."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray


def _require_xgboost() -> Any:
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "xgboost 가 필요합니다. `uv sync --extra xgboost` (또는 --extra all) 로 설치하세요."
        ) from exc
    return xgboost


@real
class XGBoostEstimator:
    """XGBClassifier(objective='multi:softprob') 래퍼."""

    def __init__(
        self,
        n_classes: int | None = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self._model: Any | None = None
        self._seen: IntArray = np.zeros(0, dtype=np.int64)
        self._single_class: int | None = None

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        xgboost = _require_xgboost()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._seen = np.unique(y).astype(np.int64)
        if self._seen.size == 0:
            raise ValueError("XGBoostEstimator 는 비어 있는 y 로 학습할 수 없습니다")
        if self._seen.size == 1:
            self._single_class = int(self._seen[0])
            self._model = None
            return self

        remap = {int(label): i for i, label in enumerate(self._seen.tolist())}
        y_local = np.asarray([remap[int(label)] for label in y], dtype=np.int64)
        model = xgboost.XGBClassifier(
            objective="multi:softprob",
            num_class=int(self._seen.size),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            eval_metric="mlogloss",
        )
        model.fit(x, y_local)
        self._model = model
        self._single_class = None
        return self

    def _full_k(self) -> int:
        seen_max = int(self._seen.max()) + 1 if self._seen.size else 0
        return max(self._n_classes or 0, seen_max)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        if self._seen.size == 0:
            raise RuntimeError("학습되지 않은 학습기입니다. 먼저 fit 을 호출하세요.")
        x = np.asarray(x, dtype=np.float64)
        full = np.zeros((x.shape[0], self._full_k()), dtype=np.float64)
        if self._single_class is not None:
            full[:, self._single_class] = 1.0
            return full
        if self._model is None:
            raise RuntimeError("학습되지 않은 학습기입니다. 먼저 fit 을 호출하세요.")
        raw = np.asarray(self._model.predict_proba(x), dtype=np.float64)
        full[:, self._seen] = raw
        return full

    def predict(self, x: FloatArray) -> IntArray:
        return np.asarray(np.argmax(self.predict_proba(x), axis=1), dtype=np.int64)
