"""CatBoost 기반 baseline estimator."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray


def _require_catboost() -> Any:
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise ImportError(
            "catboost 가 필요합니다. `uv sync --extra catboost` (또는 --extra all) 로 설치하세요."
        ) from exc
    return catboost


@real
class CatBoostEstimator:
    """CatBoostClassifier(loss_function='MultiClass') 래퍼."""

    def __init__(
        self,
        n_classes: int | None = None,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 0,
    ) -> None:
        self._n_classes = n_classes
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self._model: Any | None = None
        self._seen: IntArray = np.zeros(0, dtype=np.int64)
        self._single_class: int | None = None

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        catboost = _require_catboost()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._seen = np.unique(y).astype(np.int64)
        if self._seen.size == 0:
            raise ValueError("CatBoostEstimator 는 비어 있는 y 로 학습할 수 없습니다")
        if self._seen.size == 1:
            self._single_class = int(self._seen[0])
            self._model = None
            return self

        remap = {int(label): i for i, label in enumerate(self._seen.tolist())}
        y_local = np.asarray([remap[int(label)] for label in y], dtype=np.int64)
        model = catboost.CatBoostClassifier(
            loss_function="MultiClass",
            classes_count=int(self._seen.size),
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
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
