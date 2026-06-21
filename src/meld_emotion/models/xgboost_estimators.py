"""XGBoost 기반 baseline estimator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray

_WORKER = r"""
from __future__ import annotations

import json
import sys

import numpy as np
import xgboost

mode = sys.argv[1]
if mode == "fit":
    data = np.load(sys.argv[2])
    x = data["x"]
    y = data["y"]
    model_path = sys.argv[3]
    params = json.loads(sys.argv[4])
    n_estimators = int(params.pop("n_estimators"))
    dtrain = xgboost.DMatrix(x, label=y)
    booster = xgboost.train(params, dtrain, num_boost_round=n_estimators)
    booster.save_model(model_path)
elif mode == "predict":
    model_path = sys.argv[2]
    x_path = sys.argv[3]
    out_path = sys.argv[4]
    x = np.load(x_path)
    booster = xgboost.Booster()
    booster.load_model(model_path)
    np.save(out_path, np.asarray(booster.predict(xgboost.DMatrix(x)), dtype=np.float64))
else:
    raise ValueError(f"unknown worker mode: {mode}")
"""


def _require_xgboost() -> None:
    if importlib.util.find_spec("xgboost") is None:  # pragma: no cover - 환경 의존
        raise ImportError(
            "xgboost 가 필요합니다. `uv sync --extra xgboost` (또는 --extra all) 로 설치하세요."
        )


def _run_worker(args: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"XGBoost worker failed with exit code {completed.returncode}{detail}")


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
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._model_path: Path | None = None
        self._seen: IntArray = np.zeros(0, dtype=np.int64)
        self._single_class: int | None = None

    def __del__(self) -> None:
        self._cleanup_model_files()

    def _cleanup_model_files(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = None
        self._model_path = None

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        _require_xgboost()
        self._cleanup_model_files()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._seen = np.unique(y).astype(np.int64)
        if self._seen.size == 0:
            raise ValueError("XGBoostEstimator 는 비어 있는 y 로 학습할 수 없습니다")
        if self._seen.size == 1:
            self._single_class = int(self._seen[0])
            self._model_path = None
            return self

        remap = {int(label): i for i, label in enumerate(self._seen.tolist())}
        y_local = np.asarray([remap[int(label)] for label in y], dtype=np.int64)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="meld_xgboost_")
        tmp = Path(self._tmpdir.name)
        train_path = tmp / "train.npz"
        model_path = tmp / "model.json"
        np.savez(
            train_path,
            x=x,
            y=y_local,
        )
        params: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": int(self._seen.size),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
            "eval_metric": "mlogloss",
            "nthread": 1,
        }
        _run_worker(
            [
                "fit",
                str(train_path),
                str(model_path),
                json.dumps(params),
            ]
        )
        train_path.unlink(missing_ok=True)
        self._model_path = model_path
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
        if self._model_path is None or self._tmpdir is None:
            raise RuntimeError("학습되지 않은 학습기입니다. 먼저 fit 을 호출하세요.")
        tmp = Path(self._tmpdir.name)
        x_path = tmp / "predict_x.npy"
        out_path = tmp / "predict_proba.npy"
        np.save(x_path, x)
        _run_worker(
            [
                "predict",
                str(self._model_path),
                str(x_path),
                str(out_path),
            ]
        )
        raw = np.asarray(np.load(out_path), dtype=np.float64)
        x_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        full[:, self._seen] = raw
        return full

    def predict(self, x: FloatArray) -> IntArray:
        return np.asarray(np.argmax(self.predict_proba(x), axis=1), dtype=np.int64)
