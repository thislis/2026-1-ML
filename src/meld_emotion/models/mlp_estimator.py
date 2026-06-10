"""Small PyTorch MLP estimator for pooled/tabular embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from meld_emotion.core.status import real
from meld_emotion.core.types import FloatArray, IntArray


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch 가 필요합니다. `uv sync --extra deep` (또는 --extra all) 로 설치하세요."
        ) from exc
    return torch


def _require_nn() -> Any:
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch 가 필요합니다. `uv sync --extra deep` (또는 --extra all) 로 설치하세요."
        ) from exc
    return nn


@dataclass(frozen=True)
class _MlpParams:
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    validation_split: float
    class_weight: str
    class_weights: tuple[float, ...]
    random_seed: int
    device: str


@real
class MlpEstimator:
    """LayerNorm → Linear → GELU → Dropout → Linear classifier for pooled features."""

    def __init__(
        self,
        n_classes: int | None = None,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        validation_split: float = 0.1,
        class_weight: str = "none",
        class_weights: tuple[float, ...] = (),
        random_seed: int = 0,
        device: str = "cpu",
    ) -> None:
        _validate_params(
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            class_weight=class_weight,
        )
        self._n_classes = n_classes
        self._params = _MlpParams(
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            class_weight=class_weight,
            class_weights=class_weights,
            random_seed=random_seed,
            device=device,
        )
        self._model: Any | None = None
        self._input_dim = 0
        self._k = 0

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        torch = _require_torch()
        nn = _require_nn()
        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        if x_np.ndim != 2:
            raise ValueError(f"x must be 2D: ndim={x_np.ndim}")
        if y_np.shape != (x_np.shape[0],):
            raise ValueError(f"y shape must be ({x_np.shape[0]},): {y_np.shape}")

        _seed_everything(torch, self._params.random_seed)
        self._input_dim = int(x_np.shape[1])
        self._k = _resolve_n_classes(self._n_classes, y_np)
        self._model = _make_model(self._input_dim, self._k, self._params).to(self._params.device)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._params.learning_rate,
            weight_decay=self._params.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(weight=self._class_weight_tensor(y_np, torch))
        train_idx, val_idx = _train_val_indices(
            n_samples=x_np.shape[0],
            validation_split=self._params.validation_split,
            seed=self._params.random_seed,
        )
        x_tensor = torch.as_tensor(x_np, device=self._params.device)
        y_tensor = torch.as_tensor(y_np, device=self._params.device)

        best_state: Mapping[str, Any] | None = None
        best_loss = float("inf")
        stale = 0
        for _epoch in range(self._params.max_epochs):
            self._model.train()
            for batch_idx in _batches(train_idx, self._params.batch_size, self._params.random_seed):
                logits = self._model(x_tensor[batch_idx])
                loss = criterion(logits, y_tensor[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if val_idx.size == 0:
                continue
            val_loss = self._validation_loss(x_tensor, y_tensor, val_idx, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = _state_dict_cpu(self._model)
                stale = 0
            else:
                stale += 1
                if stale >= self._params.early_stopping_patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(dict(best_state))
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        torch = _require_torch()
        model = self._require_model()
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"x must be 2D: ndim={x_np.ndim}")
        if x_np.shape[1] != self._input_dim:
            raise ValueError(f"x feature dim changed: {x_np.shape[1]} != {self._input_dim}")
        model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_np.shape[0], self._params.batch_size):
                batch = torch.as_tensor(
                    x_np[start : start + self._params.batch_size],
                    device=self._params.device,
                )
                proba = torch.softmax(model(batch), dim=-1).cpu().numpy()
                chunks.append(np.asarray(proba, dtype=np.float64))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self._k))

    def predict(self, x: FloatArray) -> IntArray:
        return np.asarray(np.argmax(self.predict_proba(x), axis=1), dtype=np.int64)

    def save(self, path: str | Path) -> None:
        torch = _require_torch()
        model = self._require_model()
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_classes": self._k,
                "input_dim": self._input_dim,
                "params": self._params.__dict__,
                "model_state_dict": _state_dict_cpu(model),
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> MlpEstimator:
        torch = _require_torch()
        checkpoint = torch.load(Path(path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"MLP checkpoint 형식이 올바르지 않습니다: {path}")
        params_obj = checkpoint.get("params")
        if not isinstance(params_obj, Mapping):
            raise ValueError("MLP checkpoint 에 params 매핑이 없습니다")
        params = dict(params_obj)
        if device is not None:
            params["device"] = device
        estimator = cls(n_classes=int(checkpoint["n_classes"]), **params)
        estimator._input_dim = int(checkpoint["input_dim"])
        estimator._k = int(checkpoint["n_classes"])
        estimator._model = _make_model(estimator._input_dim, estimator._k, estimator._params).to(
            estimator._params.device
        )
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("MLP checkpoint 에 model_state_dict 가 없습니다")
        estimator._model.load_state_dict(dict(state))
        return estimator

    def _validation_loss(self, x: Any, y: Any, indices: np.ndarray, criterion: Any) -> float:
        model = self._require_model()
        model.eval()
        torch = _require_torch()
        with torch.no_grad():
            loss = criterion(model(x[indices]), y[indices])
        return float(loss.detach().cpu().item())

    def _class_weight_tensor(self, y: np.ndarray, torch: Any) -> Any:
        weights = _class_weights(
            y,
            n_classes=self._k,
            mode=self._params.class_weight,
            explicit=self._params.class_weights,
        )
        if weights is None:
            return None
        return torch.as_tensor(weights, dtype=torch.float32, device=self._params.device)

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("학습되지 않은 학습기입니다. 먼저 fit 을 호출하세요.")
        return self._model


def _make_model(input_dim: int, n_classes: int, params: _MlpParams) -> Any:
    nn = _require_nn()
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, params.hidden_dim),
        nn.GELU(),
        nn.Dropout(params.dropout),
        nn.Linear(params.hidden_dim, n_classes),
    )


def _validate_params(
    *,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    validation_split: float,
    class_weight: str,
) -> None:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    if class_weight not in {"none", "balanced", "explicit"}:
        raise ValueError("class_weight must be 'none', 'balanced', or 'explicit'")


def _resolve_n_classes(n_classes: int | None, y: np.ndarray) -> int:
    inferred = int(y.max()) + 1 if y.size else 0
    return max(n_classes or 0, inferred)


def _seed_everything(torch: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_val_indices(
    *,
    n_samples: int,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if n_samples < 2 or validation_split <= 0.0:
        return indices, np.zeros(0, dtype=np.int64)
    n_val = min(n_samples - 1, max(1, round(n_samples * validation_split)))
    return indices[n_val:], indices[:n_val]


def _batches(indices: np.ndarray, batch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    return [shuffled[start : start + batch_size] for start in range(0, shuffled.size, batch_size)]


def _class_weights(
    y: np.ndarray,
    *,
    n_classes: int,
    mode: str,
    explicit: tuple[float, ...],
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "explicit":
        if len(explicit) != n_classes:
            raise ValueError(f"explicit class_weights length must be {n_classes}: {len(explicit)}")
        weights = np.asarray(explicit, dtype=np.float32)
    else:
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        total = float(counts.sum())
        weights = np.zeros(n_classes, dtype=np.float32)
        present = counts > 0.0
        weights[present] = total / (float(n_classes) * counts[present])
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("class weights must be finite non-negative values")
    positive = weights > 0.0
    if np.any(positive):
        weights = weights / float(weights[positive].mean())
    return weights


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
