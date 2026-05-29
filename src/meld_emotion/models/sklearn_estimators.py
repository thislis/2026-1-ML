"""scikit-learn 기반 기초 학습기 (미구현 — 프로젝트의 명시된 베이스라인).

교수님 피드백에서 베이스라인으로 SVM, Logistic Regression 을 명확히 지정하라고 하였다.
실제 구현은 ``[text]`` extra 의 scikit-learn 으로 채운다(보통 각 클래스 ~10줄의 얇은 래퍼:
``self._model = SVC(C=..., kernel=..., probability=True)`` 등). 학습 결과를 위장하면 실험을
오염시키므로, 채워지기 전까지는 명시적 예외를 던진다.
"""

from __future__ import annotations

from typing import Self

from meld_emotion.core.status import raise_unimplemented, unimplemented
from meld_emotion.core.types import FloatArray, IntArray


@unimplemented("scikit-learn SVC 래퍼 필요 (kernel/C, probability=True)")
class SvmEstimator:
    """SVM 분류기 (베이스라인)."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf") -> None:
        self.C = C
        self.kernel = kernel

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        raise_unimplemented(self)

    def predict(self, x: FloatArray) -> IntArray:
        raise_unimplemented(self)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        raise_unimplemented(self)


@unimplemented("scikit-learn LogisticRegression 래퍼 필요 (C/max_iter, multi_class)")
class LogisticRegressionEstimator:
    """로지스틱 회귀 분류기 (베이스라인)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, x: FloatArray, y: IntArray) -> Self:
        raise_unimplemented(self)

    def predict(self, x: FloatArray) -> IntArray:
        raise_unimplemented(self)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        raise_unimplemented(self)
