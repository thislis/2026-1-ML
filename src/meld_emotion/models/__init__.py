"""기초 학습기(Estimator) 모음.

새 학습 알고리즘 추가법은 ``models/README.md`` 참고. 모두
:class:`~meld_emotion.core.protocols.Estimator` 를 만족한다.
"""

from __future__ import annotations

from meld_emotion.models.baselines import (
    MajorityClassEstimator,
    NearestCentroidEstimator,
    RandomEstimator,
)
from meld_emotion.models.sklearn_estimators import (
    LogisticRegressionEstimator,
    SvmEstimator,
)

__all__ = [
    "LogisticRegressionEstimator",
    "MajorityClassEstimator",
    "NearestCentroidEstimator",
    "RandomEstimator",
    "SvmEstimator",
]
