"""Permutation 중요도 설명기 (완전 구현).

각 특징 열을 무작위로 섞었을 때 지표가 얼마나 하락하는지로 특징 기여도를 측정한다.
개념(concept) 특징의 기여도를 보면 "관찰 가능한 단서(말의 에너지, 어휘 극성, 얼굴 움직임)가
예측에 기여하는가"라는 제안서의 질문에 답할 수 있다.

비용은 (대상 열 수 * ``n_repeats``)번의 ``predict`` 이다. 고차원 임베딩(예: 실제 TF-IDF 수천
차원)까지 모두 섞으면 비현실적으로 느려지므로, 기본적으로 **개념(concept) 특징만** 대상으로
한다(``kinds`` 로 조절). 제안서의 설명 초점도 해석 가능한 개념 벡터이므로 기본값이 자연스럽다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.protocols import Classifier, Metric
from meld_emotion.core.results import ExplanationReport, FeatureContribution
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind, IntArray


def _replace_matrix(bundle: FeatureBundle, index: int, matrix: FeatureMatrix) -> FeatureBundle:
    matrices = list(bundle.matrices)
    matrices[index] = matrix
    return FeatureBundle(
        uids=bundle.uids, matrices=tuple(matrices), availability=bundle.availability
    )


@real
class PermutationImportanceExplainer:
    """열 단위 permutation 중요도."""

    def __init__(
        self,
        metric: Metric,
        n_repeats: int = 5,
        seed: int = 0,
        top_k: int = 20,
        kinds: Sequence[FeatureKind] = (FeatureKind.CONCEPT,),
    ) -> None:
        self._metric = metric
        self._n_repeats = n_repeats
        self._seed = seed
        self._top_k = top_k
        self._kinds = tuple(kinds)

    def _score(self, model: Classifier, bundle: FeatureBundle, y_true: IntArray) -> float:
        return self._metric.compute(y_true, model.predict(bundle)).value

    def explain(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray
    ) -> ExplanationReport:
        baseline = self._score(model, bundle, y_true)
        rng = np.random.default_rng(self._seed)
        contributions: list[FeatureContribution] = []

        for m_idx, matrix in enumerate(bundle.matrices):
            if matrix.kind not in self._kinds:
                continue
            for col in range(matrix.n_features):
                drops = np.empty(self._n_repeats, dtype=np.float64)
                for r in range(self._n_repeats):
                    permuted = self._permute(matrix, col, rng)
                    perturbed = _replace_matrix(bundle, m_idx, permuted)
                    drops[r] = baseline - self._score(model, perturbed, y_true)
                contributions.append(
                    FeatureContribution(
                        name=matrix.names[col],
                        modality=matrix.modality,
                        importance=float(np.mean(drops)),
                        std=float(np.std(drops)),
                    )
                )

        contributions.sort(key=lambda c: c.importance, reverse=True)
        return ExplanationReport(feature_contributions=tuple(contributions[: self._top_k]))

    @staticmethod
    def _permute(matrix: FeatureMatrix, col: int, rng: np.random.Generator) -> FeatureMatrix:
        values = matrix.values.copy()
        values[:, col] = values[rng.permutation(values.shape[0]), col]
        return FeatureMatrix(
            values=values,
            names=matrix.names,
            modality=matrix.modality,
            kind=matrix.kind,
            source=matrix.source,
        )
