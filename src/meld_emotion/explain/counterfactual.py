"""반사실(counterfactual) 증거 제거 설명기.

특징 공간 버전(완전 구현): 각 샘플에서 평균에서 가장 많이 벗어난 개념 특징(=강한 증거)을
train 평균값으로 되돌렸을 때 예측 확률이 어떻게 변하는지 측정한다. 원문 단어 삭제 버전은
별도 메서드로 두며 아직 미구현이다(텍스트 재추출 필요).
"""

from __future__ import annotations

import numpy as np

from meld_emotion.core.features import FeatureBundle, FeatureMatrix
from meld_emotion.core.protocols import Classifier
from meld_emotion.core.results import CounterfactualResult, ExplanationReport
from meld_emotion.core.status import real
from meld_emotion.core.types import FeatureKind


@real
class CounterfactualExplainer:
    """개념 특징을 평균으로 되돌리는 반사실 설명."""

    def __init__(self, top_k: int = 5, sample_limit: int = 20) -> None:
        self._top_k = top_k
        self._sample_limit = sample_limit

    def explain(
        self, model: Classifier, bundle: FeatureBundle, y_true: np.ndarray
    ) -> ExplanationReport:
        concept_indices = [
            i for i, m in enumerate(bundle.matrices) if m.kind == FeatureKind.CONCEPT
        ]
        if not concept_indices:
            return ExplanationReport()

        means = {i: bundle.matrices[i].values.mean(axis=0) for i in concept_indices}
        n_targets = min(self._sample_limit, bundle.n_samples)
        results: list[CounterfactualResult] = []

        for row in range(n_targets):
            one = bundle.select([row])
            original = model.predict_proba(one)[0]
            top = self._top_evidence(bundle, concept_indices, means, row)
            modified_bundle = self._apply(one, means, top)
            modified = model.predict_proba(modified_bundle)[0]
            results.append(
                CounterfactualResult(
                    uid=bundle.uids[row],
                    original_proba=original,
                    modified_proba=modified,
                    removed=tuple(name for _, _, name in top),
                )
            )
        return ExplanationReport(counterfactuals=tuple(results))

    def _top_evidence(
        self,
        bundle: FeatureBundle,
        concept_indices: list[int],
        means: dict[int, np.ndarray],
        row: int,
    ) -> list[tuple[int, int, str]]:
        scored: list[tuple[float, int, int, str]] = []
        for mat_idx in concept_indices:
            matrix = bundle.matrices[mat_idx]
            for col in range(matrix.n_features):
                deviation = abs(float(matrix.values[row, col]) - float(means[mat_idx][col]))
                scored.append((deviation, mat_idx, col, matrix.names[col]))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(mi, c, name) for _, mi, c, name in scored[: self._top_k]]

    @staticmethod
    def _apply(
        one_row: FeatureBundle,
        means: dict[int, np.ndarray],
        top: list[tuple[int, int, str]],
    ) -> FeatureBundle:
        matrices = [m.values.copy() for m in one_row.matrices]
        for mat_idx, col, _ in top:
            matrices[mat_idx][0, col] = means[mat_idx][col]
        rebuilt = tuple(
            FeatureMatrix(
                values=matrices[i],
                names=m.names,
                modality=m.modality,
                kind=m.kind,
                source=m.source,
            )
            for i, m in enumerate(one_row.matrices)
        )
        return FeatureBundle(uids=one_row.uids, matrices=rebuilt, availability=one_row.availability)

    def explain_text_deletion(
        self, model: Classifier, bundle: FeatureBundle, y_true: np.ndarray
    ) -> ExplanationReport:
        """원문에서 중요한 단어를 삭제하는 반사실 (미구현 — 텍스트 재추출 필요)."""

        raise NotImplementedError(
            "원문 단어 삭제 반사실 미구현 — 토큰 제거 후 텍스트 특징 재추출 파이프라인 필요"
        )
