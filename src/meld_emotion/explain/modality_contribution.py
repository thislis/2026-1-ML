"""모달리티 기여도 설명기 (완전 구현).

각 모달리티를 제거(ablation)했을 때의 성능 하락폭으로 모달리티별 기여도를 측정한다
(제안서의 modality-wise contribution scores).
"""

from __future__ import annotations

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier, Metric
from meld_emotion.core.results import ExplanationReport, ModalityContribution
from meld_emotion.core.status import real
from meld_emotion.core.types import IntArray
from meld_emotion.fusion.masking import ModalityScenario, mask_bundle


@real
class ModalityAblationExplainer:
    """모달리티 제거 기반 기여도."""

    def __init__(self, metric: Metric) -> None:
        self._metric = metric

    def _score(self, model: Classifier, bundle: FeatureBundle, y_true: IntArray) -> float:
        return self._metric.compute(y_true, model.predict(bundle)).value

    def explain(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray
    ) -> ExplanationReport:
        present = set(bundle.modalities)
        baseline = self._score(model, bundle, y_true)
        contributions: list[ModalityContribution] = []
        for modality in bundle.modalities:
            scenario = ModalityScenario(
                name=f"no_{modality.value}", available=frozenset(present - {modality})
            )
            ablated = self._score(model, mask_bundle(bundle, scenario), y_true)
            contributions.append(
                ModalityContribution(
                    modality=modality,
                    score_drop=baseline - ablated,
                    baseline_score=baseline,
                    ablated_score=ablated,
                )
            )
        return ExplanationReport(modality_contributions=tuple(contributions))
