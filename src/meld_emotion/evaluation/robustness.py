"""강건성 평가: 여러 모달리티 시나리오에 대한 평가 (완전 구현).

각 시나리오로 특징 묶음을 마스킹한 뒤 동일한 평가기로 평가하여, 모달리티 누락 시 성능
하락을 정량화한다(제안서의 robustness 평가).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from meld_emotion.core.features import FeatureBundle
from meld_emotion.core.protocols import Classifier
from meld_emotion.core.results import EvaluationReport, RobustnessReport
from meld_emotion.core.status import real
from meld_emotion.core.types import IntArray
from meld_emotion.evaluation.evaluator import Evaluator
from meld_emotion.fusion.masking import ModalityScenario, mask_bundle

logger = logging.getLogger(__name__)


@real
class RobustnessEvaluator:
    """여러 :class:`ModalityScenario` 에 대해 평가를 반복한다."""

    def __init__(self, evaluator: Evaluator, scenarios: Sequence[ModalityScenario]) -> None:
        self._evaluator = evaluator
        self._scenarios = tuple(scenarios)

    @property
    def scenarios(self) -> tuple[ModalityScenario, ...]:
        return self._scenarios

    def evaluate(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray
    ) -> RobustnessReport:
        logger.info(
            "강건성 평가 실행: scenarios=%s",
            ",".join(scenario.name for scenario in self._scenarios),
        )
        reports: list[EvaluationReport] = []
        for scenario in self._scenarios:
            logger.info("강건성 시나리오 시작: scenario=%s", scenario.name)
            reports.append(
                self._evaluator.evaluate(model, mask_bundle(bundle, scenario), y_true, scenario.name)
            )
            logger.info("강건성 시나리오 완료: scenario=%s", scenario.name)
        return RobustnessReport(reports=tuple(reports))
