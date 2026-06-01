"""융합 전략과 모달리티 마스킹.

새 융합/결합기/시나리오 추가법은 ``fusion/README.md`` 참고. Early/Late fusion 은 모두
:class:`~meld_emotion.core.protocols.Classifier` 를 만족한다.
"""

from __future__ import annotations

from meld_emotion.fusion.combiners import (
    MeanCombiner,
    ProbabilityCombiner,
    StackingCombiner,
    WeightedCombiner,
)
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.late import LateFusionClassifier
from meld_emotion.fusion.masking import (
    SCENARIOS,
    ModalityDropout,
    ModalityScenario,
    get_scenario,
    mask_bundle,
)

__all__ = [
    "SCENARIOS",
    "EarlyFusionClassifier",
    "LateFusionClassifier",
    "MeanCombiner",
    "ModalityDropout",
    "ModalityScenario",
    "ProbabilityCombiner",
    "StackingCombiner",
    "WeightedCombiner",
    "get_scenario",
    "mask_bundle",
]
