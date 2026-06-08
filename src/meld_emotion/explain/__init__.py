"""설명(Explanation) 컴포넌트.

새 설명기 추가법은 ``explain/README.md`` 참고. 각 설명기는
:class:`~meld_emotion.core.protocols.Explainer` 를 만족하고
:class:`~meld_emotion.core.results.ExplanationReport` 의 해당 필드만 채운다.
"""

from __future__ import annotations

from meld_emotion.explain.counterfactual import CounterfactualExplainer
from meld_emotion.explain.dialogue_finegrained import DialogueFineGrainedXaiExplainer
from meld_emotion.explain.modality_contribution import ModalityAblationExplainer
from meld_emotion.explain.permutation import PermutationImportanceExplainer

__all__ = [
    "CounterfactualExplainer",
    "DialogueFineGrainedXaiExplainer",
    "ModalityAblationExplainer",
    "PermutationImportanceExplainer",
]
