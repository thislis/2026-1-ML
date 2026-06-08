"""Fine-grained XAI for the dialogue RNN classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

import numpy as np

from meld_emotion.core.features import (
    FeatureBundle,
    FeatureUnit,
    SequenceFeatureMatrix,
    UtteranceSpec,
)
from meld_emotion.core.protocols import Classifier
from meld_emotion.core.results import (
    DialogueXaiResult,
    ExplanationReport,
    ModalityXaiSummary,
    UnitAttribution,
    UtteranceAttribution,
)
from meld_emotion.core.status import real
from meld_emotion.core.types import IntArray, Modality
from meld_emotion.models.dialogue_rnn import TorchDialogueEmotionClassifier


def _load_integrated_gradients() -> Any:
    try:
        module: Any = import_module("captum.attr")
    except ImportError as exc:
        raise ImportError(
            "DialogueFineGrainedXaiExplainer requires Captum. "
            "Install it with `uv sync --extra xai`."
        ) from exc
    return module.IntegratedGradients


def _load_torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "DialogueFineGrainedXaiExplainer requires PyTorch. "
            "Install it with `uv sync --extra deep`."
        ) from exc


@real
class DialogueFineGrainedXaiExplainer:
    """Token/span/frame and dialogue-level XAI for ``TorchDialogueEmotionClassifier``."""

    def __init__(
        self,
        method: str = "integrated_gradients",
        n_steps: int = 32,
        top_k: int = 10,
        max_targets: int = 32,
        target: str = "predicted",
    ) -> None:
        if method != "integrated_gradients":
            raise ValueError("v1 dialogue_finegrained_xai 는 integrated_gradients 만 지원합니다")
        if n_steps <= 0:
            raise ValueError("n_steps 는 양수여야 합니다")
        if top_k <= 0:
            raise ValueError("top_k 는 양수여야 합니다")
        if max_targets <= 0:
            raise ValueError("max_targets 는 양수여야 합니다")
        if target not in {"predicted", "gold"}:
            raise ValueError("target 은 'predicted' 또는 'gold' 여야 합니다")
        self._n_steps = n_steps
        self._top_k = top_k
        self._max_targets = max_targets
        self._target = target

    def explain(
        self, model: Classifier, bundle: FeatureBundle, y_true: IntArray
    ) -> ExplanationReport:
        if not isinstance(model, TorchDialogueEmotionClassifier):
            raise TypeError("dialogue_finegrained_xai 는 dialogue_rnn 모델에서만 사용할 수 있습니다")
        ig_cls = _load_integrated_gradients()
        torch = _load_torch()
        prediction = model.predict(bundle)
        arrays = model.xai_arrays(bundle)
        torch_model = model.xai_model()
        torch_model.eval()
        flat_to_slot = {flat: (dialogue, slot) for flat, dialogue, slot in arrays.flat_indices}
        utterances = _utterances_or_fallback(bundle)

        results: list[DialogueXaiResult] = []
        for flat_idx in range(min(bundle.n_samples, self._max_targets)):
            dialogue_idx, slot = flat_to_slot[flat_idx]
            batch = model.xai_tensor_batch(arrays, (dialogue_idx,))
            target_class_idx = self._target_class(flat_idx, prediction.y_pred, y_true)
            target_slot = slot
            target_idx = target_class_idx
            target_class = prediction.classes[target_class_idx]

            def score_forward(
                text_x: Any,
                audio_x: Any,
                video_x: Any,
                speaker_id: Any,
                utterance_mask: Any,
                text_mask: Any,
                audio_mask: Any,
                video_mask: Any,
                modality_mask: Any,
                target_slot: int = target_slot,
                target_idx: int = target_idx,
            ) -> Any:
                output = torch_model(
                    text_x,
                    audio_x,
                    video_x,
                    speaker_id,
                    utterance_mask,
                    text_mask,
                    audio_mask,
                    video_mask,
                    modality_mask,
                )
                return output["logits"][:, target_slot, target_idx]

            inputs = (
                batch["text_x"].clone().detach().requires_grad_(True),
                batch["audio_x"].clone().detach().requires_grad_(True),
                batch["video_x"].clone().detach().requires_grad_(True),
            )
            ig = ig_cls(score_forward)
            attr_text, attr_audio, attr_video = ig.attribute(
                inputs=inputs,
                baselines=tuple(torch.zeros_like(x) for x in inputs),
                additional_forward_args=(
                    batch["speaker_id"],
                    batch["utterance_mask"],
                    batch["text_mask"],
                    batch["audio_mask"],
                    batch["video_mask"],
                    batch["modality_mask"],
                ),
                n_steps=self._n_steps,
            )
            with torch.no_grad():
                output = torch_model(
                    batch["text_x"],
                    batch["audio_x"],
                    batch["video_x"],
                    batch["speaker_id"],
                    batch["utterance_mask"],
                    batch["text_mask"],
                    batch["audio_mask"],
                    batch["video_mask"],
                    batch["modality_mask"],
                    return_xai=True,
                )

            attrs = {
                Modality.TEXT: np.abs(attr_text.detach().cpu().numpy()[0]),
                Modality.AUDIO: np.abs(attr_audio.detach().cpu().numpy()[0]),
                Modality.VIDEO: np.abs(attr_video.detach().cpu().numpy()[0]),
            }
            modality_scores = {
                modality: _masked_attr_sum(attrs[modality], batch[f"{modality.value}_mask"][0])
                for modality in (Modality.TEXT, Modality.AUDIO, Modality.VIDEO)
            }
            total_modality = sum(modality_scores.values())
            gate = output["modality_gate"].detach().cpu().numpy()[0, slot]
            target_logit = float(output["logits"].detach().cpu().numpy()[0, slot, target_class_idx])
            modality_summary = tuple(
                ModalityXaiSummary(
                    modality=modality,
                    available=_available(batch, slot, idx),
                    gate=float(gate[idx]) if _available(batch, slot, idx) else None,
                    attribution_share=(
                        float(modality_scores[modality] / total_modality)
                        if total_modality > 0.0
                        else 0.0
                    ),
                    ablation_delta_logit=self._modality_delta(
                        torch_model,
                        batch,
                        slot,
                        target_class_idx,
                        idx,
                        target_logit,
                    ),
                )
                for idx, modality in enumerate((Modality.TEXT, Modality.AUDIO, Modality.VIDEO))
            )
            block_deltas = self._block_deltas(
                torch_model, batch, slot, target_class_idx, target_logit
            )
            dialogue_utterances = self._utterance_importance(
                attrs,
                arrays.flat_indices,
                dialogue_idx,
                utterances,
                output["memory_attention"].detach().cpu().numpy()[0],
                slot,
            )
            result = DialogueXaiResult(
                uid=bundle.uids[flat_idx],
                dialogue_id=utterances[flat_idx].dialogue_id,
                utterance_id=utterances[flat_idx].utterance_id,
                speaker=utterances[flat_idx].speaker,
                pred_class=prediction.classes[int(prediction.y_pred[flat_idx])],
                pred_proba=float(prediction.proba[flat_idx, int(prediction.y_pred[flat_idx])]),
                target_class=target_class,
                target_logit=target_logit,
                modality=modality_summary,
                utterances=tuple(dialogue_utterances[: self._top_k]),
                classifier_blocks=block_deltas,
                top_text_units=self._top_units(
                    bundle, Modality.TEXT, attrs[Modality.TEXT], arrays.flat_indices, dialogue_idx
                ),
                top_audio_units=self._top_units(
                    bundle, Modality.AUDIO, attrs[Modality.AUDIO], arrays.flat_indices, dialogue_idx
                ),
                top_video_units=self._top_units(
                    bundle, Modality.VIDEO, attrs[Modality.VIDEO], arrays.flat_indices, dialogue_idx
                ),
                text_dimension_attribution=self._top_dimensions(attrs[Modality.TEXT]),
                audio_dimension_attribution=self._top_dimensions(attrs[Modality.AUDIO]),
                video_dimension_attribution=self._top_dimensions(attrs[Modality.VIDEO]),
            )
            results.append(result)
        return ExplanationReport(dialogue_xai=tuple(results))

    def _target_class(self, flat_idx: int, y_pred: IntArray, y_true: IntArray) -> int:
        if self._target == "gold":
            return int(y_true[flat_idx])
        return int(y_pred[flat_idx])

    def _modality_delta(
        self,
        torch_model: Any,
        batch: Mapping[str, Any],
        slot: int,
        target_class_idx: int,
        modality_index: int,
        target_logit: float,
    ) -> float | None:
        if bool(batch["modality_mask"][0, slot, modality_index].detach().cpu().item() <= 0.0):
            return None
        mask = batch["modality_mask"].clone()
        mask[:, :, modality_index] = 0.0
        output = torch_model(
            batch["text_x"],
            batch["audio_x"],
            batch["video_x"],
            batch["speaker_id"],
            batch["utterance_mask"],
            batch["text_mask"],
            batch["audio_mask"],
            batch["video_mask"],
            mask,
        )
        return target_logit - float(output["logits"].detach().cpu().numpy()[0, slot, target_class_idx])

    def _block_deltas(
        self,
        torch_model: Any,
        batch: Mapping[str, Any],
        slot: int,
        target_class_idx: int,
        target_logit: float,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for block in ("fused", "context", "memory"):
            output = torch_model(
                batch["text_x"],
                batch["audio_x"],
                batch["video_x"],
                batch["speaker_id"],
                batch["utterance_mask"],
                batch["text_mask"],
                batch["audio_mask"],
                batch["video_mask"],
                batch["modality_mask"],
                ablate_classifier_block=block,
            )
            result[block] = target_logit - float(
                output["logits"].detach().cpu().numpy()[0, slot, target_class_idx]
            )
        return result

    def _utterance_importance(
        self,
        attrs: Mapping[Modality, np.ndarray],
        flat_indices: Sequence[tuple[int, int, int]],
        dialogue_idx: int,
        utterances: Sequence[UtteranceSpec],
        memory_attention: np.ndarray,
        target_slot: int,
    ) -> list[UtteranceAttribution]:
        rows = [(flat, slot) for flat, dialogue, slot in flat_indices if dialogue == dialogue_idx]
        scores = []
        for flat, slot in rows:
            score = sum(float(attrs[modality][slot].sum()) for modality in attrs)
            scores.append((flat, slot, score))
        total = sum(score for _, _, score in scores)
        result = [
            UtteranceAttribution(
                uid=utterances[flat].uid,
                dialogue_id=utterances[flat].dialogue_id,
                utterance_id=utterances[flat].utterance_id,
                speaker=utterances[flat].speaker,
                score=score,
                share=score / total if total > 0.0 else 0.0,
                memory_attention=float(memory_attention[target_slot, slot]),
            )
            for flat, slot, score in scores
        ]
        return sorted(result, key=lambda item: item.score, reverse=True)

    def _top_units(
        self,
        bundle: FeatureBundle,
        modality: Modality,
        attr: np.ndarray,
        flat_indices: Sequence[tuple[int, int, int]],
        dialogue_idx: int,
    ) -> tuple[UnitAttribution, ...]:
        sequence = _first_sequence(bundle, modality)
        rows = [(flat, slot) for flat, dialogue, slot in flat_indices if dialogue == dialogue_idx]
        scored: list[UnitAttribution] = []
        for flat, slot in rows:
            units = _units_for(sequence, flat)
            scores = attr[slot].sum(axis=-1)
            for index, score in enumerate(scores.tolist()):
                if score <= 0.0:
                    continue
                unit = units[index] if index < len(units) else FeatureUnit(label=f"unit_{index}", index=index)
                scored.append(
                    UnitAttribution(
                        label=f"{bundle.uids[flat]}:{unit.label}",
                        score=float(score),
                        index=unit.index,
                        start=unit.start,
                        end=unit.end,
                        char_start=unit.char_start,
                        char_end=unit.char_end,
                        available=sequence is not None,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[: self._top_k])

    def _top_dimensions(self, attr: np.ndarray) -> tuple[UnitAttribution, ...]:
        scores = attr.sum(axis=(0, 1))
        indices = np.argsort(scores)[::-1][: self._top_k]
        return tuple(
            UnitAttribution(label=f"dim_{int(idx)}", score=float(scores[idx]), index=int(idx))
            for idx in indices
            if float(scores[idx]) > 0.0
        )


def _masked_attr_sum(attr: np.ndarray, mask_tensor: Any) -> float:
    mask = np.asarray(mask_tensor.detach().cpu().numpy(), dtype=np.float64)
    return float((attr * mask[..., None]).sum())


def _available(batch: Mapping[str, Any], slot: int, modality_index: int) -> bool:
    return bool(batch["modality_mask"][0, slot, modality_index].detach().cpu().item() > 0.0)


def _first_sequence(bundle: FeatureBundle, modality: Modality) -> SequenceFeatureMatrix | None:
    matrices = bundle.sequence_by_modality(modality)
    return matrices[0] if matrices else None


def _units_for(matrix: SequenceFeatureMatrix | None, flat_idx: int) -> tuple[FeatureUnit, ...]:
    if matrix is None:
        return ()
    return matrix.units[flat_idx]


def _utterances_or_fallback(bundle: FeatureBundle) -> tuple[UtteranceSpec, ...]:
    if bundle.utterances:
        return bundle.utterances
    return tuple(
        UtteranceSpec(uid=uid, dialogue_id=i, utterance_id=0, speaker="")
        for i, uid in enumerate(bundle.uids)
    )
