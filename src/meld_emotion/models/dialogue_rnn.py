"""Pipeline wrapper for the PyTorch dialogue RNN emotion model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from meld_emotion.config.schema import DialogueRnnConfig
from meld_emotion.core.features import FeatureBundle, UtteranceSpec
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import EMOTION_ORDER, Emotion, FloatArray, IntArray, Modality
from meld_emotion.models.calibration import (
    CalibrationParams,
    PredictionPostprocessor,
    fit_temperature,
    tune_class_thresholds,
    tune_neutral_emotion_threshold,
)
from meld_emotion.models.losses import compute_dialogue_loss, false_positive_counts
from meld_emotion.models.multimodal_emotion_model import MultimodalEmotionModel


def _require_torch() -> None:
    """Keep the installation guidance local to the deep model boundary."""

    try:
        torch.empty(0)
    except Exception as exc:  # pragma: no cover - import/runtime environment dependent
        raise ImportError(
            "PyTorch 가 필요합니다. `uv sync --extra deep` (또는 --extra all) 로 설치하세요."
        ) from exc


@dataclass(frozen=True)
class _DialogueArrays:
    text_x: np.ndarray
    audio_x: np.ndarray
    video_x: np.ndarray
    text_mask: np.ndarray
    audio_mask: np.ndarray
    video_mask: np.ndarray
    modality_mask: np.ndarray
    speaker_id: np.ndarray
    utterance_mask: np.ndarray
    labels: np.ndarray
    flat_indices: tuple[tuple[int, int, int], ...]


@real
class TorchDialogueEmotionClassifier:
    """Classifier adapter for the dialogue-level PyTorch model."""

    def __init__(
        self,
        config: DialogueRnnConfig,
        classes: tuple[Emotion, ...] = EMOTION_ORDER,
    ) -> None:
        _require_torch()
        self._config = config
        self._classes = classes
        self._model: MultimodalEmotionModel | None = None
        self._speaker_to_id: dict[str, int] = {}
        self._dims: dict[Modality, int] = {}
        self._device = torch.device(config.training.device)
        self._last_false_positive_counts = np.zeros(len(classes), dtype=np.int64)
        self._last_gate_stats: dict[str, float] = {}
        self._postprocessor = PredictionPostprocessor()

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | None = None
    ) -> TorchDialogueEmotionClassifier:
        """저장된 dialogue_rnn checkpoint 로부터 추론 가능한 classifier 를 복원한다."""

        _require_torch()
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 를 찾을 수 없습니다: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"checkpoint 형식이 올바르지 않습니다: {checkpoint_path}")

        config = _config_from_checkpoint(_require_mapping(checkpoint, "config"))
        if device is not None:
            config = replace(config, training=replace(config.training, device=device))
        classes = _classes_from_checkpoint(checkpoint.get("classes"))
        classifier = cls(config, classes)
        classifier._dims = _dims_from_checkpoint(_require_mapping(checkpoint, "dims"))
        classifier._speaker_to_id = _speaker_vocab_from_checkpoint(
            _require_mapping(checkpoint, "speaker_to_id")
        )

        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint 에 model_state_dict 가 없습니다")
        model = classifier._build_model().to(classifier._device)
        missing, unexpected = model.load_state_dict(dict(state), strict=False)
        allowed_missing_prefixes = (
            "text_aux_head.",
            "audio_aux_head.",
            "video_aux_head.",
            "classifier.utterance_head.",
            "classifier.context_head.",
            "classifier.memory_head.",
            "classifier.residual_gate.",
        )
        unexpected_keys = list(unexpected)
        missing_keys = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
        if unexpected_keys or missing_keys:
            raise ValueError(
                "checkpoint state_dict is incompatible: "
                f"missing={missing_keys} unexpected={unexpected_keys}"
            )
        classifier._model = model
        calibration = checkpoint.get("calibration")
        if isinstance(calibration, Mapping):
            params = CalibrationParams.from_dict(calibration)
            if params.class_labels and params.class_labels != tuple(c.value for c in classes):
                raise ValueError("checkpoint calibration class order does not match classes")
            classifier._postprocessor = PredictionPostprocessor(params)
        return classifier

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        return self._fit(bundle, y)

    def fit_with_distillation(
        self,
        bundle: FeatureBundle,
        y: IntArray,
        teacher_probs: FloatArray,
        *,
        temperature: float,
        weight: float,
    ) -> Self:
        """Fit with KL(teacher_probs, neural_probs) added to the supervised loss."""

        teacher = np.asarray(teacher_probs, dtype=np.float32)
        expected_shape = (bundle.n_samples, len(self._classes))
        if teacher.shape != expected_shape:
            raise ValueError(f"teacher_probs shape mismatch: {teacher.shape} != {expected_shape}")
        if temperature <= 0.0:
            raise ValueError("distillation temperature must be > 0")
        if weight < 0.0:
            raise ValueError("distillation weight must be >= 0")
        row_sums = teacher.sum(axis=1)
        if not np.all(np.isfinite(teacher)) or np.any(teacher < 0.0):
            raise ValueError("teacher_probs must contain finite non-negative values")
        if np.any(row_sums <= 0.0):
            raise ValueError("teacher_probs rows must have positive mass")
        teacher = teacher / row_sums[:, None]
        return self._fit(
            bundle,
            y,
            teacher_probs=teacher,
            distillation_temperature=float(temperature),
            distillation_weight=float(weight),
        )

    def _fit(
        self,
        bundle: FeatureBundle,
        y: IntArray,
        teacher_probs: FloatArray | None = None,
        distillation_temperature: float = 1.0,
        distillation_weight: float = 0.0,
    ) -> Self:
        if len(y) != bundle.n_samples:
            raise ValueError(f"y 길이가 샘플 수와 다릅니다: {len(y)} != {bundle.n_samples}")
        torch.manual_seed(self._config.training.seed)
        rng = np.random.default_rng(self._config.training.seed)

        self._dims = self._infer_dims(bundle)
        self._speaker_to_id = self._build_speaker_vocab(bundle)
        arrays = self._build_arrays(bundle, y)
        teacher_arrays = (
            self._teacher_prob_arrays(arrays, teacher_probs)
            if teacher_probs is not None and distillation_weight > 0.0
            else None
        )
        self._model = self._build_model().to(self._device)

        train_dialogues, val_dialogues = self._split_dialogues(arrays.text_x.shape[0], rng)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.training.lr,
            weight_decay=self._config.training.weight_decay,
        )
        class_weights = self._class_weights(y).to(self._device)
        class_counts = torch.as_tensor(
            self._class_counts(y),
            dtype=torch.float32,
            device=self._device,
        )

        best_state: Mapping[str, torch.Tensor] | None = None
        best_score = -1.0
        stale = 0
        for epoch in range(1, self._config.training.max_epochs + 1):
            self._model.train()
            for indices in self._iter_batches(train_dialogues, shuffle=True, rng=rng):
                batch = self._tensor_batch(arrays, indices)
                mask = self._drop_modalities(batch["modality_mask"])
                output = self._model(
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
                logits = output["logits"]
                labels = batch["labels"].clone()
                labels[batch["utterance_mask"] == 0] = -100
                loss = compute_dialogue_loss(
                    logits,
                    labels,
                    settings=self._config.loss,
                    class_counts=class_counts,
                    class_weights=class_weights,
                )
                loss = loss + self._gate_entropy_loss(
                    output["modality_gate"], batch["utterance_mask"]
                )
                loss = loss + self._auxiliary_loss(
                    output, labels, batch, class_counts, class_weights
                )
                if teacher_arrays is not None:
                    teacher_batch = torch.as_tensor(
                        teacher_arrays[np.asarray(indices, dtype=np.int64)],
                        dtype=torch.float32,
                        device=self._device,
                    )
                    loss = loss + self._distillation_loss(
                        logits,
                        teacher_batch,
                        batch["utterance_mask"],
                        temperature=distillation_temperature,
                        weight=distillation_weight,
                    )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.training.gradient_clip_norm,
                )
                optimizer.step()

            score_dialogues = val_dialogues if val_dialogues else train_dialogues
            score = self._validation_score(arrays, score_dialogues)
            if score > best_score:
                best_score = score
                best_state = self._state_dict_cpu(self._model)
                stale = 0
                self._save_best_checkpoint(epoch, best_score, best_state, bool(val_dialogues))
            elif val_dialogues:
                stale += 1
                if stale >= self._config.training.early_stopping_patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        if self._config.calibration.enabled:
            self._fit_calibration(arrays, val_dialogues if val_dialogues else train_dialogues)
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        logits = self._predict_logits(bundle)
        probs = self._postprocessor.probabilities(
            torch.as_tensor(logits, dtype=torch.float32, device=self._device)
        )
        return np.asarray(probs.detach().cpu().numpy(), dtype=np.float64)

    def _predict_logits(self, bundle: FeatureBundle) -> FloatArray:
        model = self._require_model()
        arrays = self._build_arrays(bundle, None)
        logits_out = np.zeros((bundle.n_samples, len(self._classes)), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for indices in self._iter_batches(tuple(range(arrays.text_x.shape[0])), shuffle=False):
                batch = self._tensor_batch(arrays, indices)
                output = model(
                    batch["text_x"],
                    batch["audio_x"],
                    batch["video_x"],
                    batch["speaker_id"],
                    batch["utterance_mask"],
                    batch["text_mask"],
                    batch["audio_mask"],
                    batch["video_mask"],
                    batch["modality_mask"],
                )
                logits = output["logits"].cpu().numpy()
                for local_b, dialogue_idx in enumerate(indices):
                    for flat_idx, slot_b, slot_n in arrays.flat_indices:
                        if slot_b == dialogue_idx:
                            logits_out[flat_idx] = logits[local_b, slot_n]
        return logits_out

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        y_pred = self._postprocessor.predict(proba)
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def xai_arrays(self, bundle: FeatureBundle) -> _DialogueArrays:
        """Build dialogue tensors for explainers without labels."""

        self._require_model()
        return self._build_arrays(bundle, None)

    def xai_tensor_batch(
        self, arrays: _DialogueArrays, dialogue_indices: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        """Return a tensor batch on the classifier device for XAI code."""

        return self._tensor_batch(arrays, dialogue_indices)

    def xai_model(self) -> MultimodalEmotionModel:
        """Return the trained PyTorch model for XAI forward passes."""

        return self._require_model()

    @property
    def last_false_positive_counts(self) -> IntArray:
        """False-positive counts by predicted class from the most recent validation scoring."""

        return np.asarray(self._last_false_positive_counts, dtype=np.int64)

    @property
    def last_gate_stats(self) -> Mapping[str, float]:
        """Latest gate mean/variance/entropy summary."""

        return dict(self._last_gate_stats)

    @property
    def calibration_params(self) -> Mapping[str, Any]:
        """Latest prediction postprocessor parameters for artifact metadata."""

        return self._postprocessor.params.to_dict()

    @property
    def requires_sequence_features(self) -> bool:
        """Whether this checkpoint/config rejects pooled fallback features."""

        return self._sequence_fallback_policy() == "error"

    def _build_model(self) -> MultimodalEmotionModel:
        enc = self._config.modality_encoder
        fusion = self._config.fusion
        context = self._config.dialogue_context
        memory = self._config.memory_attention
        head = self._config.classifier
        return MultimodalEmotionModel(
            text_input_dim=self._model_input_dim(Modality.TEXT),
            audio_input_dim=self._model_input_dim(Modality.AUDIO),
            video_input_dim=self._model_input_dim(Modality.VIDEO),
            speaker_vocab_size=max(self._speaker_to_id.values(), default=0) + 1,
            num_classes=len(self._classes),
            rnn_type=self._config.rnn_type,
            modality_encoder_type=enc.encoder_type,
            modality_proj_dim=enc.proj_dim,
            modality_hidden_dim=enc.hidden_dim,
            modality_num_layers=enc.num_layers,
            modality_num_heads=enc.num_heads,
            modality_conv_kernel_size=enc.conv_kernel_size,
            modality_ffn_multiplier=enc.ffn_multiplier,
            modality_dropout=enc.dropout,
            modality_attention_dropout=enc.attention_dropout,
            modality_pooling_type=enc.pooling_type,
            fusion_dim=fusion.fusion_dim,
            fusion_dropout=fusion.dropout,
            use_gated_fusion=fusion.use_gated_fusion,
            use_interaction_features=fusion.use_interaction_features and fusion.use_interaction,
            speaker_emb_dim=context.speaker_emb_dim,
            context_hidden_dim=context.hidden_dim,
            context_num_layers=context.num_layers,
            context_dropout=context.dropout,
            memory_enabled=memory.enabled,
            memory_attn_dim=memory.attn_dim,
            use_rope=memory.use_rope,
            use_relative_distance_bias=memory.use_relative_distance_bias,
            use_same_speaker_bias=memory.use_same_speaker_bias,
            max_relative_distance=memory.max_relative_distance,
            classifier_hidden_dim=head.hidden_dim,
            classifier_dropout=head.dropout,
            classifier_head_type=head.classifier_head_type,
            classifier_use_context=head.use_context and context.use_context,
            classifier_use_memory=head.use_memory and memory.use_memory and memory.enabled,
            classifier_gate_hidden_dim=head.gate_hidden_dim,
            classifier_gate_dropout=head.gate_dropout,
            context_state_dropout=self._config.training.context_dropout,
            input_mode=self._input_mode(),
        )

    def _model_input_dim(self, modality: Modality) -> int:
        if self._input_mode() == "multimodal":
            return self._dims[Modality.MULTIMODAL] if modality == Modality.TEXT else 1
        return self._dims[modality]

    def _infer_dims(self, bundle: FeatureBundle) -> dict[Modality, int]:
        if self._input_mode() == "multimodal":
            actual = sum(m.n_features for m in bundle.by_modality(Modality.MULTIMODAL))
            config_dim = self._config.modality_encoder.text_input_dim
            if actual == 0:
                raise ValueError(
                    "dialogue_rnn input_mode='multimodal' requires a multimodal feature matrix"
                )
            if config_dim > 0 and config_dim != actual:
                raise ValueError(
                    f"multimodal feature dim mismatch: config={config_dim}, actual={actual}"
                )
            return {Modality.MULTIMODAL: actual}
        configured = {
            Modality.TEXT: self._config.modality_encoder.text_input_dim,
            Modality.AUDIO: self._config.modality_encoder.audio_input_dim,
            Modality.VIDEO: self._config.modality_encoder.video_input_dim,
        }
        dims: dict[Modality, int] = {}
        for modality, config_dim in configured.items():
            sequence = bundle.sequence_by_modality(modality)
            if not sequence and self._sequence_fallback_policy() == "error":
                raise ValueError(
                    f"{modality.value} sequence feature is required by "
                    "modality_encoder.sequence_fallback_policy='error'"
                )
            actual = (
                sum(m.n_features for m in sequence)
                if sequence
                else sum(m.n_features for m in bundle.by_modality(modality))
            )
            if actual == 0:
                dims[modality] = config_dim if config_dim > 0 else 1
            elif config_dim > 0 and config_dim != actual:
                raise ValueError(
                    f"{modality.value} feature dim mismatch: config={config_dim}, actual={actual}"
                )
            else:
                dims[modality] = actual
        return dims

    def _build_arrays(self, bundle: FeatureBundle, y: IntArray | None) -> _DialogueArrays:
        utterances = _utterances_or_fallback(bundle)
        groups = _dialogue_groups(utterances)
        max_len = max((len(indices) for indices in groups), default=1)
        n_dialogues = len(groups)
        if self._input_mode() == "multimodal":
            return self._build_multimodal_arrays(bundle, y, utterances, groups, max_len, n_dialogues)

        text_values, text_row_mask = self._modality_sequence_values(bundle, Modality.TEXT)
        audio_values, audio_row_mask = self._modality_sequence_values(bundle, Modality.AUDIO)
        video_values, video_row_mask = self._modality_sequence_values(bundle, Modality.VIDEO)
        text_len = text_values.shape[1]
        audio_len = audio_values.shape[1]
        video_len = video_values.shape[1]
        text_x = np.zeros(
            (n_dialogues, max_len, text_len, self._dims[Modality.TEXT]), dtype=np.float32
        )
        audio_x = np.zeros(
            (n_dialogues, max_len, audio_len, self._dims[Modality.AUDIO]), dtype=np.float32
        )
        video_x = np.zeros(
            (n_dialogues, max_len, video_len, self._dims[Modality.VIDEO]), dtype=np.float32
        )
        text_mask = np.zeros((n_dialogues, max_len, text_len), dtype=np.float32)
        audio_mask = np.zeros((n_dialogues, max_len, audio_len), dtype=np.float32)
        video_mask = np.zeros((n_dialogues, max_len, video_len), dtype=np.float32)
        modality_mask = np.zeros((n_dialogues, max_len, 3), dtype=np.float32)
        speaker_id = np.zeros((n_dialogues, max_len), dtype=np.int64)
        utterance_mask = np.zeros((n_dialogues, max_len), dtype=np.float32)
        labels = np.full((n_dialogues, max_len), -100, dtype=np.int64)
        flat_indices: list[tuple[int, int, int]] = []

        for dialogue_idx, indices in enumerate(groups):
            for slot, flat_idx in enumerate(indices):
                text_x[dialogue_idx, slot] = text_values[flat_idx]
                audio_x[dialogue_idx, slot] = audio_values[flat_idx]
                video_x[dialogue_idx, slot] = video_values[flat_idx]
                text_mask[dialogue_idx, slot] = text_row_mask[flat_idx]
                audio_mask[dialogue_idx, slot] = audio_row_mask[flat_idx]
                video_mask[dialogue_idx, slot] = video_row_mask[flat_idx]
                utterance_mask[dialogue_idx, slot] = 1.0
                modality_mask[dialogue_idx, slot] = self._availability_row(bundle, flat_idx)
                speaker_id[dialogue_idx, slot] = self._speaker_to_id.get(
                    utterances[flat_idx].speaker,
                    0,
                )
                if y is not None:
                    labels[dialogue_idx, slot] = int(y[flat_idx])
                flat_indices.append((flat_idx, dialogue_idx, slot))

        return _DialogueArrays(
            text_x=text_x,
            audio_x=audio_x,
            video_x=video_x,
            text_mask=text_mask,
            audio_mask=audio_mask,
            video_mask=video_mask,
            modality_mask=modality_mask,
            speaker_id=speaker_id,
            utterance_mask=utterance_mask,
            labels=labels,
            flat_indices=tuple(flat_indices),
        )

    def _build_multimodal_arrays(
        self,
        bundle: FeatureBundle,
        y: IntArray | None,
        utterances: Sequence[UtteranceSpec],
        groups: Sequence[Sequence[int]],
        max_len: int,
        n_dialogues: int,
    ) -> _DialogueArrays:
        dim = self._dims[Modality.MULTIMODAL]
        values = self._modality_values(bundle, Modality.MULTIMODAL)
        text_x = np.zeros((n_dialogues, max_len, 1, dim), dtype=np.float32)
        audio_x = np.zeros((n_dialogues, max_len, 1, 1), dtype=np.float32)
        video_x = np.zeros((n_dialogues, max_len, 1, 1), dtype=np.float32)
        text_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        audio_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        video_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        modality_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        speaker_id = np.zeros((n_dialogues, max_len), dtype=np.int64)
        utterance_mask = np.zeros((n_dialogues, max_len), dtype=np.float32)
        labels = np.full((n_dialogues, max_len), -100, dtype=np.int64)
        flat_indices: list[tuple[int, int, int]] = []

        for dialogue_idx, indices in enumerate(groups):
            for slot, flat_idx in enumerate(indices):
                text_x[dialogue_idx, slot, 0] = values[flat_idx]
                text_mask[dialogue_idx, slot, 0] = 1.0
                modality_mask[dialogue_idx, slot, 0] = self._multimodal_availability_row(
                    bundle, flat_idx
                )
                utterance_mask[dialogue_idx, slot] = 1.0
                speaker_id[dialogue_idx, slot] = self._speaker_to_id.get(
                    utterances[flat_idx].speaker,
                    0,
                )
                if y is not None:
                    labels[dialogue_idx, slot] = int(y[flat_idx])
                flat_indices.append((flat_idx, dialogue_idx, slot))

        return _DialogueArrays(
            text_x=text_x,
            audio_x=audio_x,
            video_x=video_x,
            text_mask=text_mask,
            audio_mask=audio_mask,
            video_mask=video_mask,
            modality_mask=modality_mask,
            speaker_id=speaker_id,
            utterance_mask=utterance_mask,
            labels=labels,
            flat_indices=tuple(flat_indices),
        )

    def _modality_values(self, bundle: FeatureBundle, modality: Modality) -> np.ndarray:
        matrices = bundle.by_modality(modality)
        if matrices:
            values = np.concatenate([m.values for m in matrices], axis=1)
        else:
            values = np.zeros((bundle.n_samples, self._dims[modality]), dtype=np.float64)
        if values.shape[1] != self._dims[modality]:
            raise ValueError(
                f"{modality.value} feature dim changed: {values.shape[1]} != {self._dims[modality]}"
            )
        return np.asarray(values, dtype=np.float32)

    def _modality_sequence_values(
        self, bundle: FeatureBundle, modality: Modality
    ) -> tuple[np.ndarray, np.ndarray]:
        matrices = bundle.sequence_by_modality(modality)
        if matrices:
            reference = matrices[0]
            for matrix in matrices[1:]:
                if matrix.mask.shape != reference.mask.shape or not np.array_equal(
                    matrix.mask, reference.mask
                ):
                    raise ValueError(f"{modality.value} sequence matrices must share the same mask")
            values = np.concatenate([m.values for m in matrices], axis=2)
            mask = reference.mask
            if values.shape[2] != self._dims[modality]:
                raise ValueError(
                    f"{modality.value} sequence feature dim changed: "
                    f"{values.shape[2]} != {self._dims[modality]}"
                )
            return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32)
        if self._sequence_fallback_policy() == "error":
            raise ValueError(
                f"{modality.value} sequence feature is required by "
                "modality_encoder.sequence_fallback_policy='error'; "
                "use a sequence extractor or set the policy to 'pooled'"
            )
        values = self._modality_values(bundle, modality)
        return values[:, None, :], np.ones((bundle.n_samples, 1), dtype=np.float32)

    def _sequence_fallback_policy(self) -> str:
        policy = self._config.modality_encoder.sequence_fallback_policy.lower()
        if policy not in {"pooled", "error"}:
            raise ValueError(
                "modality_encoder.sequence_fallback_policy must be 'pooled' or 'error'"
            )
        return policy

    def _input_mode(self) -> str:
        mode = self._config.input_mode.lower()
        if mode not in {"tri_modal", "multimodal"}:
            raise ValueError("dialogue_rnn.input_mode must be 'tri_modal' or 'multimodal'")
        return mode

    @staticmethod
    def _availability_row(bundle: FeatureBundle, row: int) -> np.ndarray:
        result = np.zeros(3, dtype=np.float32)
        for idx, modality in enumerate((Modality.TEXT, Modality.AUDIO, Modality.VIDEO)):
            if not bundle.by_modality(modality) and not bundle.sequence_by_modality(modality):
                continue
            avail = bundle.availability.get(modality)
            if avail is not None:
                result[idx] = 1.0 if bool(avail[row]) else 0.0
            else:
                result[idx] = 1.0
        return result

    @staticmethod
    def _multimodal_availability_row(bundle: FeatureBundle, row: int) -> float:
        avail = bundle.availability.get(Modality.MULTIMODAL)
        if avail is not None:
            return 1.0 if bool(avail[row]) else 0.0
        return 1.0

    def _tensor_batch(
        self,
        arrays: _DialogueArrays,
        dialogue_indices: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        idx = np.asarray(dialogue_indices, dtype=np.int64)
        return {
            "text_x": torch.as_tensor(
                arrays.text_x[idx], dtype=torch.float32, device=self._device
            ),
            "audio_x": torch.as_tensor(
                arrays.audio_x[idx], dtype=torch.float32, device=self._device
            ),
            "video_x": torch.as_tensor(
                arrays.video_x[idx], dtype=torch.float32, device=self._device
            ),
            "text_mask": torch.as_tensor(
                arrays.text_mask[idx], dtype=torch.float32, device=self._device
            ),
            "audio_mask": torch.as_tensor(
                arrays.audio_mask[idx], dtype=torch.float32, device=self._device
            ),
            "video_mask": torch.as_tensor(
                arrays.video_mask[idx], dtype=torch.float32, device=self._device
            ),
            "modality_mask": torch.as_tensor(
                arrays.modality_mask[idx], dtype=torch.float32, device=self._device
            ),
            "speaker_id": torch.as_tensor(
                arrays.speaker_id[idx], dtype=torch.long, device=self._device
            ),
            "utterance_mask": torch.as_tensor(
                arrays.utterance_mask[idx], dtype=torch.float32, device=self._device
            ),
            "labels": torch.as_tensor(arrays.labels[idx], dtype=torch.long, device=self._device),
        }

    def _teacher_prob_arrays(
        self,
        arrays: _DialogueArrays,
        teacher_probs: FloatArray,
    ) -> np.ndarray:
        values = np.zeros(
            (
                arrays.text_x.shape[0],
                arrays.text_x.shape[1],
                len(self._classes),
            ),
            dtype=np.float32,
        )
        for flat_idx, dialogue_idx, slot in arrays.flat_indices:
            values[dialogue_idx, slot] = teacher_probs[flat_idx]
        return values

    def _drop_modalities(self, modality_mask: torch.Tensor) -> torch.Tensor:
        text_p = self._config.training.text_dropout
        if text_p > 0.0:
            modality_mask = modality_mask.clone()
            text_drop = (torch.rand_like(modality_mask[..., 0]) < text_p) & (
                modality_mask[..., 0] > 0.0
            )
            modality_mask[..., 0] = modality_mask[..., 0].masked_fill(text_drop, 0.0)
        p = self._config.training.modality_dropout
        if p <= 0.0:
            return self._ensure_one_modality(modality_mask, modality_mask)
        dropped = modality_mask.clone()
        random = torch.rand_like(dropped)
        drop = (random < p) & (dropped > 0.0)
        dropped = dropped.masked_fill(drop, 0.0)
        return self._ensure_one_modality(dropped, modality_mask)

    @staticmethod
    def _ensure_one_modality(dropped: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        none_left = (dropped.sum(dim=-1) == 0.0) & (original.sum(dim=-1) > 0.0)
        if bool(none_left.any()):
            first_available = torch.argmax(original, dim=-1)
            flat = dropped.reshape(-1, dropped.shape[-1])
            flat_none = none_left.reshape(-1)
            flat_first = first_available.reshape(-1)
            rows = torch.arange(flat.shape[0], device=flat.device)[flat_none]
            flat[rows, flat_first[flat_none]] = 1.0
        return dropped

    def _gate_entropy_loss(self, gate: torch.Tensor, utterance_mask: torch.Tensor) -> torch.Tensor:
        weight = self._config.fusion.gate_entropy_weight
        if weight <= 0.0:
            return gate.sum() * 0.0
        entropy = -(gate * torch.log(gate.clamp_min(1.0e-12))).sum(dim=-1)
        valid = utterance_mask > 0.0
        if not bool(valid.any()):
            return gate.sum() * 0.0
        return -float(weight) * entropy[valid].mean()

    def _auxiliary_loss(
        self,
        output: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        class_counts: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        total = output["logits"].sum() * 0.0
        specs = (
            ("aux_text_logits", 0, self._config.classifier.aux_text_loss_weight),
            ("aux_audio_logits", 1, self._config.classifier.aux_audio_loss_weight),
            ("aux_video_logits", 2, self._config.classifier.aux_video_loss_weight),
        )
        for key, modality_idx, weight in specs:
            if weight <= 0.0:
                continue
            if modality_idx >= batch["modality_mask"].shape[-1]:
                continue
            aux_labels = labels.clone()
            aux_labels[batch["modality_mask"][..., modality_idx] <= 0.0] = -100
            total = total + float(weight) * compute_dialogue_loss(
                output[key],
                aux_labels,
                settings=self._config.loss,
                class_counts=class_counts,
                class_weights=class_weights,
            )
        return total

    def _distillation_loss(
        self,
        logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        utterance_mask: torch.Tensor,
        *,
        temperature: float,
        weight: float,
    ) -> torch.Tensor:
        if weight <= 0.0:
            return logits.sum() * 0.0
        valid = utterance_mask > 0.0
        if not bool(valid.any()):
            return logits.sum() * 0.0
        t = float(temperature)
        teacher = teacher_probs.clamp_min(1.0e-12)
        teacher_logits = torch.log(teacher)
        teacher_soft = torch.softmax(teacher_logits / t, dim=-1)
        student_log = F.log_softmax(logits / t, dim=-1)
        return float(weight) * (t * t) * F.kl_div(
            student_log[valid],
            teacher_soft[valid],
            reduction="batchmean",
        )

    def _validation_score(self, arrays: _DialogueArrays, dialogue_indices: Sequence[int]) -> float:
        model = self._require_model()
        y_true: list[int] = []
        y_pred: list[int] = []
        model.eval()
        gates: list[np.ndarray] = []
        context_alpha: list[np.ndarray] = []
        memory_alpha: list[np.ndarray] = []
        with torch.no_grad():
            for indices in self._iter_batches(tuple(dialogue_indices), shuffle=False):
                batch = self._tensor_batch(arrays, indices)
                output = model(
                    batch["text_x"],
                    batch["audio_x"],
                    batch["video_x"],
                    batch["speaker_id"],
                    batch["utterance_mask"],
                    batch["text_mask"],
                    batch["audio_mask"],
                    batch["video_mask"],
                    batch["modality_mask"],
                )
                pred = torch.argmax(output["logits"], dim=-1)
                valid = batch["labels"] != -100
                gates.append(output["modality_gate"][valid].cpu().numpy())
                if "alpha_context" in output:
                    context_alpha.append(output["alpha_context"][valid].cpu().numpy())
                if "alpha_memory" in output:
                    memory_alpha.append(output["alpha_memory"][valid].cpu().numpy())
                y_true.extend(batch["labels"][valid].cpu().numpy().astype(np.int64).tolist())
                y_pred.extend(pred[valid].cpu().numpy().astype(np.int64).tolist())
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        self._last_false_positive_counts = false_positive_counts(
            y_true_arr,
            y_pred_arr,
            n_classes=len(self._classes),
        )
        self._last_gate_stats = {
            **_gate_stats(gates),
            **_scalar_chunks_stats("context_alpha", context_alpha),
            **_scalar_chunks_stats("memory_alpha", memory_alpha),
        }
        return _weighted_f1(y_true_arr, y_pred_arr)

    def _fit_calibration(self, arrays: _DialogueArrays, dialogue_indices: Sequence[int]) -> None:
        logits, labels = self._collect_logits_and_labels(arrays, dialogue_indices)
        if logits.numel() == 0:
            return
        temperature = 1.0
        if self._config.calibration.temperature_scaling:
            temperature = fit_temperature(logits, labels)
        params = CalibrationParams(
            temperature=temperature,
            class_thresholds=(),
            rare_classes=self._config.calibration.rare_classes,
            rare_class_threshold=self._config.calibration.rare_class_threshold,
            rare_class_margin=self._config.calibration.rare_class_margin
            if self._config.calibration.rare_class_margin_enabled
            else 0.0,
            neutral_gate_enabled=self._config.neutral_gate.enabled,
            neutral_class_index=self._config.neutral_gate.neutral_class_index,
            neutral_emotion_threshold=self._config.neutral_gate.threshold,
            neutral_gate_tuned=False,
            class_labels=tuple(emotion.value for emotion in self._classes),
        )
        postprocessor = PredictionPostprocessor(params)
        if self._config.calibration.threshold_tuning:
            probs = postprocessor.probabilities(logits).detach().cpu().numpy()
            params = CalibrationParams(
                temperature=temperature,
                class_thresholds=tune_class_thresholds(probs, labels.detach().cpu().numpy()),
                rare_classes=params.rare_classes,
                rare_class_threshold=params.rare_class_threshold,
                rare_class_margin=params.rare_class_margin,
                neutral_gate_enabled=params.neutral_gate_enabled,
                neutral_class_index=params.neutral_class_index,
                neutral_emotion_threshold=params.neutral_emotion_threshold,
                neutral_gate_tuned=params.neutral_gate_tuned,
                class_labels=params.class_labels,
            )
            postprocessor = PredictionPostprocessor(params)
        if self._config.neutral_gate.enabled:
            probs = postprocessor.probabilities(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            before_params = CalibrationParams(
                temperature=params.temperature,
                class_thresholds=params.class_thresholds,
                rare_classes=params.rare_classes,
                rare_class_threshold=params.rare_class_threshold,
                rare_class_margin=params.rare_class_margin,
                neutral_gate_enabled=False,
                neutral_class_index=params.neutral_class_index,
                neutral_emotion_threshold=params.neutral_emotion_threshold,
                neutral_gate_tuned=False,
                class_labels=params.class_labels,
            )
            before = PredictionPostprocessor(before_params).predict(probs)
            threshold = self._config.neutral_gate.threshold
            tuned = False
            if self._config.neutral_gate.threshold_tuning:
                threshold = tune_neutral_emotion_threshold(
                    probs,
                    labels_np,
                    neutral_class_index=self._config.neutral_gate.neutral_class_index,
                )
                tuned = True
            tuned_params = CalibrationParams(
                temperature=params.temperature,
                class_thresholds=params.class_thresholds,
                rare_classes=params.rare_classes,
                rare_class_threshold=params.rare_class_threshold,
                rare_class_margin=params.rare_class_margin,
                neutral_gate_enabled=True,
                neutral_class_index=self._config.neutral_gate.neutral_class_index,
                neutral_emotion_threshold=threshold,
                neutral_gate_tuned=tuned,
                neutral_gate_before_accuracy=float(np.mean(before == labels_np))
                if labels_np.size
                else 0.0,
                class_labels=params.class_labels,
            )
            after = PredictionPostprocessor(tuned_params).predict(probs)
            params = CalibrationParams(
                temperature=tuned_params.temperature,
                class_thresholds=tuned_params.class_thresholds,
                rare_classes=tuned_params.rare_classes,
                rare_class_threshold=tuned_params.rare_class_threshold,
                rare_class_margin=tuned_params.rare_class_margin,
                neutral_gate_enabled=tuned_params.neutral_gate_enabled,
                neutral_class_index=tuned_params.neutral_class_index,
                neutral_emotion_threshold=tuned_params.neutral_emotion_threshold,
                neutral_gate_tuned=tuned_params.neutral_gate_tuned,
                neutral_gate_before_accuracy=tuned_params.neutral_gate_before_accuracy,
                neutral_gate_after_accuracy=float(np.mean(after == labels_np))
                if labels_np.size
                else 0.0,
                class_labels=tuned_params.class_labels,
            )
        self._postprocessor = PredictionPostprocessor(params)

    def _collect_logits_and_labels(
        self,
        arrays: _DialogueArrays,
        dialogue_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._require_model()
        logits_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for indices in self._iter_batches(tuple(dialogue_indices), shuffle=False):
                batch = self._tensor_batch(arrays, indices)
                output = model(
                    batch["text_x"],
                    batch["audio_x"],
                    batch["video_x"],
                    batch["speaker_id"],
                    batch["utterance_mask"],
                    batch["text_mask"],
                    batch["audio_mask"],
                    batch["video_mask"],
                    batch["modality_mask"],
                )
                valid = batch["labels"] != -100
                logits_chunks.append(output["logits"][valid].detach())
                label_chunks.append(batch["labels"][valid].detach())
        if not logits_chunks:
            return (
                torch.zeros((0, len(self._classes)), device=self._device),
                torch.zeros(0, dtype=torch.long, device=self._device),
            )
        return torch.cat(logits_chunks, dim=0), torch.cat(label_chunks, dim=0)

    def _split_dialogues(
        self,
        n_dialogues: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        indices = np.arange(n_dialogues, dtype=np.int64)
        rng.shuffle(indices)
        fraction = self._config.training.validation_fraction
        if n_dialogues < 2 or fraction <= 0.0:
            return tuple(int(i) for i in indices), ()
        n_val = min(n_dialogues - 1, max(1, round(float(n_dialogues) * fraction)))
        val = tuple(int(i) for i in indices[:n_val])
        train = tuple(int(i) for i in indices[n_val:])
        return train, val

    def _iter_batches(
        self,
        dialogue_indices: Sequence[int],
        *,
        shuffle: bool,
        rng: np.random.Generator | None = None,
    ) -> Iterator[tuple[int, ...]]:
        indices = np.asarray(dialogue_indices, dtype=np.int64)
        if shuffle and rng is not None:
            rng.shuffle(indices)
        batch_size = max(1, self._config.training.batch_size)
        for start in range(0, len(indices), batch_size):
            yield tuple(int(i) for i in indices[start : start + batch_size])

    def _build_speaker_vocab(self, bundle: FeatureBundle) -> dict[str, int]:
        vocab: dict[str, int] = {}
        for utterance in _utterances_or_fallback(bundle):
            if utterance.speaker not in vocab:
                vocab[utterance.speaker] = len(vocab) + 1
        return vocab

    def _class_weights(self, y: IntArray) -> torch.Tensor:
        counts = self._class_counts(y)
        total = float(counts.sum())
        weights = np.zeros(len(self._classes), dtype=np.float32)
        present = counts > 0
        weights[present] = total / (float(len(self._classes)) * counts[present])
        return torch.as_tensor(weights, dtype=torch.float32)

    def _class_counts(self, y: IntArray) -> np.ndarray:
        return np.bincount(np.asarray(y, dtype=np.int64), minlength=len(self._classes)).astype(
            np.float32
        )

    @staticmethod
    def _state_dict_cpu(model: MultimodalEmotionModel) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _save_best_checkpoint(
        self,
        epoch: int,
        score: float,
        state_dict: Mapping[str, torch.Tensor],
        used_validation: bool,
    ) -> None:
        path = self._config.training.best_checkpoint_path
        if path is None:
            return
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "score": score,
                "score_name": "weighted_f1",
                "score_split": "validation" if used_validation else "train",
                "model_state_dict": dict(state_dict),
                "speaker_to_id": dict(self._speaker_to_id),
                "dims": {modality.value: dim for modality, dim in self._dims.items()},
                "classes": [emotion.value for emotion in self._classes],
                "config": _checkpoint_config(self._config),
                "false_positive_counts": self._last_false_positive_counts.tolist(),
                "gate_stats": dict(self._last_gate_stats),
                "calibration": self._postprocessor.params.to_dict(),
            },
            checkpoint_path,
        )

    def _require_model(self) -> MultimodalEmotionModel:
        if self._model is None:
            raise RuntimeError("학습되지 않은 분류기입니다. 먼저 fit 을 호출하세요.")
        return self._model


def _checkpoint_config(config: DialogueRnnConfig) -> dict[str, Any]:
    result = asdict(config)
    result["type"] = DialogueRnnConfig.type
    return result


def _config_from_checkpoint(data: Mapping[str, Any]) -> DialogueRnnConfig:
    from meld_emotion.config.loader import from_dict

    config = from_dict({"model": dict(data)}).model
    if not isinstance(config, DialogueRnnConfig):
        raise ValueError("checkpoint config 가 dialogue_rnn 설정이 아닙니다")
    return config


def _require_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint 에 {key!r} 매핑이 없습니다")
    return value


def _dims_from_checkpoint(data: Mapping[str, Any]) -> dict[Modality, int]:
    dims: dict[Modality, int] = {}
    if Modality.MULTIMODAL.value in data:
        dims[Modality.MULTIMODAL] = int(data[Modality.MULTIMODAL.value])
        return dims
    for modality in (Modality.TEXT, Modality.AUDIO, Modality.VIDEO):
        value = data.get(modality.value)
        if value is None:
            raise ValueError(f"checkpoint dims 에 {modality.value!r} 값이 없습니다")
        dims[modality] = int(value)
    return dims


def _speaker_vocab_from_checkpoint(data: Mapping[str, Any]) -> dict[str, int]:
    return {str(speaker): int(index) for speaker, index in data.items()}


def _classes_from_checkpoint(value: object) -> tuple[Emotion, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("checkpoint 에 classes 목록이 없습니다")
    classes = tuple(Emotion(str(label)) for label in value)
    if not classes:
        raise ValueError("checkpoint classes 가 비어 있습니다")
    return classes


def _utterances_or_fallback(bundle: FeatureBundle) -> tuple[UtteranceSpec, ...]:
    if bundle.utterances:
        return bundle.utterances
    return tuple(
        UtteranceSpec(uid=uid, dialogue_id=i, utterance_id=0, speaker="")
        for i, uid in enumerate(bundle.uids)
    )


def _dialogue_groups(utterances: Sequence[UtteranceSpec]) -> tuple[tuple[int, ...], ...]:
    by_dialogue: dict[int, list[int]] = {}
    for idx, utterance in enumerate(utterances):
        by_dialogue.setdefault(utterance.dialogue_id, []).append(idx)
    groups: list[tuple[int, ...]] = []
    for dialogue_id in sorted(by_dialogue):
        groups.append(
            tuple(
                sorted(
                    by_dialogue[dialogue_id],
                    key=lambda i: utterances[i].utterance_id,
                )
            )
        )
    return tuple(groups)


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    total = float(y_true.size)
    score = 0.0
    for label in range(len(EMOTION_ORDER)):
        true = y_true == label
        pred = y_pred == label
        support = float(true.sum())
        if support == 0.0:
            continue
        tp = float((true & pred).sum())
        fp = float((~true & pred).sum())
        fn = float((true & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        score += support / total * f1
    return score


def _gate_stats(chunks: Sequence[np.ndarray]) -> dict[str, float]:
    valid_chunks = [chunk for chunk in chunks if chunk.size]
    if not valid_chunks:
        return {}
    gate = np.concatenate(valid_chunks, axis=0)
    if gate.size == 0:
        return {}
    entropy = -(gate * np.log(np.clip(gate, 1.0e-12, 1.0))).sum(axis=1)
    names = ("multimodal",) if gate.shape[1] == 1 else ("text", "audio", "video")
    stats: dict[str, float] = {
        "gate_entropy_mean": float(np.mean(entropy)),
        "gate_entropy_std": float(np.std(entropy)),
        "gate_entropy_min": float(np.min(entropy)),
        "gate_entropy_max": float(np.max(entropy)),
    }
    for idx, name in enumerate(names):
        stats[f"gate_{name}_mean"] = float(np.mean(gate[:, idx]))
        stats[f"gate_{name}_var"] = float(np.var(gate[:, idx]))
        stats[f"gate_{name}_std"] = float(np.std(gate[:, idx]))
        stats[f"gate_{name}_min"] = float(np.min(gate[:, idx]))
        stats[f"gate_{name}_max"] = float(np.max(gate[:, idx]))
    return stats


def _scalar_chunks_stats(prefix: str, chunks: Sequence[np.ndarray]) -> dict[str, float]:
    valid_chunks = [chunk.reshape(-1) for chunk in chunks if chunk.size]
    if not valid_chunks:
        return {}
    values = np.concatenate(valid_chunks, axis=0)
    if values.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }
