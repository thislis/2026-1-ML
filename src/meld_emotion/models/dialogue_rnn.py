"""Pipeline wrapper for the PyTorch dialogue RNN emotion model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from torch import nn

from meld_emotion.config.schema import DialogueRnnConfig
from meld_emotion.core.features import FeatureBundle, UtteranceSpec
from meld_emotion.core.results import PredictionSet
from meld_emotion.core.status import real
from meld_emotion.core.types import EMOTION_ORDER, Emotion, FloatArray, IntArray, Modality
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

    @property
    def classes(self) -> tuple[Emotion, ...]:
        return self._classes

    def fit(self, bundle: FeatureBundle, y: IntArray) -> Self:
        if len(y) != bundle.n_samples:
            raise ValueError(f"y 길이가 샘플 수와 다릅니다: {len(y)} != {bundle.n_samples}")
        torch.manual_seed(self._config.training.seed)
        rng = np.random.default_rng(self._config.training.seed)

        self._dims = self._infer_dims(bundle)
        self._speaker_to_id = self._build_speaker_vocab(bundle)
        arrays = self._build_arrays(bundle, y)
        self._model = self._build_model().to(self._device)

        train_dialogues, val_dialogues = self._split_dialogues(arrays.text_x.shape[0], rng)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.training.lr,
            weight_decay=self._config.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(
            weight=self._class_weights(y).to(self._device),
            ignore_index=-100,
        )

        best_state: Mapping[str, torch.Tensor] | None = None
        best_score = -1.0
        stale = 0
        for _ in range(self._config.training.max_epochs):
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
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.training.gradient_clip_norm,
                )
                optimizer.step()

            if val_dialogues:
                score = self._validation_score(arrays, val_dialogues)
                if score > best_score:
                    best_score = score
                    best_state = self._state_dict_cpu(self._model)
                    stale = 0
                else:
                    stale += 1
                    if stale >= self._config.training.early_stopping_patience:
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict_proba(self, bundle: FeatureBundle) -> FloatArray:
        model = self._require_model()
        arrays = self._build_arrays(bundle, None)
        proba = np.zeros((bundle.n_samples, len(self._classes)), dtype=np.float64)
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
                probs = torch.softmax(output["logits"], dim=-1).cpu().numpy()
                for local_b, dialogue_idx in enumerate(indices):
                    for flat_idx, slot_b, slot_n in arrays.flat_indices:
                        if slot_b == dialogue_idx:
                            proba[flat_idx] = probs[local_b, slot_n]
        return proba

    def predict(self, bundle: FeatureBundle) -> PredictionSet:
        proba = self.predict_proba(bundle)
        y_pred = np.argmax(proba, axis=1).astype(np.int64)
        return PredictionSet(uids=bundle.uids, y_pred=y_pred, proba=proba, classes=self._classes)

    def _build_model(self) -> MultimodalEmotionModel:
        enc = self._config.modality_encoder
        fusion = self._config.fusion
        context = self._config.dialogue_context
        memory = self._config.memory_attention
        head = self._config.classifier
        return MultimodalEmotionModel(
            text_input_dim=self._dims[Modality.TEXT],
            audio_input_dim=self._dims[Modality.AUDIO],
            video_input_dim=self._dims[Modality.VIDEO],
            speaker_vocab_size=max(self._speaker_to_id.values(), default=0) + 1,
            num_classes=len(self._classes),
            rnn_type=self._config.rnn_type,
            modality_proj_dim=enc.proj_dim,
            modality_hidden_dim=enc.hidden_dim,
            modality_dropout=enc.dropout,
            fusion_dim=fusion.fusion_dim,
            fusion_dropout=fusion.dropout,
            use_gated_fusion=fusion.use_gated_fusion,
            use_interaction_features=fusion.use_interaction_features,
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
        )

    def _infer_dims(self, bundle: FeatureBundle) -> dict[Modality, int]:
        configured = {
            Modality.TEXT: self._config.modality_encoder.text_input_dim,
            Modality.AUDIO: self._config.modality_encoder.audio_input_dim,
            Modality.VIDEO: self._config.modality_encoder.video_input_dim,
        }
        dims: dict[Modality, int] = {}
        for modality, config_dim in configured.items():
            actual = sum(m.n_features for m in bundle.by_modality(modality))
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

        text_values = self._modality_values(bundle, Modality.TEXT)
        audio_values = self._modality_values(bundle, Modality.AUDIO)
        video_values = self._modality_values(bundle, Modality.VIDEO)
        text_x = np.zeros((n_dialogues, max_len, 1, self._dims[Modality.TEXT]), dtype=np.float32)
        audio_x = np.zeros((n_dialogues, max_len, 1, self._dims[Modality.AUDIO]), dtype=np.float32)
        video_x = np.zeros((n_dialogues, max_len, 1, self._dims[Modality.VIDEO]), dtype=np.float32)
        text_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        audio_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        video_mask = np.zeros((n_dialogues, max_len, 1), dtype=np.float32)
        modality_mask = np.zeros((n_dialogues, max_len, 3), dtype=np.float32)
        speaker_id = np.zeros((n_dialogues, max_len), dtype=np.int64)
        utterance_mask = np.zeros((n_dialogues, max_len), dtype=np.float32)
        labels = np.full((n_dialogues, max_len), -100, dtype=np.int64)
        flat_indices: list[tuple[int, int, int]] = []

        for dialogue_idx, indices in enumerate(groups):
            for slot, flat_idx in enumerate(indices):
                text_x[dialogue_idx, slot, 0] = text_values[flat_idx]
                audio_x[dialogue_idx, slot, 0] = audio_values[flat_idx]
                video_x[dialogue_idx, slot, 0] = video_values[flat_idx]
                text_mask[dialogue_idx, slot, 0] = 1.0
                audio_mask[dialogue_idx, slot, 0] = 1.0
                video_mask[dialogue_idx, slot, 0] = 1.0
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

    @staticmethod
    def _availability_row(bundle: FeatureBundle, row: int) -> np.ndarray:
        result = np.zeros(3, dtype=np.float32)
        for idx, modality in enumerate((Modality.TEXT, Modality.AUDIO, Modality.VIDEO)):
            if not bundle.by_modality(modality):
                continue
            avail = bundle.availability.get(modality)
            if avail is not None:
                result[idx] = 1.0 if bool(avail[row]) else 0.0
            else:
                result[idx] = 1.0
        return result

    def _tensor_batch(
        self,
        arrays: _DialogueArrays,
        dialogue_indices: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        idx = np.asarray(dialogue_indices, dtype=np.int64)
        return {
            "text_x": torch.as_tensor(arrays.text_x[idx], device=self._device),
            "audio_x": torch.as_tensor(arrays.audio_x[idx], device=self._device),
            "video_x": torch.as_tensor(arrays.video_x[idx], device=self._device),
            "text_mask": torch.as_tensor(arrays.text_mask[idx], device=self._device),
            "audio_mask": torch.as_tensor(arrays.audio_mask[idx], device=self._device),
            "video_mask": torch.as_tensor(arrays.video_mask[idx], device=self._device),
            "modality_mask": torch.as_tensor(arrays.modality_mask[idx], device=self._device),
            "speaker_id": torch.as_tensor(arrays.speaker_id[idx], device=self._device),
            "utterance_mask": torch.as_tensor(arrays.utterance_mask[idx], device=self._device),
            "labels": torch.as_tensor(arrays.labels[idx], device=self._device),
        }

    def _drop_modalities(self, modality_mask: torch.Tensor) -> torch.Tensor:
        p = self._config.training.modality_dropout
        if p <= 0.0:
            return modality_mask
        dropped = modality_mask.clone()
        random = torch.rand_like(dropped)
        drop = (random < p) & (dropped > 0.0)
        dropped = dropped.masked_fill(drop, 0.0)
        none_left = (dropped.sum(dim=-1) == 0.0) & (modality_mask.sum(dim=-1) > 0.0)
        if bool(none_left.any()):
            first_available = torch.argmax(modality_mask, dim=-1)
            flat = dropped.reshape(-1, dropped.shape[-1])
            flat_none = none_left.reshape(-1)
            flat_first = first_available.reshape(-1)
            rows = torch.arange(flat.shape[0], device=flat.device)[flat_none]
            flat[rows, flat_first[flat_none]] = 1.0
        return dropped

    def _validation_score(self, arrays: _DialogueArrays, dialogue_indices: Sequence[int]) -> float:
        model = self._require_model()
        y_true: list[int] = []
        y_pred: list[int] = []
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
                pred = torch.argmax(output["logits"], dim=-1)
                valid = batch["labels"] != -100
                y_true.extend(batch["labels"][valid].cpu().numpy().astype(np.int64).tolist())
                y_pred.extend(pred[valid].cpu().numpy().astype(np.int64).tolist())
        return _weighted_f1(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))

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
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=len(self._classes)).astype(np.float32)
        total = float(counts.sum())
        weights = np.zeros(len(self._classes), dtype=np.float32)
        present = counts > 0
        weights[present] = total / (float(len(self._classes)) * counts[present])
        return torch.as_tensor(weights, dtype=torch.float32)

    @staticmethod
    def _state_dict_cpu(model: MultimodalEmotionModel) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _require_model(self) -> MultimodalEmotionModel:
        if self._model is None:
            raise RuntimeError("학습되지 않은 분류기입니다. 먼저 fit 을 호출하세요.")
        return self._model


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
