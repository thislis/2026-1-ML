"""Wav2Vec2 XLS-R fine-tuning on MELD audio emotion labels."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from meld_emotion.core.data import AudioInput
from meld_emotion.core.types import EMOTION_ORDER, Emotion
from meld_emotion.data.media import MediaLoader

EMOTION_LABELS: tuple[str, ...] = tuple(emotion.value for emotion in EMOTION_ORDER)
EARLY_STOPPING_METRICS = frozenset(("none", "eval_loss", "eval_macro_f1", "eval_weighted_f1"))
ON_ERROR_POLICIES = frozenset(("drop_sample", "fail_fast"))
_LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
_REQUIRED_COLUMNS = frozenset(("Emotion", "Dialogue_ID", "Utterance_ID"))


class _FeatureExtractor(Protocol):
    def __call__(self, waveforms: Sequence[np.ndarray], **kwargs: object) -> Mapping[str, object]: ...

    def save_pretrained(self, path: str) -> None: ...


class _FeatureExtractorClass(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> _FeatureExtractor: ...


class _ModelClass(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> object: ...


class _TrainerClass(Protocol):
    def __call__(self, **kwargs: object) -> Any: ...


class _TrainingArgumentsClass(Protocol):
    def __call__(self, **kwargs: object) -> Any: ...


class _EarlyStoppingCallbackClass(Protocol):
    def __call__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float = 0.0,
    ) -> object: ...


class _EvalPrediction(Protocol):
    predictions: object
    label_ids: object


@dataclass(frozen=True)
class MeldAudioEmotionExample:
    """Single MELD utterance mapped to its split MP4 clip and emotion label."""

    uid: str
    mp4_path: Path
    emotion: str
    label: int
    dialogue_id: int
    utterance_id: int


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic train/eval split as original example indices."""

    train: tuple[int, ...]
    eval: tuple[int, ...]


@dataclass(frozen=True)
class Wav2Vec2FineTuneConfig:
    """Configuration for MELD emotion-supervised Wav2Vec2 fine-tuning."""

    csv_path: Path = Path("MELD.Raw/train/train_sent_emo.csv")
    mp4_dir: Path = Path("MELD.Raw/train/train_splits")
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    output_dir: Path = Path("outputs/wav2vec2_meld_finetuned")
    epochs: float = 1.0
    batch_size: int = 4
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    eval_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    max_steps: int | None = None
    save_total_limit: int = 2
    eval_steps: int = 100
    early_stopping_metric: str = "eval_loss"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    sampling_rate: int = 16000
    max_audio_seconds: float | None = 60.0
    min_audio_seconds: float | None = 0.025
    freeze_feature_encoder: bool = True
    on_error: str = "drop_sample"


@dataclass(frozen=True)
class Wav2Vec2FineTuneSummary:
    """Serializable summary of a Wav2Vec2 fine-tuning run."""

    csv_path: str
    mp4_dir: str
    model_name: str
    output_dir: str
    final_classifier_dir: str
    final_encoder_dir: str
    n_examples: int
    n_train: int
    n_eval: int
    label_counts: dict[str, int]
    label_to_id: dict[str, int]
    skipped: dict[str, int]
    epochs: float
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    eval_fraction: float
    seed: int
    device: str
    fp16: bool
    bf16: bool
    max_steps: int | None
    eval_steps: int
    early_stopping_metric: str
    early_stopping_patience: int
    early_stopping_threshold: float
    sampling_rate: int
    max_audio_seconds: float | None
    min_audio_seconds: float | None
    freeze_feature_encoder: bool
    on_error: str


@dataclass(frozen=True)
class _TrainingDependencies:
    feature_extractor_cls: _FeatureExtractorClass
    sequence_classifier_cls: _ModelClass
    encoder_cls: _ModelClass
    trainer_cls: _TrainerClass
    training_arguments_cls: _TrainingArgumentsClass
    early_stopping_callback_cls: _EarlyStoppingCallbackClass
    torch: Any


@dataclass
class _SkipStats:
    missing_mp4: int = 0
    decode_error: int = 0
    empty_batch: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "missing_mp4": self.missing_mp4,
            "decode_error": self.decode_error,
            "empty_batch": self.empty_batch,
        }

    def merge(self, other: _SkipStats) -> None:
        self.missing_mp4 += other.missing_mp4
        self.decode_error += other.decode_error
        self.empty_batch += other.empty_batch


class MeldAudioDataset:
    """Small torch-dataset-compatible wrapper around immutable examples."""

    def __init__(self, examples: Sequence[MeldAudioEmotionExample]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> MeldAudioEmotionExample:
        return self._examples[index]


class MeldAudioDataCollator:
    """Load split MP4 audio lazily and build padded Wav2Vec2 batches."""

    def __init__(
        self,
        feature_extractor: _FeatureExtractor,
        media_loader: MediaLoader,
        torch: Any,
        sampling_rate: int,
        on_error: str,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._media_loader = media_loader
        self._torch = torch
        self._sampling_rate = sampling_rate
        self._on_error = on_error
        self.skipped = _SkipStats()

    def __call__(self, features: Sequence[MeldAudioEmotionExample]) -> dict[str, Any]:
        waveforms: list[np.ndarray] = []
        labels: list[int] = []
        for example in features:
            try:
                loaded = self._media_loader.load_audio(
                    AudioInput(sample_rate=self._sampling_rate, source_path=example.mp4_path)
                )
                if loaded.waveform is None:
                    raise ValueError(f"오디오 waveform 이 없습니다: {example.mp4_path}")
                wave = np.asarray(loaded.waveform, dtype=np.float32).reshape(-1)
                if wave.size == 0:
                    raise ValueError(f"오디오 waveform 이 비어 있습니다: {example.mp4_path}")
            except Exception:
                if self._on_error == "fail_fast":
                    raise
                self.skipped.decode_error += 1
                continue
            waveforms.append(wave)
            labels.append(example.label)

        if not waveforms:
            self.skipped.empty_batch += 1
            raise ValueError("batch 안의 모든 오디오 샘플을 drop 해서 학습할 수 없습니다")

        batch = self._feature_extractor(
            waveforms,
            sampling_rate=self._sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        values = dict(batch)
        values["labels"] = self._torch.tensor(labels, dtype=self._torch.long)
        return values


def load_meld_audio_emotion_examples(
    csv_path: str | Path,
    mp4_dir: str | Path,
    on_error: str = "drop_sample",
) -> tuple[MeldAudioEmotionExample, ...]:
    """Load MELD CSV rows and map them to utterance-level MP4 paths."""

    examples, _ = _load_meld_audio_emotion_examples_with_stats(csv_path, mp4_dir, on_error)
    return examples


def stratified_train_eval_split(
    examples: Sequence[MeldAudioEmotionExample],
    eval_fraction: float = 0.1,
    seed: int = 0,
) -> SplitIndices:
    """Create a deterministic label-stratified split."""

    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction 은 0 이상 1 미만이어야 합니다")
    if not examples:
        raise ValueError("examples 는 비어 있을 수 없습니다")

    by_label: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_label[example.label].append(index)

    missing_labels = sorted(set(range(len(EMOTION_LABELS))).difference(by_label))
    if missing_labels:
        labels = ", ".join(EMOTION_LABELS[index] for index in missing_labels)
        raise ValueError(f"Wav2Vec2 fine-tuning 에 필요한 감정 라벨이 없습니다: {labels}")

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for label in range(len(EMOTION_LABELS)):
        indices = list(by_label[label])
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                "Wav2Vec2 fine-tuning 은 라벨별 최소 2개 이상의 예제가 필요합니다: "
                f"label={EMOTION_LABELS[label]}"
            )
        eval_count = 0
        if eval_fraction > 0.0 and len(indices) > 2:
            eval_count = max(1, round(len(indices) * eval_fraction))
            eval_count = min(eval_count, len(indices) - 1)
        eval_indices.extend(indices[:eval_count])
        train_indices.extend(indices[eval_count:])

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return SplitIndices(train=tuple(train_indices), eval=tuple(eval_indices))


def run_wav2vec2_fine_tuning(config: Wav2Vec2FineTuneConfig) -> Wav2Vec2FineTuneSummary:
    """Fine-tune Wav2Vec2 on MELD emotion labels and save an extractor-compatible encoder."""

    _validate_config(config)
    dependencies = _load_training_dependencies()
    examples, skipped = _load_meld_audio_emotion_examples_with_stats(
        config.csv_path,
        config.mp4_dir,
        config.on_error,
    )
    split = stratified_train_eval_split(examples, config.eval_fraction, config.seed)
    train_examples = tuple(examples[index] for index in split.train)
    eval_examples = tuple(examples[index] for index in split.eval)
    if config.early_stopping_metric != "none" and not eval_examples:
        raise ValueError("early stopping 을 사용하려면 eval split 에 샘플이 있어야 합니다")

    feature_extractor = dependencies.feature_extractor_cls.from_pretrained(config.model_name)
    id2label = dict(enumerate(EMOTION_LABELS))
    label2id = dict(_LABEL_TO_ID)
    model = dependencies.sequence_classifier_cls.from_pretrained(
        config.model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if config.freeze_feature_encoder:
        _freeze_feature_encoder(model)
    if config.device != "auto":
        _move_to_device(model, config.device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    media_loader = MediaLoader(
        audio_sample_rate=config.sampling_rate,
        max_audio_seconds=config.max_audio_seconds,
        min_audio_seconds=config.min_audio_seconds,
    )
    collator = MeldAudioDataCollator(
        feature_extractor=feature_extractor,
        media_loader=media_loader,
        torch=dependencies.torch,
        sampling_rate=config.sampling_rate,
        on_error=config.on_error,
    )
    eval_dataset = MeldAudioDataset(eval_examples) if eval_examples else None
    callbacks: list[object] = []
    if config.early_stopping_metric != "none":
        callbacks.append(
            dependencies.early_stopping_callback_cls(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        )
    should_load_best_model = config.early_stopping_metric != "none"
    args = dependencies.training_arguments_cls(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=50,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        load_best_model_at_end=should_load_best_model,
        metric_for_best_model=(
            config.early_stopping_metric if should_load_best_model else None
        ),
        greater_is_better=(
            config.early_stopping_metric != "eval_loss" if should_load_best_model else None
        ),
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = dependencies.trainer_cls(
        model=model,
        args=args,
        train_dataset=MeldAudioDataset(train_examples),
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    skipped.merge(collator.skipped)
    final_classifier_dir = config.output_dir / "final_classifier"
    final_encoder_dir = config.output_dir / "final_encoder"
    _save_pretrained(model, final_classifier_dir)
    _save_pretrained(feature_extractor, final_classifier_dir)
    _save_encoder(
        model=model,
        feature_extractor=feature_extractor,
        encoder_cls=dependencies.encoder_cls,
        final_classifier_dir=final_classifier_dir,
        final_encoder_dir=final_encoder_dir,
    )
    summary = _build_summary(
        config=config,
        examples=examples,
        split=split,
        skipped=skipped,
        final_classifier_dir=final_classifier_dir,
        final_encoder_dir=final_encoder_dir,
    )
    _write_summary(config.output_dir / "training_summary.json", summary)
    return summary


def _load_meld_audio_emotion_examples_with_stats(
    csv_path: str | Path,
    mp4_dir: str | Path,
    on_error: str,
) -> tuple[tuple[MeldAudioEmotionExample, ...], _SkipStats]:
    if on_error not in ON_ERROR_POLICIES:
        allowed = ", ".join(sorted(ON_ERROR_POLICIES))
        raise ValueError(f"on_error 는 {allowed} 중 하나여야 합니다")

    path = Path(csv_path)
    media_dir = Path(mp4_dir)
    skipped = _SkipStats()
    examples: list[MeldAudioEmotionExample] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"MELD CSV header 를 읽을 수 없습니다: {path}")
        missing = sorted(_REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"MELD CSV 필수 컬럼이 없습니다: {joined}")
        for row_number, row in enumerate(reader, start=2):
            dialogue_id = _required_int(row, "Dialogue_ID", row_number)
            utterance_id = _required_int(row, "Utterance_ID", row_number)
            emotion_text = row["Emotion"].strip()
            try:
                emotion = Emotion(emotion_text)
            except ValueError as exc:
                raise ValueError(
                    f"알 수 없는 MELD Emotion 값입니다: row={row_number}, value={emotion_text!r}"
                ) from exc
            mp4_path = media_dir / f"dia{dialogue_id}_utt{utterance_id}.mp4"
            if not mp4_path.exists():
                if on_error == "fail_fast":
                    raise FileNotFoundError(f"MELD MP4 파일을 찾을 수 없습니다: {mp4_path}")
                skipped.missing_mp4 += 1
                continue
            examples.append(
                MeldAudioEmotionExample(
                    uid=f"{dialogue_id}_{utterance_id}",
                    mp4_path=mp4_path,
                    emotion=emotion.value,
                    label=_LABEL_TO_ID[emotion.value],
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                )
            )

    if not examples:
        raise ValueError(f"MELD CSV/MP4 에 학습 가능한 row 가 없습니다: {path}, {media_dir}")
    return tuple(examples), skipped


def _required_int(row: dict[str, str], key: str, row_number: int) -> int:
    value = row[key].strip()
    if not value:
        raise ValueError(f"MELD CSV 값이 비어 있습니다: row={row_number}, column={key}")
    return int(value)


def _validate_config(config: Wav2Vec2FineTuneConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs 는 양수여야 합니다")
    if config.batch_size <= 0:
        raise ValueError("batch_size 는 양수여야 합니다")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 는 양수여야 합니다")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio 는 0 이상 1 미만이어야 합니다")
    if not 0.0 <= config.eval_fraction < 1.0:
        raise ValueError("eval_fraction 은 0 이상 1 미만이어야 합니다")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps 는 양수이거나 None 이어야 합니다")
    if config.save_total_limit <= 0:
        raise ValueError("save_total_limit 는 양수여야 합니다")
    if config.eval_steps <= 0:
        raise ValueError("eval_steps 는 양수여야 합니다")
    if config.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device 는 auto/cpu/mps/cuda 중 하나여야 합니다")
    if config.early_stopping_metric not in EARLY_STOPPING_METRICS:
        allowed = ", ".join(sorted(EARLY_STOPPING_METRICS))
        raise ValueError(f"early_stopping_metric 은 {allowed} 중 하나여야 합니다")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience 는 양수여야 합니다")
    if config.early_stopping_threshold < 0.0:
        raise ValueError("early_stopping_threshold 는 0 이상이어야 합니다")
    if config.early_stopping_metric != "none" and config.eval_fraction <= 0.0:
        raise ValueError("early stopping 을 사용하려면 eval_fraction 이 0보다 커야 합니다")
    if config.sampling_rate <= 0:
        raise ValueError("sampling_rate 는 양수여야 합니다")
    if config.max_audio_seconds is not None and config.max_audio_seconds <= 0.0:
        raise ValueError("max_audio_seconds 는 양수이거나 None 이어야 합니다")
    if config.min_audio_seconds is not None and config.min_audio_seconds <= 0.0:
        raise ValueError("min_audio_seconds 는 양수이거나 None 이어야 합니다")
    if config.on_error not in ON_ERROR_POLICIES:
        allowed = ", ".join(sorted(ON_ERROR_POLICIES))
        raise ValueError(f"on_error 는 {allowed} 중 하나여야 합니다")


def _load_training_dependencies() -> _TrainingDependencies:
    try:
        transformers_module = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2 fine-tuning requires transformers. Install it with `uv sync --extra audio`."
        ) from exc
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2 fine-tuning requires PyTorch. Install it with `uv sync --extra audio`."
        ) from exc
    try:
        feature_extractor_cls = transformers_module.AutoFeatureExtractor
        sequence_classifier_cls = transformers_module.Wav2Vec2ForSequenceClassification
        encoder_cls = transformers_module.Wav2Vec2Model
        trainer_cls = transformers_module.Trainer
        training_arguments_cls = transformers_module.TrainingArguments
    except AttributeError as exc:
        raise ImportError(
            "The installed transformers package does not expose the Wav2Vec2 training classes."
        ) from exc
    return _TrainingDependencies(
        feature_extractor_cls=cast(_FeatureExtractorClass, feature_extractor_cls),
        sequence_classifier_cls=cast(_ModelClass, sequence_classifier_cls),
        encoder_cls=cast(_ModelClass, encoder_cls),
        trainer_cls=cast(_TrainerClass, trainer_cls),
        training_arguments_cls=cast(_TrainingArgumentsClass, training_arguments_cls),
        early_stopping_callback_cls=cast(
            _EarlyStoppingCallbackClass,
            _load_early_stopping_callback_class(),
        ),
        torch=torch,
    )


def _load_early_stopping_callback_class() -> object:
    try:
        transformers_module = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2 fine-tuning requires transformers. Install it with `uv sync --extra audio`."
        ) from exc
    try:
        return transformers_module.EarlyStoppingCallback
    except AttributeError:
        pass
    try:
        callback_module = import_module("transformers.trainer_callback")
    except ImportError as exc:
        raise ImportError("transformers EarlyStoppingCallback 을 import 할 수 없습니다") from exc
    try:
        return callback_module.__dict__["EarlyStoppingCallback"]
    except KeyError as exc:
        raise ImportError("The installed transformers package does not expose EarlyStoppingCallback.") from exc


def _freeze_feature_encoder(model: object) -> None:
    freeze = getattr(model, "freeze_feature_encoder", None)
    if callable(freeze):
        freeze()
        return
    freeze_extractor = getattr(model, "freeze_feature_extractor", None)
    if callable(freeze_extractor):
        freeze_extractor()


def _move_to_device(model: object, device: str) -> None:
    to_device = getattr(model, "to", None)
    if callable(to_device):
        to_device(device)


def _compute_metrics(eval_prediction: _EvalPrediction) -> dict[str, float]:
    predictions = eval_prediction.predictions
    labels = np.asarray(eval_prediction.label_ids, dtype=np.int64)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    logits = np.asarray(predictions, dtype=np.float64)
    y_pred = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)
    macro_f1, weighted_f1 = _macro_weighted_f1(labels, y_pred, len(EMOTION_LABELS))
    accuracy = float(np.mean(y_pred == labels)) if labels.size else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def _macro_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[float, float]:
    f1_scores: list[float] = []
    supports: list[int] = []
    for label in range(n_classes):
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(f1)
        supports.append(int(np.sum(true_mask)))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    total = sum(supports)
    weighted_f1 = (
        float(sum(score * support for score, support in zip(f1_scores, supports, strict=True)) / total)
        if total > 0
        else 0.0
    )
    return macro_f1, weighted_f1


def _save_pretrained(model_or_processor: object, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save = getattr(model_or_processor, "save_pretrained", None)
    if not callable(save):
        raise TypeError(f"save_pretrained 를 지원하지 않는 객체입니다: {type(model_or_processor).__name__}")
    save(str(path))


def _save_encoder(
    model: object,
    feature_extractor: _FeatureExtractor,
    encoder_cls: _ModelClass,
    final_classifier_dir: Path,
    final_encoder_dir: Path,
) -> None:
    encoder = getattr(model, "wav2vec2", None)
    if encoder is not None and callable(getattr(encoder, "save_pretrained", None)):
        _save_pretrained(encoder, final_encoder_dir)
    else:
        loaded_encoder = encoder_cls.from_pretrained(str(final_classifier_dir))
        _save_pretrained(loaded_encoder, final_encoder_dir)
    _save_pretrained(feature_extractor, final_encoder_dir)


def _build_summary(
    config: Wav2Vec2FineTuneConfig,
    examples: Sequence[MeldAudioEmotionExample],
    split: SplitIndices,
    skipped: _SkipStats,
    final_classifier_dir: Path,
    final_encoder_dir: Path,
) -> Wav2Vec2FineTuneSummary:
    counts = Counter(example.emotion for example in examples)
    return Wav2Vec2FineTuneSummary(
        csv_path=str(config.csv_path),
        mp4_dir=str(config.mp4_dir),
        model_name=config.model_name,
        output_dir=str(config.output_dir),
        final_classifier_dir=str(final_classifier_dir),
        final_encoder_dir=str(final_encoder_dir),
        n_examples=len(examples),
        n_train=len(split.train),
        n_eval=len(split.eval),
        label_counts={label: counts[label] for label in EMOTION_LABELS},
        label_to_id=dict(_LABEL_TO_ID),
        skipped=skipped.to_dict(),
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        eval_fraction=config.eval_fraction,
        seed=config.seed,
        device=config.device,
        fp16=config.fp16,
        bf16=config.bf16,
        max_steps=config.max_steps,
        eval_steps=config.eval_steps,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold,
        sampling_rate=config.sampling_rate,
        max_audio_seconds=config.max_audio_seconds,
        min_audio_seconds=config.min_audio_seconds,
        freeze_feature_encoder=config.freeze_feature_encoder,
        on_error=config.on_error,
    )


def _write_summary(path: Path, summary: Wav2Vec2FineTuneSummary) -> None:
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
