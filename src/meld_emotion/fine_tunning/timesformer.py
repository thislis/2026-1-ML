"""TimeSformer fine-tuning on MELD video emotion labels."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from meld_emotion.core.data import VideoInput
from meld_emotion.core.types import EMOTION_ORDER, Emotion
from meld_emotion.data.media import MediaLoader

EMOTION_LABELS: tuple[str, ...] = tuple(emotion.value for emotion in EMOTION_ORDER)
EARLY_STOPPING_METRICS = frozenset(("none", "eval_loss", "eval_macro_f1", "eval_weighted_f1"))
ON_ERROR_POLICIES = frozenset(("drop_sample", "fail_fast"))
_LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
_REQUIRED_COLUMNS = frozenset(("Emotion", "Dialogue_ID", "Utterance_ID"))
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


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
class MeldVideoEmotionExample:
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
class TimeSformerFineTuneConfig:
    """Configuration for MELD emotion-supervised TimeSformer fine-tuning."""

    csv_path: Path = Path("MELD.Raw/train/train_sent_emo.csv")
    mp4_dir: Path = Path("MELD.Raw/train/train_splits")
    model_name: str = "facebook/timesformer-base-finetuned-k400"
    output_dir: Path = Path("outputs/timesformer_meld_finetuned")
    epochs: float = 1.0
    batch_size: int = 2
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
    num_frames: int = 8
    frame_size: int = 224
    freeze_backbone: bool = False
    on_error: str = "drop_sample"


@dataclass(frozen=True)
class TimeSformerFineTuneSummary:
    """Serializable summary of a TimeSformer fine-tuning run."""

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
    num_frames: int
    frame_size: int
    freeze_backbone: bool
    on_error: str


@dataclass(frozen=True)
class _TrainingDependencies:
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


class MeldVideoDataset:
    """Small torch-dataset-compatible wrapper around immutable examples."""

    def __init__(self, examples: Sequence[MeldVideoEmotionExample]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> MeldVideoEmotionExample:
        return self._examples[index]


class MeldVideoDataCollator:
    """Load split MP4 video lazily and build TimeSformer pixel-value batches."""

    def __init__(
        self,
        media_loader: MediaLoader,
        torch: Any,
        num_frames: int,
        frame_size: int,
        on_error: str,
    ) -> None:
        self._media_loader = media_loader
        self._torch = torch
        self._num_frames = num_frames
        self._frame_size = frame_size
        self._on_error = on_error
        self.skipped = _SkipStats()

    def __call__(self, features: Sequence[MeldVideoEmotionExample]) -> dict[str, Any]:
        pixel_values: list[np.ndarray] = []
        labels: list[int] = []
        for example in features:
            try:
                loaded = self._media_loader.load_video(
                    VideoInput(fps=25.0, source_path=example.mp4_path)
                )
                if loaded.frames is None:
                    raise ValueError(f"비디오 frames 가 없습니다: {example.mp4_path}")
                frames = _prepare_frames(loaded.frames, self._num_frames, self._frame_size)
            except Exception:
                if self._on_error == "fail_fast":
                    raise
                self.skipped.decode_error += 1
                continue
            pixel_values.append(frames)
            labels.append(example.label)

        if not pixel_values:
            self.skipped.empty_batch += 1
            raise ValueError("batch 안의 모든 비디오 샘플을 drop 해서 학습할 수 없습니다")

        pixels = np.asarray(pixel_values, dtype=np.float32)
        return {
            "pixel_values": self._torch.as_tensor(pixels, dtype=self._torch.float32),
            "labels": self._torch.tensor(labels, dtype=self._torch.long),
        }


def load_meld_video_emotion_examples(
    csv_path: str | Path,
    mp4_dir: str | Path,
    on_error: str = "drop_sample",
) -> tuple[MeldVideoEmotionExample, ...]:
    """Load MELD CSV rows and map them to utterance-level MP4 paths."""

    examples, _ = _load_meld_video_emotion_examples_with_stats(csv_path, mp4_dir, on_error)
    return examples


def stratified_train_eval_split(
    examples: Sequence[MeldVideoEmotionExample],
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
        raise ValueError(f"TimeSformer fine-tuning 에 필요한 감정 라벨이 없습니다: {labels}")

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for label in range(len(EMOTION_LABELS)):
        indices = list(by_label[label])
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                "TimeSformer fine-tuning 은 라벨별 최소 2개 이상의 예제가 필요합니다: "
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


def run_timesformer_fine_tuning(
    config: TimeSformerFineTuneConfig,
) -> TimeSformerFineTuneSummary:
    """Fine-tune TimeSformer on MELD emotion labels and save an extractor-compatible encoder."""

    _validate_config(config)
    dependencies = _load_training_dependencies()
    examples, skipped = _load_meld_video_emotion_examples_with_stats(
        config.csv_path,
        config.mp4_dir,
        config.on_error,
    )
    split = stratified_train_eval_split(examples, config.eval_fraction, config.seed)
    train_examples = tuple(examples[index] for index in split.train)
    eval_examples = tuple(examples[index] for index in split.eval)
    if config.early_stopping_metric != "none" and not eval_examples:
        raise ValueError("early stopping 을 사용하려면 eval split 에 샘플이 있어야 합니다")

    id2label = dict(enumerate(EMOTION_LABELS))
    label2id = dict(_LABEL_TO_ID)
    model = dependencies.sequence_classifier_cls.from_pretrained(
        config.model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if config.freeze_backbone:
        _freeze_backbone(model)
    if config.device != "auto":
        _move_to_device(model, config.device)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    media_loader = MediaLoader(
        video_max_frames=config.num_frames,
        video_frame_size=(config.frame_size, config.frame_size),
    )
    collator = MeldVideoDataCollator(
        media_loader=media_loader,
        torch=dependencies.torch,
        num_frames=config.num_frames,
        frame_size=config.frame_size,
        on_error=config.on_error,
    )
    eval_dataset = MeldVideoDataset(eval_examples) if eval_examples else None
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
        train_dataset=MeldVideoDataset(train_examples),
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
    _save_encoder(
        model=model,
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


def _load_meld_video_emotion_examples_with_stats(
    csv_path: str | Path,
    mp4_dir: str | Path,
    on_error: str,
) -> tuple[tuple[MeldVideoEmotionExample, ...], _SkipStats]:
    if on_error not in ON_ERROR_POLICIES:
        allowed = ", ".join(sorted(ON_ERROR_POLICIES))
        raise ValueError(f"on_error 는 {allowed} 중 하나여야 합니다")

    path = Path(csv_path)
    media_dir = Path(mp4_dir)
    skipped = _SkipStats()
    examples: list[MeldVideoEmotionExample] = []
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
                MeldVideoEmotionExample(
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


def _validate_config(config: TimeSformerFineTuneConfig) -> None:
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
    if config.num_frames <= 0:
        raise ValueError("num_frames 는 양수여야 합니다")
    if config.frame_size <= 0:
        raise ValueError("frame_size 는 양수여야 합니다")
    if config.on_error not in ON_ERROR_POLICIES:
        allowed = ", ".join(sorted(ON_ERROR_POLICIES))
        raise ValueError(f"on_error 는 {allowed} 중 하나여야 합니다")


def _load_training_dependencies() -> _TrainingDependencies:
    try:
        transformers_module = import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "TimeSformer fine-tuning requires transformers. Install it with `uv sync --extra video`."
        ) from exc
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "TimeSformer fine-tuning requires PyTorch. Install it with `uv sync --extra video`."
        ) from exc
    try:
        sequence_classifier_cls = transformers_module.TimesformerForVideoClassification
        encoder_cls = transformers_module.TimesformerModel
        trainer_cls = transformers_module.Trainer
        training_arguments_cls = transformers_module.TrainingArguments
    except AttributeError as exc:
        raise ImportError(
            "The installed transformers package does not expose the TimeSformer training classes."
        ) from exc
    return _TrainingDependencies(
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
            "TimeSformer fine-tuning requires transformers. Install it with `uv sync --extra video`."
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


def _prepare_frames(frames: np.ndarray, num_frames: int, frame_size: int) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.size == 0:
        raise ValueError("비디오 frames 가 비어 있습니다")
    if values.ndim != 4:
        raise ValueError(f"TimeSformer 입력 frames 는 (T,H,W,C) 형식이어야 합니다: ndim={values.ndim}")
    if values.shape[-1] == 1:
        values = np.repeat(values, 3, axis=-1)
    elif values.shape[-1] >= 3:
        values = values[..., :3]
    else:
        raise ValueError(f"TimeSformer 입력 channel 수가 올바르지 않습니다: {values.shape[-1]}")
    if float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    values = np.nan_to_num(values, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    values = _sample_frames(values, num_frames)
    values = _resize_frames(values, frame_size)
    normalized = (values - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32, copy=False)


def _sample_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if frames.shape[0] == num_frames:
        return frames
    indices = np.linspace(0, frames.shape[0] - 1, num_frames)
    return frames[np.rint(indices).astype(np.int64)]


def _resize_frames(frames: np.ndarray, frame_size: int) -> np.ndarray:
    if frames.shape[1] == frame_size and frames.shape[2] == frame_size:
        return np.asarray(frames, dtype=np.float32)
    try:
        cv2: Any = import_module("cv2")
    except ImportError:
        resized = [_nearest_resize(frame, frame_size) for frame in frames]
    else:
        resized = [
            cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ]
    return np.asarray(resized, dtype=np.float32)


def _nearest_resize(frame: np.ndarray, size: int) -> np.ndarray:
    y = np.rint(np.linspace(0, frame.shape[0] - 1, size)).astype(np.int64)
    x = np.rint(np.linspace(0, frame.shape[1] - 1, size)).astype(np.int64)
    return frame[y][:, x]


def _freeze_backbone(model: object) -> None:
    backbone = getattr(model, "timesformer", None)
    if backbone is None:
        return
    parameters = getattr(backbone, "parameters", None)
    if not callable(parameters):
        return
    for parameter in parameters():
        parameter.requires_grad = False


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


def _save_pretrained(model: object, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save = getattr(model, "save_pretrained", None)
    if not callable(save):
        raise TypeError(f"save_pretrained 를 지원하지 않는 객체입니다: {type(model).__name__}")
    save(str(path))


def _save_encoder(
    model: object,
    encoder_cls: _ModelClass,
    final_classifier_dir: Path,
    final_encoder_dir: Path,
) -> None:
    encoder = getattr(model, "timesformer", None)
    if encoder is not None and callable(getattr(encoder, "save_pretrained", None)):
        _save_pretrained(encoder, final_encoder_dir)
    else:
        loaded_encoder = encoder_cls.from_pretrained(str(final_classifier_dir))
        _save_pretrained(loaded_encoder, final_encoder_dir)


def _build_summary(
    config: TimeSformerFineTuneConfig,
    examples: Sequence[MeldVideoEmotionExample],
    split: SplitIndices,
    skipped: _SkipStats,
    final_classifier_dir: Path,
    final_encoder_dir: Path,
) -> TimeSformerFineTuneSummary:
    counts = Counter(example.emotion for example in examples)
    return TimeSformerFineTuneSummary(
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
        num_frames=config.num_frames,
        frame_size=config.frame_size,
        freeze_backbone=config.freeze_backbone,
        on_error=config.on_error,
    )


def _write_summary(path: Path, summary: TimeSformerFineTuneSummary) -> None:
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
