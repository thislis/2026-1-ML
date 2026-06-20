"""EmbeddingGemma fine-tuning on MELD emotion labels."""

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

from meld_emotion.core.types import EMOTION_ORDER, Emotion

EMOTION_LABELS: tuple[str, ...] = tuple(emotion.value for emotion in EMOTION_ORDER)
_LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
_REQUIRED_COLUMNS = frozenset(("Utterance", "Emotion"))


class _DatasetClass(Protocol):
    @classmethod
    def from_dict(cls, mapping: Mapping[str, Sequence[object]]) -> object: ...


class _SentenceTransformerClass(Protocol):
    def __call__(self, model_name: str, **kwargs: object) -> Any: ...


class _TrainerClass(Protocol):
    def __call__(self, **kwargs: object) -> Any: ...


class _TrainingArgumentsClass(Protocol):
    def __call__(self, **kwargs: object) -> Any: ...


class _LossClass(Protocol):
    def __call__(self, model: object) -> object: ...


@dataclass(frozen=True)
class MeldEmotionExample:
    """Single MELD utterance prepared for label-supervised embedding fine-tuning."""

    sentence: str
    emotion: str
    label: int
    dialogue_id: int | None
    utterance_id: int | None


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic train/eval split as original row indices."""

    train: tuple[int, ...]
    eval: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddingGemmaFineTuneConfig:
    """Configuration for MELD emotion-supervised EmbeddingGemma fine-tuning."""

    csv_path: Path = Path("MELD.Raw/train/train_sent_emo.csv")
    model_name: str = "google/embeddinggemma-300m"
    output_dir: Path = Path("outputs/embeddinggemma_meld_finetuned")
    epochs: float = 1.0
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    max_steps: int | None = None
    save_total_limit: int = 2


@dataclass(frozen=True)
class FineTuneSummary:
    """Serializable summary of an EmbeddingGemma fine-tuning run."""

    csv_path: str
    model_name: str
    output_dir: str
    final_model_dir: str
    n_examples: int
    n_train: int
    n_eval: int
    label_counts: dict[str, int]
    label_to_id: dict[str, int]
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


@dataclass(frozen=True)
class _TrainingDependencies:
    dataset_cls: _DatasetClass
    sentence_transformer_cls: _SentenceTransformerClass
    trainer_cls: _TrainerClass
    training_arguments_cls: _TrainingArgumentsClass
    batch_all_triplet_loss_cls: _LossClass
    batch_samplers_cls: Any


def load_meld_emotion_examples(csv_path: str | Path) -> tuple[MeldEmotionExample, ...]:
    """Load MELD utterances and convert emotion labels to the project class order."""

    path = Path(csv_path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"MELD CSV header 를 읽을 수 없습니다: {path}")
        missing = sorted(_REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"MELD CSV 필수 컬럼이 없습니다: {joined}")

        examples: list[MeldEmotionExample] = []
        for row_number, row in enumerate(reader, start=2):
            sentence = row["Utterance"].strip()
            emotion_text = row["Emotion"].strip()
            try:
                emotion = Emotion(emotion_text)
            except ValueError as exc:
                raise ValueError(
                    f"알 수 없는 MELD Emotion 값입니다: row={row_number}, value={emotion_text!r}"
                ) from exc
            if not sentence:
                raise ValueError(f"MELD Utterance 가 비어 있습니다: row={row_number}")
            examples.append(
                MeldEmotionExample(
                    sentence=sentence,
                    emotion=emotion.value,
                    label=_LABEL_TO_ID[emotion.value],
                    dialogue_id=_optional_int(row.get("Dialogue_ID")),
                    utterance_id=_optional_int(row.get("Utterance_ID")),
                )
            )

    if not examples:
        raise ValueError(f"MELD CSV 에 학습 가능한 row 가 없습니다: {path}")
    return tuple(examples)


def stratified_train_eval_split(
    examples: Sequence[MeldEmotionExample],
    eval_fraction: float = 0.1,
    seed: int = 0,
) -> SplitIndices:
    """Create a deterministic stratified split while preserving at least 2 train rows/class."""

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
        raise ValueError(f"MELD fine-tuning 에 필요한 감정 라벨이 없습니다: {labels}")

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for label in range(len(EMOTION_LABELS)):
        indices = list(by_label[label])
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                "BatchAllTripletLoss 는 라벨별 최소 2개 이상의 train 예제가 필요합니다: "
                f"label={EMOTION_LABELS[label]}"
            )
        eval_count = 0
        if eval_fraction > 0.0 and len(indices) > 2:
            eval_count = max(1, round(len(indices) * eval_fraction))
            eval_count = min(eval_count, len(indices) - 2)
        eval_indices.extend(indices[:eval_count])
        train_indices.extend(indices[eval_count:])

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return SplitIndices(train=tuple(train_indices), eval=tuple(eval_indices))


def run_embeddinggemma_fine_tuning(
    config: EmbeddingGemmaFineTuneConfig,
) -> FineTuneSummary:
    """Fine-tune EmbeddingGemma on MELD emotion labels and save a reusable model."""

    _validate_config(config)
    dependencies = _load_training_dependencies()
    examples = load_meld_emotion_examples(config.csv_path)
    split = stratified_train_eval_split(examples, config.eval_fraction, config.seed)
    train_dataset = _dataset_from_indices(dependencies.dataset_cls, examples, split.train)
    eval_dataset = (
        _dataset_from_indices(dependencies.dataset_cls, examples, split.eval) if split.eval else None
    )

    model_kwargs: dict[str, object] = {}
    if config.device != "auto":
        model_kwargs["device"] = config.device
    model = dependencies.sentence_transformer_cls(config.model_name, **model_kwargs)
    loss = dependencies.batch_all_triplet_loss_cls(model)

    config.output_dir.mkdir(parents=True, exist_ok=True)
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
        save_total_limit=config.save_total_limit,
        logging_steps=50,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        batch_sampler=dependencies.batch_samplers_cls.GROUP_BY_LABEL,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = dependencies.trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    final_model_dir = config.output_dir / "final"
    model.save_pretrained(str(final_model_dir))
    summary = _build_summary(config, examples, split, final_model_dir)
    _write_summary(config.output_dir / "training_summary.json", summary)
    return summary


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _validate_config(config: EmbeddingGemmaFineTuneConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs 는 양수여야 합니다")
    if config.batch_size <= 0:
        raise ValueError("batch_size 는 양수여야 합니다")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 는 양수여야 합니다")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio 는 0 이상 1 미만이어야 합니다")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps 는 양수이거나 None 이어야 합니다")
    if config.save_total_limit <= 0:
        raise ValueError("save_total_limit 는 양수여야 합니다")
    if config.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device 는 auto/cpu/mps/cuda 중 하나여야 합니다")


def _dataset_from_indices(
    dataset_cls: _DatasetClass,
    examples: Sequence[MeldEmotionExample],
    indices: Sequence[int],
) -> object:
    selected = [examples[index] for index in indices]
    return dataset_cls.from_dict(
        {
            "sentence": [example.sentence for example in selected],
            "label": [example.label for example in selected],
        }
    )


def _load_training_dependencies() -> _TrainingDependencies:
    try:
        datasets_module = import_module("datasets")
    except ImportError as exc:
        raise ImportError(
            "EmbeddingGemma fine-tuning requires the 'datasets' package. "
            "Install it with `uv sync --extra text`."
        ) from exc
    try:
        sentence_transformers_module = import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            "EmbeddingGemma fine-tuning requires sentence-transformers. "
            "Install it with `uv sync --extra text`."
        ) from exc

    losses_module = _import_first(
        "sentence_transformers.sentence_transformer.losses",
        "sentence_transformers.losses",
    )
    training_args_module = _import_first(
        "sentence_transformers.sentence_transformer.training_args",
        "sentence_transformers.training_args",
    )

    try:
        dataset_cls = datasets_module.__dict__["Dataset"]
        sentence_transformer_cls = sentence_transformers_module.__dict__["SentenceTransformer"]
        trainer_cls = sentence_transformers_module.__dict__["SentenceTransformerTrainer"]
        training_arguments_cls = sentence_transformers_module.__dict__[
            "SentenceTransformerTrainingArguments"
        ]
        batch_all_triplet_loss_cls = losses_module.__dict__["BatchAllTripletLoss"]
        batch_samplers_cls = training_args_module.__dict__["BatchSamplers"]
    except KeyError as exc:
        raise ImportError(
            "EmbeddingGemma fine-tuning requires a recent sentence-transformers release "
            "that exposes SentenceTransformerTrainer, SentenceTransformerTrainingArguments, "
            "BatchAllTripletLoss, and BatchSamplers."
        ) from exc

    return _TrainingDependencies(
        dataset_cls=cast(_DatasetClass, dataset_cls),
        sentence_transformer_cls=cast(_SentenceTransformerClass, sentence_transformer_cls),
        trainer_cls=cast(_TrainerClass, trainer_cls),
        training_arguments_cls=cast(_TrainingArgumentsClass, training_arguments_cls),
        batch_all_triplet_loss_cls=cast(_LossClass, batch_all_triplet_loss_cls),
        batch_samplers_cls=batch_samplers_cls,
    )


def _import_first(*module_names: str) -> Any:
    errors: list[ImportError] = []
    for module_name in module_names:
        try:
            return import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
    names = ", ".join(module_names)
    raise ImportError(f"다음 모듈 중 하나를 import 해야 합니다: {names}") from errors[-1]


def _build_summary(
    config: EmbeddingGemmaFineTuneConfig,
    examples: Sequence[MeldEmotionExample],
    split: SplitIndices,
    final_model_dir: Path,
) -> FineTuneSummary:
    counts = Counter(example.emotion for example in examples)
    return FineTuneSummary(
        csv_path=str(config.csv_path),
        model_name=config.model_name,
        output_dir=str(config.output_dir),
        final_model_dir=str(final_model_dir),
        n_examples=len(examples),
        n_train=len(split.train),
        n_eval=len(split.eval),
        label_counts={label: counts[label] for label in EMOTION_LABELS},
        label_to_id=dict(_LABEL_TO_ID),
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
    )


def _write_summary(path: Path, summary: FineTuneSummary) -> None:
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
