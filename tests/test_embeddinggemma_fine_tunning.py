"""EmbeddingGemma MELD fine-tuning utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest

from meld_emotion.cli import build_parser
from meld_emotion.fine_tunning import embeddinggemma as fine_tune_module
from meld_emotion.fine_tunning.embeddinggemma import (
    EMOTION_LABELS,
    EmbeddingGemmaFineTuneConfig,
    MeldCentroidF1Evaluator,
    MeldEmotionExample,
    _load_early_stopping_callback_class,
    _TrainingDependencies,
    load_meld_emotion_examples,
    run_embeddinggemma_fine_tuning,
    stratified_train_eval_split,
)

_ROOT = Path(__file__).resolve().parents[1]
_MELD_TRAIN_CSV = _ROOT / "MELD.Raw" / "train" / "train_sent_emo.csv"


def test_load_meld_train_csv_has_expected_rows_and_labels() -> None:
    if not _MELD_TRAIN_CSV.exists():
        pytest.skip("MELD.Raw train CSV is not available")

    examples = load_meld_emotion_examples(_MELD_TRAIN_CSV)
    counts = {label: sum(example.emotion == label for example in examples) for label in EMOTION_LABELS}

    assert len(examples) == 9989
    assert counts == {
        "neutral": 4710,
        "joy": 1743,
        "sadness": 683,
        "anger": 1109,
        "surprise": 1205,
        "fear": 268,
        "disgust": 271,
    }


def test_label_encoding_matches_project_emotion_order(tmp_path: Path) -> None:
    path = _write_tiny_meld_csv(tmp_path)

    examples = load_meld_emotion_examples(path)

    assert tuple(example.emotion for example in examples[: len(EMOTION_LABELS)]) == EMOTION_LABELS
    assert tuple(example.label for example in examples[: len(EMOTION_LABELS)]) == tuple(
        range(len(EMOTION_LABELS))
    )


def test_load_meld_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("Utterance\nhello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="필수 컬럼"):
        load_meld_emotion_examples(path)


def test_stratified_split_is_deterministic_and_preserves_train_pairs() -> None:
    examples = tuple(
        MeldEmotionExample(
            sentence=f"{label}-{i}",
            emotion=label,
            label=label_id,
            dialogue_id=None,
            utterance_id=None,
        )
        for label_id, label in enumerate(EMOTION_LABELS)
        for i in range(5)
    )

    first = stratified_train_eval_split(examples, eval_fraction=0.2, seed=7)
    second = stratified_train_eval_split(examples, eval_fraction=0.2, seed=7)

    assert first == second
    assert len(first.eval) == len(EMOTION_LABELS)
    train_counts = dict.fromkeys(range(len(EMOTION_LABELS)), 0)
    for index in first.train:
        train_counts[examples[index].label] += 1
    assert set(train_counts.values()) == {4}


def test_cli_parser_accepts_fine_tune_embeddinggemma_command() -> None:
    args = build_parser().parse_args(
        [
            "fine-tune-embeddinggemma",
            "--csv",
            "MELD.Raw/train/train_sent_emo.csv",
            "--output-dir",
            "outputs/embeddinggemma_meld_finetuned",
            "--max-steps",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert args.command == "fine-tune-embeddinggemma"
    assert args.model_name == "google/embeddinggemma-300m"
    assert args.batch_size == 16
    assert args.max_steps == 1
    assert args.device == "cpu"
    assert args.early_stopping_metric == "eval_loss"
    assert args.eval_steps == 100


def test_run_embeddinggemma_fine_tuning_uses_sentence_transformer_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _write_tiny_meld_csv(tmp_path)
    output_dir = tmp_path / "model"
    dependencies = _fake_dependencies()
    monkeypatch.setattr(
        fine_tune_module,
        "_load_training_dependencies",
        lambda: dependencies,
    )

    summary = run_embeddinggemma_fine_tuning(
        EmbeddingGemmaFineTuneConfig(
            csv_path=csv_path,
            output_dir=output_dir,
        batch_size=4,
        eval_fraction=0.0,
        early_stopping_metric="none",
        max_steps=1,
        device="cpu",
    )
    )

    trainer = cast(_FakeTrainer, _FakeTrainer.last)
    assert trainer.train_called is True
    assert trainer.kwargs["eval_dataset"] is None
    loss = trainer.kwargs["loss"]
    assert isinstance(loss, _FakeLoss)
    assert loss.model is trainer.kwargs["model"]
    args = trainer.kwargs["args"]
    assert isinstance(args, _FakeTrainingArguments)
    assert args.kwargs["batch_sampler"] == "group_by_label"
    assert args.kwargs["max_steps"] == 1
    assert args.kwargs["eval_strategy"] == "no"
    assert summary.final_model_dir == str(output_dir / "final")
    payload = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert payload["n_examples"] == len(EMOTION_LABELS) * 2
    assert payload["n_train"] == len(EMOTION_LABELS) * 2


def test_run_embeddinggemma_fine_tuning_configures_eval_loss_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeTrainer.last = None
    _FakeEarlyStoppingCallback.created = []
    dependencies = _fake_dependencies()
    monkeypatch.setattr(fine_tune_module, "_load_training_dependencies", lambda: dependencies)

    run_embeddinggemma_fine_tuning(
        EmbeddingGemmaFineTuneConfig(
            csv_path=_write_tiny_meld_csv(tmp_path, repeats_per_label=4),
            output_dir=tmp_path / "model",
            eval_fraction=0.5,
            eval_steps=5,
            early_stopping_metric="eval_loss",
            early_stopping_patience=2,
            early_stopping_threshold=0.01,
            max_steps=1,
        )
    )

    trainer = cast(_FakeTrainer, _FakeTrainer.last)
    args = trainer.kwargs["args"]
    assert isinstance(args, _FakeTrainingArguments)
    assert args.kwargs["eval_strategy"] == "steps"
    assert args.kwargs["eval_steps"] == 5
    assert args.kwargs["save_steps"] == 5
    assert args.kwargs["load_best_model_at_end"] is True
    assert args.kwargs["metric_for_best_model"] == "eval_loss"
    assert args.kwargs["greater_is_better"] is False
    assert _FakeEarlyStoppingCallback.created == [(2, 0.01)]
    assert trainer.kwargs["callbacks"] == [_FakeEarlyStoppingCallback.last]
    assert trainer.kwargs["evaluator"] is None


def test_run_embeddinggemma_fine_tuning_configures_macro_f1_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeTrainer.last = None
    dependencies = _fake_dependencies()
    monkeypatch.setattr(fine_tune_module, "_load_training_dependencies", lambda: dependencies)

    run_embeddinggemma_fine_tuning(
        EmbeddingGemmaFineTuneConfig(
            csv_path=_write_tiny_meld_csv(tmp_path, repeats_per_label=4),
            output_dir=tmp_path / "model",
            eval_fraction=0.5,
            early_stopping_metric="eval_macro_f1",
            max_steps=1,
        )
    )

    trainer = cast(_FakeTrainer, _FakeTrainer.last)
    args = trainer.kwargs["args"]
    assert isinstance(args, _FakeTrainingArguments)
    assert args.kwargs["metric_for_best_model"] == "eval_macro_f1"
    assert args.kwargs["greater_is_better"] is True
    evaluators = trainer.kwargs["evaluator"]
    assert isinstance(evaluators, list)
    evaluator = evaluators[0]
    assert isinstance(evaluator, MeldCentroidF1Evaluator)
    assert evaluator.primary_metric == "macro_f1"


def test_centroid_f1_evaluator_scores_nearest_centroid_predictions() -> None:
    train_examples = (
        MeldEmotionExample("neutral train", "neutral", 0, None, None),
        MeldEmotionExample("joy train", "joy", 1, None, None),
        MeldEmotionExample("neutral train 2", "neutral", 0, None, None),
        MeldEmotionExample("joy train 2", "joy", 1, None, None),
        *tuple(
            MeldEmotionExample(f"{label} train", label, label_id, None, None)
            for label_id, label in enumerate(EMOTION_LABELS[2:], start=2)
            for _ in range(2)
        ),
    )
    eval_examples = (
        MeldEmotionExample("neutral eval", "neutral", 0, None, None),
        MeldEmotionExample("joy eval", "joy", 1, None, None),
    )
    model = _FakeEncodingModel(
        {
            "neutral": [1.0, 0.0],
            "joy": [0.0, 1.0],
            "sadness": [-1.0, 0.0],
            "anger": [0.0, -1.0],
            "surprise": [0.7, 0.7],
            "fear": [-0.7, 0.7],
            "disgust": [0.7, -0.7],
        }
    )

    metrics = MeldCentroidF1Evaluator(
        train_examples=train_examples,
        eval_examples=eval_examples,
        metric_name="macro_f1",
        batch_size=4,
    )(model)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(2.0 / 7.0)
    assert metrics["weighted_f1"] == 1.0


def test_early_stopping_callback_loader_falls_back_to_trainer_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_callback = object()

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace()
        if name == "transformers.trainer_callback":
            return SimpleNamespace(EarlyStoppingCallback=fake_callback)
        raise AssertionError(name)

    monkeypatch.setattr(fine_tune_module, "import_module", fake_import_module)

    assert _load_early_stopping_callback_class() is fake_callback


def _write_tiny_meld_csv(tmp_path: Path, repeats_per_label: int = 2) -> Path:
    path = tmp_path / "train_sent_emo.csv"
    rows = ["Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID"]
    serial = 1
    for repeat in range(repeats_per_label):
        for label in EMOTION_LABELS:
            rows.append(f"{serial},{label} sample {repeat},speaker,{label},neutral,0,{serial}")
            serial += 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


class _FakeDataset:
    def __init__(self, mapping: Mapping[str, Sequence[object]]) -> None:
        self.mapping = mapping

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Sequence[object]]) -> _FakeDataset:
        return cls(mapping)


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "fake-model.txt").write_text(self.model_name, encoding="utf-8")


class _FakeTrainer:
    last: ClassVar[_FakeTrainer | None] = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.train_called = False
        _FakeTrainer.last = self

    def train(self) -> None:
        self.train_called = True


class _FakeTrainingArguments:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeLoss:
    def __init__(self, model: object) -> None:
        self.model = model


class _FakeBatchSamplers:
    GROUP_BY_LABEL: ClassVar[object] = "group_by_label"


class _FakeEarlyStoppingCallback:
    created: ClassVar[list[tuple[int, float]]] = []
    last: ClassVar[_FakeEarlyStoppingCallback | None] = None

    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        _FakeEarlyStoppingCallback.created.append(
            (early_stopping_patience, early_stopping_threshold)
        )
        _FakeEarlyStoppingCallback.last = self


class _FakeEncodingModel:
    def __init__(self, vectors_by_label: Mapping[str, Sequence[float]]) -> None:
        self._vectors_by_label = vectors_by_label

    def encode(self, sentences: Sequence[str], **kwargs: object) -> list[list[float]]:
        del kwargs
        vectors: list[list[float]] = []
        for sentence in sentences:
            label = sentence.split()[0]
            vectors.append(list(self._vectors_by_label[label]))
        return vectors


def _fake_dependencies() -> _TrainingDependencies:
    return _TrainingDependencies(
        dataset_cls=_FakeDataset,
        sentence_transformer_cls=_FakeSentenceTransformer,
        trainer_cls=_FakeTrainer,
        training_arguments_cls=_FakeTrainingArguments,
        batch_all_triplet_loss_cls=_FakeLoss,
        batch_samplers_cls=_FakeBatchSamplers,
        early_stopping_callback_cls=_FakeEarlyStoppingCallback,
    )
