"""Wav2Vec2 MELD audio fine-tuning utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest

from meld_emotion.cli import build_parser
from meld_emotion.fine_tunning import wav2vec2 as fine_tune_module
from meld_emotion.fine_tunning.wav2vec2 import (
    EMOTION_LABELS,
    MeldAudioEmotionExample,
    Wav2Vec2FineTuneConfig,
    _TrainingDependencies,
    load_meld_audio_emotion_examples,
    run_wav2vec2_fine_tuning,
    stratified_train_eval_split,
)


def test_load_meld_audio_examples_maps_csv_rows_to_mp4_paths(tmp_path: Path) -> None:
    csv_path, mp4_dir = _write_tiny_audio_meld(tmp_path)

    examples = load_meld_audio_emotion_examples(csv_path, mp4_dir)

    first = examples[0]
    assert first.uid == "0_0"
    assert first.mp4_path == mp4_dir / "dia0_utt0.mp4"
    assert first.emotion == "neutral"
    assert first.label == 0
    assert first.dialogue_id == 0
    assert first.utterance_id == 0
    assert len(examples) == len(EMOTION_LABELS) * 2


def test_load_meld_audio_examples_drops_missing_mp4_by_default(tmp_path: Path) -> None:
    csv_path, mp4_dir = _write_tiny_audio_meld(tmp_path, repeats_per_label=2)
    (mp4_dir / "dia0_utt0.mp4").unlink()

    examples = load_meld_audio_emotion_examples(csv_path, mp4_dir)

    assert len(examples) == len(EMOTION_LABELS) * 2 - 1
    assert all(example.mp4_path.name != "dia0_utt0.mp4" for example in examples)


def test_load_meld_audio_examples_can_fail_fast_on_missing_mp4(tmp_path: Path) -> None:
    csv_path, mp4_dir = _write_tiny_audio_meld(tmp_path)
    (mp4_dir / "dia0_utt0.mp4").unlink()

    with pytest.raises(FileNotFoundError, match=r"dia0_utt0\.mp4"):
        load_meld_audio_emotion_examples(csv_path, mp4_dir, on_error="fail_fast")


def test_load_meld_audio_examples_rejects_bad_csv(tmp_path: Path) -> None:
    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("Emotion,Dialogue_ID\nneutral,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_meld_audio_emotion_examples(missing_columns, tmp_path)

    bad_label = tmp_path / "bad_label.csv"
    bad_label.write_text(
        "Emotion,Dialogue_ID,Utterance_ID\nconfused,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "dia0_utt0.mp4").write_bytes(b"fake")
    with pytest.raises(ValueError, match="알 수 없는 MELD Emotion"):
        load_meld_audio_emotion_examples(bad_label, tmp_path)


def test_stratified_audio_split_is_deterministic_and_preserves_labels() -> None:
    examples = tuple(
        MeldAudioEmotionExample(
            uid=f"{label}-{i}",
            mp4_path=Path(f"dia{label_id}_utt{i}.mp4"),
            emotion=label,
            label=label_id,
            dialogue_id=label_id,
            utterance_id=i,
        )
        for label_id, label in enumerate(EMOTION_LABELS)
        for i in range(5)
    )

    first = stratified_train_eval_split(examples, eval_fraction=0.2, seed=11)
    second = stratified_train_eval_split(examples, eval_fraction=0.2, seed=11)

    assert first == second
    assert len(first.eval) == len(EMOTION_LABELS)
    train_counts = dict.fromkeys(range(len(EMOTION_LABELS)), 0)
    eval_counts = dict.fromkeys(range(len(EMOTION_LABELS)), 0)
    for index in first.train:
        train_counts[examples[index].label] += 1
    for index in first.eval:
        eval_counts[examples[index].label] += 1
    assert set(train_counts.values()) == {4}
    assert set(eval_counts.values()) == {1}


def test_cli_parser_accepts_fine_tune_wav2vec2_command() -> None:
    args = build_parser().parse_args(
        [
            "fine-tune-wav2vec2",
            "--csv",
            "MELD.Raw/train/train_sent_emo.csv",
            "--mp4-dir",
            "MELD.Raw/train/train_splits",
            "--output-dir",
            "outputs/wav2vec2_meld_finetuned",
            "--max-steps",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert args.command == "fine-tune-wav2vec2"
    assert args.model_name == "facebook/wav2vec2-xls-r-300m"
    assert args.batch_size == 4
    assert args.max_steps == 1
    assert args.device == "cpu"
    assert args.early_stopping_metric == "eval_loss"
    assert args.freeze_feature_encoder is True


def test_wav2vec2_dependency_loader_supports_transformers_lazy_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LazyTransformersModule:
        def __getattr__(self, name: str) -> object:
            values: dict[str, object] = {
                "AutoFeatureExtractor": _FakeFeatureExtractor,
                "Wav2Vec2ForSequenceClassification": _FakeSequenceClassifier,
                "Wav2Vec2Model": _FakeEncoder,
                "Trainer": _FakeTrainer,
                "TrainingArguments": _FakeTrainingArguments,
                "EarlyStoppingCallback": _FakeEarlyStoppingCallback,
            }
            try:
                return values[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return _LazyTransformersModule()
        if name == "torch":
            return SimpleNamespace(tensor=lambda values, dtype=None: list(values), long="long")
        raise AssertionError(name)

    monkeypatch.setattr(fine_tune_module, "import_module", fake_import_module)

    dependencies = fine_tune_module._load_training_dependencies()

    assert dependencies.feature_extractor_cls is _FakeFeatureExtractor
    assert dependencies.sequence_classifier_cls is _FakeSequenceClassifier
    assert dependencies.encoder_cls is _FakeEncoder
    assert dependencies.trainer_cls is _FakeTrainer
    assert dependencies.training_arguments_cls is _FakeTrainingArguments
    assert dependencies.early_stopping_callback_cls is _FakeEarlyStoppingCallback


def test_run_wav2vec2_fine_tuning_uses_transformers_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path, mp4_dir = _write_tiny_audio_meld(tmp_path, repeats_per_label=4)
    output_dir = tmp_path / "model"
    dependencies = _fake_dependencies()
    monkeypatch.setattr(fine_tune_module, "_load_training_dependencies", lambda: dependencies)

    summary = run_wav2vec2_fine_tuning(
        Wav2Vec2FineTuneConfig(
            csv_path=csv_path,
            mp4_dir=mp4_dir,
            output_dir=output_dir,
            batch_size=2,
            eval_fraction=0.5,
            eval_steps=5,
            early_stopping_metric="eval_weighted_f1",
            early_stopping_patience=2,
            max_steps=1,
            device="cpu",
        )
    )

    trainer = cast(_FakeTrainer, _FakeTrainer.last)
    assert trainer.train_called is True
    assert isinstance(trainer.kwargs["train_dataset"], fine_tune_module.MeldAudioDataset)
    assert isinstance(trainer.kwargs["eval_dataset"], fine_tune_module.MeldAudioDataset)
    assert callable(trainer.kwargs["compute_metrics"])
    model = trainer.kwargs["model"]
    assert isinstance(model, _FakeSequenceClassifier)
    assert model.freeze_called is True
    assert model.device == "cpu"
    args = trainer.kwargs["args"]
    assert isinstance(args, _FakeTrainingArguments)
    assert args.kwargs["remove_unused_columns"] is False
    assert args.kwargs["eval_strategy"] == "steps"
    assert args.kwargs["eval_steps"] == 5
    assert args.kwargs["metric_for_best_model"] == "eval_weighted_f1"
    assert args.kwargs["greater_is_better"] is True
    assert _FakeEarlyStoppingCallback.created == [(2, 0.0)]
    assert summary.final_classifier_dir == str(output_dir / "final_classifier")
    assert summary.final_encoder_dir == str(output_dir / "final_encoder")
    assert (output_dir / "final_classifier" / "fake-classifier.txt").exists()
    assert (output_dir / "final_encoder" / "fake-encoder.txt").exists()
    assert (output_dir / "final_encoder" / "fake-feature-extractor.txt").exists()
    payload = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert payload["n_examples"] == len(EMOTION_LABELS) * 4
    assert payload["n_train"] == len(EMOTION_LABELS) * 2
    assert payload["n_eval"] == len(EMOTION_LABELS) * 2
    assert payload["skipped"] == {"missing_mp4": 0, "decode_error": 0, "empty_batch": 0}


def _write_tiny_audio_meld(
    tmp_path: Path,
    repeats_per_label: int = 2,
) -> tuple[Path, Path]:
    mp4_dir = tmp_path / "train_splits"
    mp4_dir.mkdir()
    csv_path = tmp_path / "train_sent_emo.csv"
    rows = ["Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID"]
    serial = 1
    for repeat in range(repeats_per_label):
        for label_id, label in enumerate(EMOTION_LABELS):
            dialogue_id = repeat
            utterance_id = label_id
            rows.append(
                f"{serial},{label} sample,speaker,{label},neutral,{dialogue_id},{utterance_id}"
            )
            (mp4_dir / f"dia{dialogue_id}_utt{utterance_id}.mp4").write_bytes(b"fake")
            serial += 1
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path, mp4_dir


class _FakeFeatureExtractor:
    created: ClassVar[list[str]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> _FakeFeatureExtractor:
        del kwargs
        cls.created.append(model_name)
        return cls()

    def __call__(self, waveforms: Sequence[object], **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"input_values": list(waveforms)}

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "fake-feature-extractor.txt").write_text("feature-extractor", encoding="utf-8")


class _FakeEncoder:
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> _FakeEncoder:
        del model_name, kwargs
        return cls()

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "fake-encoder.txt").write_text("encoder", encoding="utf-8")


class _FakeSequenceClassifier:
    created: ClassVar[list[tuple[str, dict[str, object]]]] = []

    def __init__(self) -> None:
        self.wav2vec2 = _FakeEncoder()
        self.freeze_called = False
        self.device = "auto"

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> _FakeSequenceClassifier:
        cls.created.append((model_name, dict(kwargs)))
        return cls()

    def freeze_feature_encoder(self) -> None:
        self.freeze_called = True

    def to(self, device: str) -> None:
        self.device = device

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "fake-classifier.txt").write_text("classifier", encoding="utf-8")


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


class _FakeEarlyStoppingCallback:
    created: ClassVar[list[tuple[int, float]]] = []

    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        _FakeEarlyStoppingCallback.created.append(
            (early_stopping_patience, early_stopping_threshold)
        )


def _fake_dependencies() -> _TrainingDependencies:
    _FakeFeatureExtractor.created = []
    _FakeSequenceClassifier.created = []
    _FakeTrainer.last = None
    _FakeEarlyStoppingCallback.created = []
    return _TrainingDependencies(
        feature_extractor_cls=_FakeFeatureExtractor,
        sequence_classifier_cls=_FakeSequenceClassifier,
        encoder_cls=_FakeEncoder,
        trainer_cls=_FakeTrainer,
        training_arguments_cls=_FakeTrainingArguments,
        early_stopping_callback_cls=_FakeEarlyStoppingCallback,
        torch=SimpleNamespace(tensor=lambda values, dtype=None: list(values), long="long"),
    )
