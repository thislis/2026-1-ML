"""SVM artifact save/load and inference XAI."""

from __future__ import annotations

from pathlib import Path

from meld_emotion.config.loader import load_config
from meld_emotion.config.schema import (
    BowTextConfig,
    EarlyFusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    SvmConfig,
    SyntheticConfig,
)
from meld_emotion.inference import run_inference
from meld_emotion.inference_svm_batch import run_svm_batch_inference
from meld_emotion.models.artifact import load_classifier_artifact
from meld_emotion.pipeline.builder import build_experiment


def test_svm_training_saves_artifact_and_inference_uses_it(tmp_path: Path) -> None:
    artifact_path = tmp_path / "svm.pkl"
    config = ExperimentConfig(
        name="svm_artifact_smoke",
        dataset=SyntheticConfig(n_train=80, n_test=20, with_audio=False, with_video=False),
        extractors=(BowTextConfig(n_features=32),),
        model=EarlyFusionConfig(
            artifact_path=str(artifact_path),
            base=SvmConfig(C=0.5, kernel="linear"),
            use_concepts=False,
        ),
        evaluation=EvaluationConfig(metrics=("accuracy",), confusion=False, scenarios=("full",)),
        reporters=(),
    )

    build_experiment(config).run()

    artifact = load_classifier_artifact(artifact_path)
    assert artifact.config.name == "svm_artifact_smoke"
    assert artifact.metadata["classifier"] == "EarlyFusionClassifier"

    mp4 = tmp_path / "sample.mp4"
    mp4.write_bytes(b"placeholder")
    result = run_inference(
        mp4,
        "I am happy and joyful",
        checkpoint_path=artifact_path,
        device="cpu",
        include_xai=True,
        top_k=3,
        xai_top_k=4,
    )

    assert result.svm_xai is not None
    assert result.xai == ()
    assert len(result.svm_xai.top_features) <= 4
    assert len(result.svm_xai.top_text_units) <= 4
    assert {unit.label for unit in result.svm_xai.top_text_units}
    assert result.svm_xai.modality[0].modality.value == "text"


def test_requested_single_run_yaml_loads() -> None:
    config = load_config(
        "configs/test/finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm.yaml"
    )

    assert config.name == "finetuned_embeddinggemma_finetuned_wav2vec2_original_timesformer_svm"
    assert isinstance(config.model, EarlyFusionConfig)
    assert isinstance(config.model.base, SvmConfig)
    assert config.model.artifact_path is not None
    assert config.extractors[0].type == "text_embeddinggemma"
    assert config.extractors[1].type == "audio_wav2vec2_xlsr"
    assert config.extractors[2].type == "video_timesformer"


def test_svm_batch_inference_writes_jsonl_and_summary(tmp_path: Path) -> None:
    artifact_path = tmp_path / "svm.pkl"
    config = ExperimentConfig(
        name="svm_batch_smoke",
        dataset=SyntheticConfig(n_train=80, n_test=20, with_audio=False, with_video=False),
        extractors=(BowTextConfig(n_features=32),),
        model=EarlyFusionConfig(
            artifact_path=str(artifact_path),
            base=SvmConfig(C=0.5, kernel="linear"),
            use_concepts=False,
        ),
        evaluation=EvaluationConfig(metrics=("accuracy",), confusion=False, scenarios=("full",)),
        reporters=(),
    )
    build_experiment(config).run()

    root = tmp_path / "MELD.Raw"
    mp4_dir = root / "output_repeated_splits_test"
    mp4_dir.mkdir(parents=True)
    (mp4_dir / "dia0_utt0.mp4").write_bytes(b"placeholder")
    (mp4_dir / "dia0_utt1.mp4").write_bytes(b"placeholder")
    csv = root / "test_sent_emo.csv"
    csv.write_text(
        "\n".join(
            [
                "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime",
                "1,I am happy,Joey,joy,positive,0,0,1,1,\"00:00:00,000\",\"00:00:01,000\"",
                "2,I am angry,Rachel,anger,negative,0,1,1,1,\"00:00:01,000\",\"00:00:02,000\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_svm_batch_inference(
        csv_path=csv,
        mp4_dir=mp4_dir,
        checkpoint_path=artifact_path,
        device="cpu",
        predictions_path=tmp_path / "predictions.jsonl",
        summary_path=tmp_path / "summary.json",
        xai_top_k=2,
    )

    lines = result.paths.predictions.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"uid": "test:0_0"' in lines[0]
    assert '"svm_xai"' in lines[0]
    assert result.summary["n_completed"] == 2
    assert result.paths.summary.exists()
