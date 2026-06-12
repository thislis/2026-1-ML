"""설정 dataclass ↔ YAML 왕복 변환."""

from __future__ import annotations

from pathlib import Path

from meld_emotion.config.loader import dump_config, from_dict, load_config, load_suite, to_dict
from meld_emotion.config.schema import (
    CalibrationSettings,
    ClassifierHeadSettings,
    DialogueContextSettings,
    DialogueFineGrainedXaiConfig,
    DialogueRnnConfig,
    EarlyFusionConfig,
    ExperimentConfig,
    FusionSettings,
    HardNegativeMiningSettings,
    LateFusionConfig,
    LinearRegressionConfig,
    LogitAdjustmentSettings,
    LossSettings,
    MeanCombinerConfig,
    MediaConfig,
    MemoryAttentionSettings,
    MfccConfig,
    MlpConfig,
    SvmConfig,
    SvmMarginTwoStageConfig,
    SyntheticConfig,
    TextConceptConfig,
    TextTokenEmbeddingConfig,
    TfidfConfig,
    VideoFrameEmbeddingConfig,
    Wav2Vec2XlsrAudioSequenceConfig,
)


def _complex_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="rt",
        seed=1,
        dataset=SyntheticConfig(n_train=10),
        extractors=(TextConceptConfig(), TfidfConfig(max_features=100), MfccConfig(n_mfcc=5)),
        model=EarlyFusionConfig(base=SvmConfig(C=2.0), use_concepts=False),
        media=MediaConfig(
            audio_sample_rate=8000,
            video_max_frames=8,
            video_frame_size=(32, 48),
            max_audio_seconds=60.0,
            min_audio_seconds=0.025,
        ),
    )


def test_dict_roundtrip() -> None:
    config = _complex_config()
    assert from_dict(to_dict(config)) == config


def test_late_fusion_roundtrip() -> None:
    config = ExperimentConfig(
        name="late",
        model=LateFusionConfig(base=SvmConfig(), combiner=MeanCombinerConfig()),
    )
    assert from_dict(to_dict(config)) == config


def test_new_baseline_configs_roundtrip() -> None:
    from meld_emotion.config.schema import XGBoostConfig

    for base in (
        LinearRegressionConfig(alpha=0.01, fit_intercept=False),
        XGBoostConfig(n_estimators=10, max_depth=3, learning_rate=0.2),
        MlpConfig(
            hidden_dim=32,
            dropout=0.1,
            validation_split=0.2,
            class_weight="balanced",
            class_weights=(1.0, 2.0, 3.0),
        ),
    ):
        config = ExperimentConfig(name="base", model=EarlyFusionConfig(base=base))
        assert from_dict(to_dict(config)) == config


def test_svm_margin_two_stage_config_roundtrip() -> None:
    config = ExperimentConfig(
        name="svm_margin_two_stage",
        model=SvmMarginTwoStageConfig(
            stage1=SvmConfig(C=2.0, kernel="linear"),
            stage2=EarlyFusionConfig(base=LinearRegressionConfig(alpha=0.01)),
            margin_threshold=0.33,
            stage1_use_concepts=False,
        ),
    )
    assert from_dict(to_dict(config)) == config


def test_dialogue_ablation_toggles_roundtrip() -> None:
    config = ExperimentConfig(
        name="dialogue_ablation",
        model=DialogueRnnConfig(
            fusion=FusionSettings(use_interaction=False, use_interaction_features=False),
            dialogue_context=DialogueContextSettings(use_context=False, use_speaker=False),
            memory_attention=MemoryAttentionSettings(use_memory=False, enabled=False),
            classifier=ClassifierHeadSettings(
                classifier_head_type="gated_residual",
                use_context=False,
                use_memory=False,
                gate_hidden_dim=64,
                gate_dropout=0.2,
            ),
        ),
    )
    assert from_dict(to_dict(config)) == config


def test_dialogue_loss_config_roundtrip() -> None:
    config = ExperimentConfig(
        name="dialogue_loss",
        model=DialogueRnnConfig(
            loss=LossSettings(
                type="class_balanced_focal",
                gamma=1.5,
                class_balanced_beta=0.99,
                label_smoothing=0.05,
                logit_adjustment=LogitAdjustmentSettings(enabled=True, tau=0.5),
                hard_negative_mining=HardNegativeMiningSettings(
                    enabled=True,
                    weight=2.0,
                    target_classes=(1, 3),
                ),
            )
        ),
    )
    assert from_dict(to_dict(config)) == config


def test_dialogue_calibration_config_roundtrip() -> None:
    config = ExperimentConfig(
        name="dialogue_calibration",
        model=DialogueRnnConfig(
            calibration=CalibrationSettings(
                enabled=True,
                temperature_scaling=True,
                threshold_tuning=True,
                rare_class_margin_enabled=True,
                rare_classes=(5, 6),
                rare_class_threshold=0.25,
                rare_class_margin=0.1,
            )
        ),
    )
    assert from_dict(to_dict(config)) == config


def test_finegrained_xai_configs_roundtrip() -> None:
    config = ExperimentConfig(
        name="xai",
        extractors=(
            TextTokenEmbeddingConfig(max_tokens=12, output_dim=32),
            Wav2Vec2XlsrAudioSequenceConfig(max_steps=8, output_dim=16),
            VideoFrameEmbeddingConfig(num_frames=4, output_dim=24),
        ),
        explainers=(DialogueFineGrainedXaiConfig(n_steps=4, top_k=3, max_targets=2),),
    )
    assert from_dict(to_dict(config)) == config


def test_yaml_file_roundtrip(tmp_path: Path) -> None:
    config = _complex_config()
    path = tmp_path / "exp.yaml"
    dump_config(config, path)
    assert load_config(path) == config


def test_example_configs_load() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in (
        "example_synthetic.yaml",
        "example_meld_early_svm.yaml",
        "example_meld_dialogue_rnn.yaml",
        "meld_sequence_dialogue_rnn.yaml",
        "two_stage_svm_dialogue_rnn.yaml",
    ):
        config = load_config(root / name)
        assert isinstance(config, ExperimentConfig)
        assert config.name


def test_ablation_core_suite_loads_with_required_experiments() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    suite = load_suite(root / "ablation_core.yaml")
    names = {config.name for config in suite.experiments}
    assert names == {
        "text_only_svm",
        "text_only_logreg",
        "text_audio_svm",
        "text_video_svm",
        "all_svm",
        "text_only_mlp",
        "all_mlp",
        "dialogue_rnn_fused_only",
        "dialogue_rnn_fused_context",
        "dialogue_rnn_fused_context_memory",
        "dialogue_rnn_no_speaker",
        "dialogue_rnn_no_interaction",
    }


def test_meld_sequence_dialogue_rnn_uses_sequence_extractors_and_separate_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "meld_sequence_dialogue_rnn.yaml")
    assert [type(extractor).type for extractor in config.extractors] == [
        "text_token_embeddings",
        "audio_wav2vec2_xlsr_sequence",
        "video_frame_embeddings",
    ]
    assert isinstance(config.model, DialogueRnnConfig)
    assert (
        config.model.training.best_checkpoint_path == "outputs/meld_sequence_dialogue_rnn_best.pt"
    )
    assert config.model.training.best_checkpoint_path != "outputs/best_model.pt"
