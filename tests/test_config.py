"""설정 dataclass ↔ YAML 왕복 변환."""

from __future__ import annotations

from pathlib import Path

from meld_emotion.config.loader import dump_config, from_dict, load_config, to_dict
from meld_emotion.config.schema import (
    EarlyFusionConfig,
    ExperimentConfig,
    LateFusionConfig,
    LinearRegressionConfig,
    MeanCombinerConfig,
    MediaConfig,
    MfccConfig,
    SvmConfig,
    SyntheticConfig,
    TextConceptConfig,
    TfidfConfig,
)


def _complex_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="rt",
        seed=1,
        dataset=SyntheticConfig(n_train=10),
        extractors=(TextConceptConfig(), TfidfConfig(max_features=100), MfccConfig(n_mfcc=5)),
        model=EarlyFusionConfig(base=SvmConfig(C=2.0), use_concepts=False),
        media=MediaConfig(audio_sample_rate=8000, video_max_frames=8, video_frame_size=(32, 48)),
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
    ):
        config = ExperimentConfig(name="base", model=EarlyFusionConfig(base=base))
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
    ):
        config = load_config(root / name)
        assert isinstance(config, ExperimentConfig)
        assert config.name
