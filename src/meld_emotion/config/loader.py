"""설정 dataclass ↔ YAML 직렬화.

하이브리드 방식의 YAML 경계. ``type`` 식별자를 읽어 :mod:`meld_emotion.config.schema` 의
레지스트리에서 알맞은 설정 dataclass 를 복원한다. 중첩 설정(model.base, late.combiner,
stacking.meta)도 재귀적으로 처리한다.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from meld_emotion.config.schema import (
    CACHE_CONFIGS,
    COMBINER_CONFIGS,
    DATASET_CONFIGS,
    ESTIMATOR_CONFIGS,
    EXPLAINER_CONFIGS,
    EXTRACTOR_CONFIGS,
    MODEL_CONFIGS,
    REPORTER_CONFIGS,
    CacheConfig,
    CalibrationSettings,
    ClassifierHeadSettings,
    CombinerConfig,
    DatasetConfig,
    DialogueContextSettings,
    DialogueRnnConfig,
    DialogueTrainingSettings,
    DropoutConfig,
    EnsembleConfig,
    EnsembleDistillationSettings,
    EnsembleSettings,
    EstimatorConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExplainerConfig,
    ExtractorConfig,
    FusionSettings,
    HardNegativeMiningSettings,
    LogitAdjustmentSettings,
    LossSettings,
    MediaConfig,
    MemoryAttentionSettings,
    MlpConfig,
    ModalityEncoderSettings,
    ModelConfig,
    MoeConfig,
    MoeExpertSettings,
    MoeSettings,
    NeutralGateSettings,
    RareExpertSettings,
    ReporterConfig,
    StackingCombinerConfig,
    SuiteConfig,
)


def _pop_type(data: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    rest = dict(data)
    try:
        type_name = str(rest.pop("type"))
    except KeyError:
        raise ValueError(f"설정 항목에 'type' 키가 필요합니다: {data!r}") from None
    return type_name, rest


def _dataset(data: Mapping[str, Any]) -> DatasetConfig:
    name, rest = _pop_type(data)
    return DATASET_CONFIGS.create(name, **rest)


def _extractor(data: Mapping[str, Any]) -> ExtractorConfig:
    name, rest = _pop_type(data)
    return EXTRACTOR_CONFIGS.create(name, **rest)


def _estimator(data: Mapping[str, Any]) -> EstimatorConfig:
    name, rest = _pop_type(data)
    if name == MlpConfig.type and "class_weights" in rest:
        rest["class_weights"] = tuple(float(v) for v in rest["class_weights"])
    return ESTIMATOR_CONFIGS.create(name, **rest)


def _combiner(data: Mapping[str, Any]) -> CombinerConfig:
    name, rest = _pop_type(data)
    if name == StackingCombinerConfig.type and "meta" in rest:
        rest["meta"] = _estimator(rest["meta"])
    return COMBINER_CONFIGS.create(name, **rest)


def _model(data: Mapping[str, Any]) -> ModelConfig:
    name, rest = _pop_type(data)
    if name != EnsembleConfig.type and "base" in rest:
        rest["base"] = _estimator(rest["base"])
    if "combiner" in rest:
        rest["combiner"] = _combiner(rest["combiner"])
    if name == EnsembleConfig.type:
        if "base" in rest:
            rest["base"] = _model(rest["base"])
        if "ensemble" in rest:
            ensemble_data = dict(rest["ensemble"])
            if "distillation" in ensemble_data:
                ensemble_data["distillation"] = EnsembleDistillationSettings(
                    **ensemble_data["distillation"]
                )
            rest["ensemble"] = EnsembleSettings(**ensemble_data)
    if name == MoeConfig.type and "moe" in rest:
        moe_data = dict(rest["moe"])
        if "experts" in moe_data:
            moe_data["experts"] = MoeExpertSettings(**moe_data["experts"])
        if "rare_expert" in moe_data:
            rare_data = dict(moe_data["rare_expert"])
            if "target_classes" in rare_data:
                rare_data["target_classes"] = tuple(int(v) for v in rare_data["target_classes"])
            moe_data["rare_expert"] = RareExpertSettings(**rare_data)
        rest["moe"] = MoeSettings(**moe_data)
    if name == DialogueRnnConfig.type:
        if "modality_encoder" in rest:
            rest["modality_encoder"] = ModalityEncoderSettings(**rest["modality_encoder"])
        if "fusion" in rest:
            rest["fusion"] = FusionSettings(**rest["fusion"])
        if "dialogue_context" in rest:
            rest["dialogue_context"] = DialogueContextSettings(**rest["dialogue_context"])
        if "memory_attention" in rest:
            rest["memory_attention"] = MemoryAttentionSettings(**rest["memory_attention"])
        if "classifier" in rest:
            rest["classifier"] = ClassifierHeadSettings(**rest["classifier"])
        if "training" in rest:
            rest["training"] = DialogueTrainingSettings(**rest["training"])
        if "loss" in rest:
            loss_data = dict(rest["loss"])
            if "logit_adjustment" in loss_data:
                loss_data["logit_adjustment"] = LogitAdjustmentSettings(
                    **loss_data["logit_adjustment"]
                )
            if "hard_negative_mining" in loss_data:
                hard_negative_data = dict(loss_data["hard_negative_mining"])
                if "target_classes" in hard_negative_data:
                    hard_negative_data["target_classes"] = tuple(
                        int(v) for v in hard_negative_data["target_classes"]
                    )
                loss_data["hard_negative_mining"] = HardNegativeMiningSettings(**hard_negative_data)
            rest["loss"] = LossSettings(**loss_data)
        if "calibration" in rest:
            calibration_data = dict(rest["calibration"])
            if "rare_classes" in calibration_data:
                calibration_data["rare_classes"] = tuple(
                    int(v) for v in calibration_data["rare_classes"]
                )
            rest["calibration"] = CalibrationSettings(**calibration_data)
        if "neutral_gate" in rest:
            rest["neutral_gate"] = NeutralGateSettings(**rest["neutral_gate"])
    return MODEL_CONFIGS.create(name, **rest)


def _explainer(data: Mapping[str, Any]) -> ExplainerConfig:
    name, rest = _pop_type(data)
    if "kinds" in rest:  # YAML 리스트 → tuple (frozen dataclass 동등성/왕복 보존)
        rest["kinds"] = tuple(rest["kinds"])
    return EXPLAINER_CONFIGS.create(name, **rest)


def _cache(data: Mapping[str, Any]) -> CacheConfig:
    name, rest = _pop_type(data)
    return CACHE_CONFIGS.create(name, **rest)


def _reporter(data: Mapping[str, Any]) -> ReporterConfig:
    name, rest = _pop_type(data)
    return REPORTER_CONFIGS.create(name, **rest)


def _evaluation(data: Mapping[str, Any]) -> EvaluationConfig:
    kwargs: dict[str, Any] = {}
    if "metrics" in data:
        kwargs["metrics"] = tuple(data["metrics"])
    if "scenarios" in data:
        kwargs["scenarios"] = tuple(data["scenarios"])
    if "confusion" in data:
        kwargs["confusion"] = bool(data["confusion"])
    return EvaluationConfig(**kwargs)


def _dropout(data: Mapping[str, Any]) -> DropoutConfig:
    kwargs: dict[str, Any] = {}
    if "drop_prob" in data:
        kwargs["drop_prob"] = float(data["drop_prob"])
    if "seed" in data:
        kwargs["seed"] = int(data["seed"])
    return DropoutConfig(**kwargs)


def _media(data: Mapping[str, Any]) -> MediaConfig:
    kwargs: dict[str, Any] = {}
    if "audio_sample_rate" in data:
        kwargs["audio_sample_rate"] = int(data["audio_sample_rate"])
    if "video_max_frames" in data:
        kwargs["video_max_frames"] = int(data["video_max_frames"])
    if "video_frame_size" in data:
        kwargs["video_frame_size"] = _pair_ints(data["video_frame_size"], "video_frame_size")
    if "on_error" in data:
        kwargs["on_error"] = str(data["on_error"])
    if "max_audio_seconds" in data and data["max_audio_seconds"] is not None:
        kwargs["max_audio_seconds"] = float(data["max_audio_seconds"])
    if "min_audio_seconds" in data and data["min_audio_seconds"] is not None:
        kwargs["min_audio_seconds"] = float(data["min_audio_seconds"])
    return MediaConfig(**kwargs)


def _pair_ints(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} 는 두 정수 값이어야 합니다: {value!r}")
    return (int(value[0]), int(value[1]))


def from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    """평범한 dict(예: YAML 파싱 결과)를 :class:`ExperimentConfig` 로 복원한다."""

    kwargs: dict[str, Any] = {}
    for scalar in ("name", "seed", "output_dir", "train_split", "eval_split"):
        if scalar in data:
            kwargs[scalar] = data[scalar]
    if "dataset" in data:
        kwargs["dataset"] = _dataset(data["dataset"])
    if "extractors" in data:
        kwargs["extractors"] = tuple(_extractor(e) for e in data["extractors"])
    if "model" in data:
        kwargs["model"] = _model(data["model"])
    if data.get("dropout") is not None:
        kwargs["dropout"] = _dropout(data["dropout"])
    if "media" in data:
        kwargs["media"] = _media(data["media"])
    if "evaluation" in data:
        kwargs["evaluation"] = _evaluation(data["evaluation"])
    if "explainers" in data:
        kwargs["explainers"] = tuple(_explainer(x) for x in data["explainers"])
    if "cache" in data:
        kwargs["cache"] = _cache(data["cache"])
    if "reporters" in data:
        kwargs["reporters"] = tuple(_reporter(r) for r in data["reporters"])
    return ExperimentConfig(**kwargs)


def _serialize(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        type_name = getattr(type(obj), "type", None)
        if isinstance(type_name, str) and type_name != "base":
            out["type"] = type_name
        for f in dataclasses.fields(obj):
            out[f.name] = _serialize(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple | list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """설정을 YAML 직렬화 가능한 dict 로 변환한다(``type`` 식별자 포함)."""

    result = _serialize(config)
    assert isinstance(result, dict)
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """YAML 파일에서 실험 설정을 읽는다."""

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"설정 파일 최상위는 매핑이어야 합니다: {path}")
    return from_dict(data)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    """실험 설정을 YAML 파일로 쓴다."""

    Path(path).write_text(
        yaml.safe_dump(to_dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# --- 다중 실험 비교(suite) -------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` 를 ``base`` 위에 재귀 병합한다(매핑은 깊게, 그 외/리스트는 덮어씀).

    공유 설정(``base``)에 변형(experiment)별 차이만 얹는 데 쓴다. 규칙:

    - 매핑끼리는 깊게 병합하되, **``type`` 식별자가 서로 다르면** 베이스의 키는 다른 종류의
      것이므로 통째로 교체한다(예: base ``dataset.type=synthetic`` 위에 변형이
      ``dataset.type=meld`` 를 지정하면 ``n_train`` 같은 synthetic 전용 키를 끌고 오지 않는다).
    - 리스트(예: ``extractors``)는 부분 병합이 모호하므로 통째로 교체한다.
    """

    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if (
            isinstance(existing, Mapping)
            and isinstance(value, Mapping)
            and not _type_conflict(existing, value)
        ):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _type_conflict(base: Mapping[str, Any], override: Mapping[str, Any]) -> bool:
    """두 설정 매핑이 서로 다른 다형성 ``type`` 을 지정하는지 여부."""

    return "type" in base and "type" in override and base["type"] != override["type"]


def suite_from_dict(data: Mapping[str, Any]) -> SuiteConfig:
    """suite dict(공유 ``base`` + ``experiments`` 목록)를 :class:`SuiteConfig` 로 복원한다."""

    base = data.get("base", {})
    if not isinstance(base, Mapping):
        raise ValueError("suite 'base' 는 매핑이어야 합니다")
    raw = data.get("experiments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("suite 에는 비어 있지 않은 'experiments' 리스트가 필요합니다")

    experiments: list[ExperimentConfig] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"각 실험 항목은 매핑이어야 합니다: {entry!r}")
        config = from_dict(_deep_merge(base, entry))
        if config.name in seen:
            raise ValueError(f"실험 이름이 중복됩니다: {config.name!r} (비교표 키가 충돌)")
        seen.add(config.name)
        experiments.append(config)

    kwargs: dict[str, Any] = {"experiments": tuple(experiments)}
    for scalar in ("name", "robustness_metric", "output_path"):
        if scalar in data:
            kwargs[scalar] = data[scalar]
    if "metrics" in data:
        kwargs["metrics"] = tuple(data["metrics"])
    return SuiteConfig(**kwargs)


def load_suite(path: str | Path) -> SuiteConfig:
    """YAML 파일에서 비교 묶음(suite) 설정을 읽는다."""

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"suite 파일 최상위는 매핑이어야 합니다: {path}")
    return suite_from_dict(data)
