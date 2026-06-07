"""Progress logging behavior."""

from __future__ import annotations

import logging

from meld_emotion.config.schema import (
    BowTextConfig,
    EarlyFusionConfig,
    EvaluationConfig,
    ExperimentConfig,
    NearestCentroidConfig,
    NullCacheConfig,
    SyntheticConfig,
)
from meld_emotion.pipeline.builder import build_experiment


def test_runner_emits_progress_logs(caplog) -> None:
    config = ExperimentConfig(
        name="logged",
        dataset=SyntheticConfig(n_train=30, n_dev=0, n_test=20),
        extractors=(BowTextConfig(n_features=12),),
        model=EarlyFusionConfig(base=NearestCentroidConfig()),
        evaluation=EvaluationConfig(metrics=("accuracy",), scenarios=()),
        cache=NullCacheConfig(),
        reporters=(),
    )

    with caplog.at_level(logging.INFO, logger="meld_emotion"):
        result = build_experiment(config).run()

    messages = [record.getMessage() for record in caplog.records]
    assert result.name == "logged"
    assert any("실험 시작: logged" in message for message in messages)
    assert any("데이터 적재 완료" in message for message in messages)
    assert any("특징 변환 완료" in message for message in messages)
    assert any("평가 완료" in message for message in messages)
    assert any("실험 완료: logged" in message for message in messages)
