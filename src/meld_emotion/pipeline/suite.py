"""다중 실험 비교 러너 (완전 구현) — 기존 파이프라인 위의 얇은 층.

여러 :class:`ExperimentConfig` 를 받아 각각을 기존 :func:`build_experiment` 로 조립·실행하고,
결과를 하나의 :class:`ComparisonReport` 로 모은다. 핵심 설계:

- **아키텍처 비침습**: ``core`` 계약이나 ``builder``/``runner`` 를 건드리지 않는다. 단일 실험
  실행 경로(``build_experiment(config).run()``)를 그대로 재사용한다.
- **경계 내성(boundary-tolerant)**: 일부 변형이 미구현 경계(``@unimplemented``)에 닿아
  예외를 던져도 비교 전체를 멈추지 않고, 그 변형만 실패로 기록한다(상태 주도 프로젝트와
  자연스럽게 맞물린다 — 구현된 변형끼리 먼저 비교 가능).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from meld_emotion.config.loader import to_dict
from meld_emotion.config.schema import ExperimentConfig
from meld_emotion.core.results import ComparisonReport, ExperimentOutcome
from meld_emotion.core.status import real
from meld_emotion.pipeline.builder import build_experiment
from meld_emotion.pipeline.cache import InMemoryFeatureCache

logger = logging.getLogger(__name__)


@real
class SuiteRunner:
    """여러 실험을 실행해 비교 결과를 만든다."""

    def __init__(self, name: str, configs: Sequence[ExperimentConfig]) -> None:
        if not configs:
            raise ValueError("비교할 실험이 최소 한 개 필요합니다")
        self._name = name
        self._configs = tuple(configs)
        self._feature_caches: dict[str, InMemoryFeatureCache] = {}

    def run(self) -> ComparisonReport:
        logger.info("비교 suite 시작: name=%s experiments=%d", self._name, len(self._configs))
        report = ComparisonReport(
            name=self._name,
            outcomes=tuple(self._run_one(config) for config in self._configs),
        )
        logger.info(
            "비교 suite 완료: name=%s success=%d failed=%d",
            self._name,
            len(report.successful()),
            len(report.failed()),
        )
        return report

    def _run_one(self, config: ExperimentConfig) -> ExperimentOutcome:
        logger.info("suite 실험 시작: %s", config.name)
        try:
            result = build_experiment(config, feature_cache=self._cache_for(config)).run()
        except Exception as exc:  # 미구현 경계 등은 해당 변형만 실패로 기록하고 계속 진행.
            logger.exception("suite 실험 실패: %s", config.name)
            return ExperimentOutcome(name=config.name, error=f"{type(exc).__name__}: {exc}")
        logger.info("suite 실험 완료: %s", config.name)
        return ExperimentOutcome(name=config.name, result=result)

    def _cache_for(self, config: ExperimentConfig) -> InMemoryFeatureCache:
        signature = _feature_signature(config)
        cache = self._feature_caches.get(signature)
        if cache is None:
            cache = InMemoryFeatureCache()
            self._feature_caches[signature] = cache
            logger.info("suite feature cache 생성: experiment=%s signature=%s", config.name, signature)
        else:
            logger.info("suite feature cache 재사용: experiment=%s signature=%s", config.name, signature)
        return cache


def _feature_signature(config: ExperimentConfig) -> str:
    data = to_dict(config)
    payload = {
        "dataset": data["dataset"],
        "extractors": data["extractors"],
        "media": data["media"],
        "train_split": data["train_split"],
        "eval_split": data["eval_split"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
