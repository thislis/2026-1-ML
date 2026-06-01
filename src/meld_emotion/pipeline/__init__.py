"""파이프라인 오케스트레이션과 구성 루트.

조립 방식은 ``pipeline/README.md`` 참고. :func:`build_experiment` 가 설정을 받아
:class:`ExperimentRunner` 를 생성하는 단일 진입점이다.
"""

from __future__ import annotations

from meld_emotion.pipeline.builder import build_experiment
from meld_emotion.pipeline.cache import (
    DiskFeatureCache,
    InMemoryFeatureCache,
    NullFeatureCache,
)
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline
from meld_emotion.pipeline.runner import ExperimentRunner
from meld_emotion.pipeline.suite import SuiteRunner

__all__ = [
    "DiskFeatureCache",
    "ExperimentRunner",
    "FeaturePipeline",
    "InMemoryFeatureCache",
    "NullFeatureCache",
    "SuiteRunner",
    "build_experiment",
]
