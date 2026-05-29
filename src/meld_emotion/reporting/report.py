"""실험 결과 리포터.

:class:`JsonReporter` 와 :class:`ConsoleReporter` 는 완전 구현이다.
:class:`DashboardExporter` 는 제안서의 case-study 대시보드용 JSON 데이터 구조를 내보내지만
실제 시각화 렌더링은 미구현이라 임시(placeholder)로 표시한다.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.results import EvaluationReport, ExperimentResult
from meld_emotion.core.status import note_placeholder_use, placeholder, real


def _key(key: object) -> str:
    return key.value if isinstance(key, Enum) else str(key)


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {_key(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


@real
class JsonReporter:
    """실험 결과 전체를 JSON 파일로 저장."""

    def __init__(self, path: str = "outputs/result.json") -> None:
        self._path = Path(path)

    def save(self, result: ExperimentResult) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@real
class ConsoleReporter:
    """실험 요약을 콘솔에 출력."""

    def save(self, result: ExperimentResult) -> None:
        print(self.format(result))

    def format(self, result: ExperimentResult) -> str:
        lines: list[str] = [f"=== Experiment: {result.name} ==="]
        meta = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
        if meta:
            lines.append(meta)
        lines.append("[Evaluation: full]")
        lines.extend(self._format_metrics(result.evaluation))

        if result.robustness is not None:
            lines.append("[Robustness]")
            for report in result.robustness.reports:
                acc = report.metric("accuracy")
                f1 = report.metric("macro_f1")
                acc_s = f"{acc.value:.3f}" if acc else "-"
                f1_s = f"{f1.value:.3f}" if f1 else "-"
                lines.append(f"  {report.scenario:12} acc={acc_s} macro_f1={f1_s}")

        explanation = result.explanation
        if explanation is not None:
            if explanation.modality_contributions:
                lines.append("[Modality contribution (score drop)]")
                for mc in explanation.modality_contributions:
                    lines.append(f"  {mc.modality.value:6} drop={mc.score_drop:+.3f}")
            if explanation.feature_contributions:
                lines.append("[Top feature contributions]")
                for fc in explanation.feature_contributions[:8]:
                    lines.append(f"  {fc.modality.value:6} {fc.name:24} imp={fc.importance:+.3f}")
        return "\n".join(lines)

    @staticmethod
    def _format_metrics(evaluation: EvaluationReport) -> list[str]:
        return [f"  {metric.name:18} {metric.value:.4f}" for metric in evaluation.metrics]


@placeholder("case-study 대시보드 시각화 렌더링 미구현 — 현재는 JSON 데이터 구조만 내보냄")
class DashboardExporter:
    """case-study 대시보드 데이터(JSON) 내보내기(임시: 렌더링 없음)."""

    def __init__(self, path: str = "outputs/dashboard.json") -> None:
        self._path = Path(path)

    def save(self, result: ExperimentResult) -> None:
        note_placeholder_use(self)
        payload = {
            "name": result.name,
            "metrics": _jsonable(result.evaluation.metrics),
            "modality_contributions": _jsonable(
                result.explanation.modality_contributions if result.explanation else ()
            ),
            "counterfactuals": _jsonable(
                result.explanation.counterfactuals if result.explanation else ()
            ),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
