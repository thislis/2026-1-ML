"""실험 결과 리포터.

:class:`JsonReporter` 와 :class:`ConsoleReporter` 는 완전 구현이다.
:class:`DashboardExporter` 는 제안서의 case-study 대시보드용 JSON 데이터 구조를 내보내지만
실제 시각화 렌더링은 미구현이라 임시(placeholder)로 표시한다.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from meld_emotion.core.results import (
    ComparisonReport,
    EvaluationReport,
    ExperimentOutcome,
    ExperimentResult,
)
from meld_emotion.core.status import note_placeholder_use, placeholder, real

logger = logging.getLogger(__name__)


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
        logger.info("JSON 리포트 저장 완료: path=%s", self._path)


@real
class ConsoleReporter:
    """실험 요약을 콘솔에 출력."""

    def save(self, result: ExperimentResult) -> None:
        print(self.format(result))
        logger.info("콘솔 리포트 출력 완료: experiment=%s", result.name)

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
            if explanation.dialogue_xai:
                lines.append("[Fine-grained dialogue XAI]")
                for item in explanation.dialogue_xai[:5]:
                    top_modality = max(
                        item.modality,
                        key=lambda modality: modality.attribution_share,
                    )
                    top_utt = item.utterances[0] if item.utterances else None
                    text = item.top_text_units[0].label if item.top_text_units else "-"
                    audio = item.top_audio_units[0].label if item.top_audio_units else "-"
                    video = item.top_video_units[0].label if item.top_video_units else "-"
                    source = (
                        f"utt={top_utt.utterance_id} share={top_utt.share:.2f}"
                        if top_utt is not None
                        else "utt=-"
                    )
                    lines.append(
                        "  "
                        f"{item.uid} pred={item.pred_class.value} "
                        f"mod={top_modality.modality.value}:{top_modality.attribution_share:.2f} "
                        f"{source} text={text} audio={audio} video={video}"
                    )
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
            "finegrained_xai": _dashboard_xai(result),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("대시보드 데이터 저장 완료: path=%s", self._path)


def _dashboard_xai(result: ExperimentResult) -> dict[str, Any]:
    explanation = result.explanation
    if explanation is None or not explanation.dialogue_xai:
        return {"targets": []}
    targets = []
    for item in explanation.dialogue_xai:
        targets.append(
            {
                "uid": item.uid,
                "speaker": item.speaker,
                "pred_class": item.pred_class.value,
                "pred_proba": item.pred_proba,
                "target_class": item.target_class.value,
                "target_logit": item.target_logit,
                "modality_panel": _jsonable(item.modality),
                "dialogue_panel": _jsonable(item.utterances),
                "block_panel": _jsonable(item.classifier_blocks),
                "text_panel": _jsonable(item.top_text_units),
                "audio_panel": _jsonable(item.top_audio_units),
                "video_panel": _jsonable(item.top_video_units),
                "dimension_panel": {
                    "text": _jsonable(item.text_dimension_attribution),
                    "audio": _jsonable(item.audio_dimension_attribution),
                    "video": _jsonable(item.video_dimension_attribution),
                },
            }
        )
    return {"targets": targets}


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """헤더/행을 열 폭에 맞춰 정렬한 표 라인들로 만든다."""

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: Sequence[str]) -> str:
        return "  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    return [_fmt(headers), *[_fmt(row) for row in rows]]


@real
class ComparisonReporter:
    """여러 실험의 :class:`ComparisonReport` 를 콘솔 표 + JSON 으로 내보낸다.

    :class:`Reporter` Protocol(단일 ``ExperimentResult``)과는 입력 타입이 다른, 비교 전용
    리포터이다(파이프라인 단계에 주입되지 않고 :class:`SuiteRunner` 결과를 받는다).
    """

    def __init__(
        self,
        metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1"),
        robustness_metric: str = "macro_f1",
        path: str = "outputs/comparison.json",
    ) -> None:
        self._metrics = metrics
        self._robustness_metric = robustness_metric
        self._path = Path(path)

    def save(self, report: ComparisonReport) -> None:
        print(self.format(report))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("비교 리포트 저장 완료: path=%s", self._path)

    def format(self, report: ComparisonReport) -> str:
        ok = list(report.successful())
        lines: list[str] = [f"=== Suite: {report.name} ({len(ok)}/{len(report.outcomes)} ok) ==="]

        lines.append("[Metrics]")
        lines.extend(self._metrics_table(ok))

        robustness = self._robustness_table(ok)
        if robustness:
            lines.append(f"[Robustness: {self._robustness_metric} by scenario]")
            lines.extend(robustness)

        failed = report.failed()
        if failed:
            lines.append("[Failed]")
            lines.extend(f"  {o.name}: {o.error}" for o in failed)
        return "\n".join(lines)

    def _metrics_table(self, outcomes: Sequence[ExperimentOutcome]) -> list[str]:
        headers = ["experiment", *self._metrics]
        rows: list[list[str]] = []
        for outcome in outcomes:
            result = outcome.result
            if result is None:
                continue
            cells = [outcome.name]
            for metric_name in self._metrics:
                found = result.evaluation.metric(metric_name)
                cells.append(f"{found.value:.4f}" if found is not None else "-")
            rows.append(cells)
        return _render_table(headers, rows)

    def _robustness_table(self, outcomes: Sequence[ExperimentOutcome]) -> list[str]:
        scenarios: list[str] = []
        for outcome in outcomes:
            result = outcome.result
            if result is None or result.robustness is None:
                continue
            for report in result.robustness.reports:
                if report.scenario not in scenarios:
                    scenarios.append(report.scenario)
        if not scenarios:
            return []

        headers = ["experiment", *scenarios]
        rows: list[list[str]] = []
        for outcome in outcomes:
            result = outcome.result
            if result is None or result.robustness is None:
                continue
            by_scenario = {r.scenario: r for r in result.robustness.reports}
            cells = [outcome.name]
            for scenario in scenarios:
                scenario_report = by_scenario.get(scenario)
                found = (
                    scenario_report.metric(self._robustness_metric)
                    if scenario_report is not None
                    else None
                )
                cells.append(f"{found.value:.4f}" if found is not None else "-")
            rows.append(cells)
        return _render_table(headers, rows)
