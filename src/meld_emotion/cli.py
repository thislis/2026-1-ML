"""명령줄 인터페이스.

- ``meld-emotion run --config <path>`` : YAML 설정으로 실험을 실행한다.
- ``meld-emotion infer --mp4 <path> --text <text>`` : 저장된 checkpoint 로 단일 입력을 추론한다.
- ``meld-emotion status`` : 모든 컴포넌트의 구현 상태(REAL/PLACEHOLDER/UNIMPLEMENTED)를
  코드에서 직접 읽어 출력한다(할 일 목록의 단일 진실 공급원).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# builder 를 import 하면 모든 구체 컴포넌트가 로드되어 상태 레지스트리가 채워진다.
from meld_emotion.config.loader import load_config, load_suite
from meld_emotion.core.status import ComponentStatus, iter_status
from meld_emotion.logging_config import configure_logging
from meld_emotion.pipeline import builder
from meld_emotion.pipeline.suite import SuiteRunner
from meld_emotion.reporting.report import ComparisonReporter

logger = logging.getLogger(__name__)


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서도 한글/기호가 깨지지 않도록 UTF-8 로 강제한다."""

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _cmd_run(config_path: str, log_level: str, log_file: str | None) -> int:
    configure_logging(log_level, log_file)
    logger.info("단일 실험 실행 준비: config=%s", config_path)
    config = load_config(config_path)
    runner = builder.build_experiment(config)
    runner.run()
    logger.info("단일 실험 실행 완료: %s", config.name)
    return 0


def _cmd_compare(config_path: str, log_level: str, log_file: str | None) -> int:
    configure_logging(log_level, log_file)
    logger.info("비교 suite 실행 준비: config=%s", config_path)
    suite = load_suite(config_path)
    report = SuiteRunner(suite.name, suite.experiments).run()
    ComparisonReporter(
        metrics=suite.metrics,
        robustness_metric=suite.robustness_metric,
        path=suite.output_path,
    ).save(report)
    logger.info("비교 suite 실행 완료: %s", suite.name)
    return 0


def _cmd_status() -> int:
    records = list(iter_status())
    by_status: dict[ComponentStatus, list[str]] = {s: [] for s in ComponentStatus}
    for record in records:
        label = record.qualname.removeprefix("meld_emotion.")
        suffix = f"  — {record.reason}" if record.reason else ""
        by_status[record.status].append(f"  {label}{suffix}")

    order = [
        ComponentStatus.REAL,
        ComponentStatus.PLACEHOLDER,
        ComponentStatus.UNIMPLEMENTED,
    ]
    for status in order:
        items = by_status[status]
        print(f"[{status.value.upper()}] ({len(items)})")
        for line in sorted(items):
            print(line)
        print()
    total = len(records)
    done = len(by_status[ComponentStatus.REAL])
    print(
        f"요약: 전체 {total}개 중 완전구현 {done}, "
        f"임시 {len(by_status[ComponentStatus.PLACEHOLDER])}, "
        f"미구현 {len(by_status[ComponentStatus.UNIMPLEMENTED])}"
    )
    return 0


def _cmd_infer(
    mp4_path: str,
    text: str,
    checkpoint: str,
    device: str,
    top_k: int,
    json_output: bool,
    include_xai: bool,
    xai_steps: int,
    xai_top_k: int,
    xai_dashboard: str | None,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.inference import (
        dashboard_to_json,
        format_inference_result,
        result_to_json,
        run_inference,
    )

    logger.info("단일 입력 추론 준비: mp4=%s checkpoint=%s", mp4_path, checkpoint)
    result = run_inference(
        mp4_path=mp4_path,
        text=text,
        checkpoint_path=checkpoint,
        device=device,
        top_k=top_k,
        include_xai=include_xai,
        xai_n_steps=xai_steps,
        xai_top_k=xai_top_k,
    )
    if xai_dashboard is not None:
        dashboard_path = Path(xai_dashboard)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(dashboard_to_json(result), encoding="utf-8")
        logger.info("inference XAI dashboard 저장 완료: path=%s", dashboard_path)
    print(result_to_json(result) if json_output else format_inference_result(result))
    logger.info("단일 입력 추론 완료: label=%s probability=%.6f", result.label.value, result.probability)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meld-emotion", description="MELD 멀티모달 감정 인식")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="YAML 설정으로 실험 실행")
    run_parser.add_argument("--config", required=True, help="실험 설정 YAML 경로")
    _add_logging_args(run_parser)

    compare_parser = sub.add_parser("compare", help="여러 실험을 실행하고 비교표 출력")
    compare_parser.add_argument("--config", required=True, help="비교 묶음(suite) YAML 경로")
    _add_logging_args(compare_parser)

    infer_parser = sub.add_parser("infer", help="MP4 파일과 텍스트로 단일 감정 추론")
    infer_parser.add_argument("--mp4", required=True, help="추론할 MP4 파일 경로")
    infer_parser.add_argument("--text", required=True, help="MP4 발화에 대응하는 텍스트")
    infer_parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="dialogue_rnn checkpoint 경로",
    )
    infer_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="추론 장치",
    )
    infer_parser.add_argument("--top-k", type=int, default=7, help="출력할 상위 감정 수")
    infer_parser.add_argument(
        "--xai",
        action="store_true",
        help="감정 예측과 함께 fine-grained XAI 결과를 계산",
    )
    infer_parser.add_argument(
        "--xai-steps",
        type=int,
        default=32,
        help="Integrated Gradients step 수",
    )
    infer_parser.add_argument(
        "--xai-top-k",
        type=int,
        default=10,
        help="XAI 항목별 top-k 개수",
    )
    infer_parser.add_argument(
        "--xai-dashboard",
        default=None,
        help="단일 inference dashboard JSON 저장 경로",
    )
    infer_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="결과를 JSON 으로 출력",
    )
    _add_logging_args(infer_parser)

    sub.add_parser("status", help="컴포넌트 구현 상태 출력")
    return parser


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="진행 상황 로그 레벨",
    )
    parser.add_argument("--log-file", default=None, help="로그를 추가로 저장할 파일 경로")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.config, args.log_level, args.log_file)
    if args.command == "compare":
        return _cmd_compare(args.config, args.log_level, args.log_file)
    if args.command == "infer":
        return _cmd_infer(
            args.mp4,
            args.text,
            args.checkpoint,
            args.device,
            args.top_k,
            args.json_output,
            args.xai,
            args.xai_steps,
            args.xai_top_k,
            args.xai_dashboard,
            args.log_level,
            args.log_file,
        )
    if args.command == "status":
        return _cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
