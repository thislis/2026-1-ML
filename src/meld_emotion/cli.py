"""명령줄 인터페이스.

- ``meld-emotion run --config <path>`` : YAML 설정으로 실험을 실행한다.
- ``meld-emotion status`` : 모든 컴포넌트의 구현 상태(REAL/PLACEHOLDER/UNIMPLEMENTED)를
  코드에서 직접 읽어 출력한다(할 일 목록의 단일 진실 공급원).
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence

# builder 를 import 하면 모든 구체 컴포넌트가 로드되어 상태 레지스트리가 채워진다.
from meld_emotion.config.loader import load_config
from meld_emotion.core.status import ComponentStatus, iter_status
from meld_emotion.pipeline import builder


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서도 한글/기호가 깨지지 않도록 UTF-8 로 강제한다."""

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _cmd_run(config_path: str) -> int:
    config = load_config(config_path)
    runner = builder.build_experiment(config)
    runner.run()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meld-emotion", description="MELD 멀티모달 감정 인식")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="YAML 설정으로 실험 실행")
    run_parser.add_argument("--config", required=True, help="실험 설정 YAML 경로")

    sub.add_parser("status", help="컴포넌트 구현 상태 출력")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.config)
    if args.command == "status":
        return _cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
