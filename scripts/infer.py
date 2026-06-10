"""Single-MP4 inference wrapper with guide-compatible argument names."""

from __future__ import annotations

import argparse

from meld_emotion.cli import main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer emotion from one MP4 plus transcript")
    parser.add_argument("--config", default=None, help="Reserved for extractor/checkpoint presets")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt", help="Checkpoint path")
    parser.add_argument("--input", required=True, help="Input MP4 path")
    parser.add_argument("--text", required=True, help="Transcript text")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--markdown-output", default=None, help="Markdown explanation output path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--explain", action="store_true", help="Enable fine-grained XAI")
    parser.add_argument("--xai-steps", type=int, default=32)
    parser.add_argument("--xai-top-k", type=int, default=10)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-file", default=None)
    return parser


def cli() -> int:
    args = _parser().parse_args()
    argv = [
        "infer",
        "--mp4",
        args.input,
        "--text",
        args.text,
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
        "--top-k",
        str(args.top_k),
        "--json",
        "--log-level",
        args.log_level,
    ]
    if args.explain:
        argv.extend(["--xai", "--xai-steps", str(args.xai_steps), "--xai-top-k", str(args.xai_top_k)])
    if args.markdown_output is not None:
        argv.extend(["--markdown-output", args.markdown_output])
    if args.log_file is not None:
        argv.extend(["--log-file", args.log_file])
    if args.output is None:
        return main(argv)

    from contextlib import redirect_stdout
    from pathlib import Path

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
        return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
