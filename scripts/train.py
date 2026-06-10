"""Training wrapper for the multimodal emotion pipeline."""

from __future__ import annotations

import argparse

from meld_emotion.cli import main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate from an experiment config")
    parser.add_argument("--config", default="configs/default.yaml", help="Experiment YAML path")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-file", default=None)
    return parser


def cli() -> int:
    args = _parser().parse_args()
    argv = ["run", "--config", args.config, "--log-level", args.log_level]
    if args.log_file is not None:
        argv.extend(["--log-file", args.log_file])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())

