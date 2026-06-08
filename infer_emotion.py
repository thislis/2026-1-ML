"""Convenience wrapper for ``meld-emotion infer``."""

from __future__ import annotations

import sys

from meld_emotion.cli import main

if __name__ == "__main__":
    raise SystemExit(main(("infer", *sys.argv[1:])))
