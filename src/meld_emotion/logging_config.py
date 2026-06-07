"""CLI logging setup for the MELD emotion pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

_PACKAGE_LOGGER = "meld_emotion"
_MANAGED_HANDLER = "_meld_emotion_managed"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level_name: str = "INFO", log_file: str | None = None) -> None:
    """Configure package-scoped logging for command-line runs.

    The package installs a ``NullHandler`` by default, so importing it stays quiet. The CLI calls
    this function to attach human-readable console/file handlers only for explicit executions.
    """

    level = _parse_level(level_name)
    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.setLevel(level)
    logger.propagate = False

    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _MANAGED_HANDLER, True)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MANAGED_HANDLER, True)
        logger.addHandler(file_handler)


def _parse_level(level_name: str) -> int:
    normalized = level_name.upper()
    level = getattr(logging, normalized, None)
    if not isinstance(level, int):
        choices = ", ".join(("DEBUG", "INFO", "WARNING", "ERROR"))
        raise ValueError(f"알 수 없는 로그 레벨입니다: {level_name!r} (가능: {choices})")
    return level
