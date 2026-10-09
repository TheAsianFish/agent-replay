"""structlog setup for the CLI: human-readable or JSON lines on stderr.

``REPLAY_LOG_LEVEL`` (default ``WARNING``) and ``REPLAY_LOG_FORMAT``
(``console`` or ``json``, default ``console``) control it.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import NoReturn

import structlog


def _usage_error(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def configure_logging() -> None:
    level_name = os.environ.get("REPLAY_LOG_LEVEL", "WARNING").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        _usage_error(
            f"REPLAY_LOG_LEVEL={level_name!r} is not a log level; use DEBUG, INFO, WARNING or ERROR"
        )
    fmt = os.environ.get("REPLAY_LOG_FORMAT", "console")
    if fmt not in ("console", "json"):
        _usage_error(f"REPLAY_LOG_FORMAT={fmt!r} is not supported; use console or json")
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # Resolve sys.stderr per call: it may be swapped (pytest, CliRunner) after setup.
        logger_factory=lambda *_: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
