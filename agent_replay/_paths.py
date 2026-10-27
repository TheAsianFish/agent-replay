"""Paths into JSON values (``items[*].price``), shared by ``ignore_fields`` and ``identify``."""

from __future__ import annotations

import re

from agent_replay.errors import ConfigError

type Segment = str | int
type Path = tuple[Segment, ...]

WILDCARD = "*"
_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[(\*|\d+)\]")


def parse_path(text: str, *, setting: str = "ignore_fields") -> Path:
    """``items[*].price`` -> ``("items", "*", "price")``; raises :class:`ConfigError`.

    Keys are separated by ``.``; ``[n]`` is a list index and ``[*]`` (or a bare
    ``*`` key) matches any one key or index. ``setting`` names the suite setting
    the path came from, in the error message.
    """
    segments: list[Segment] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        # A key is dotted everywhere but at the start: "a.b", never ".a" or "[0]b".
        misplaced_key = (
            match is not None
            and match.group(1) is not None
            and text.startswith(".", pos) == (pos == 0)
        )
        if match is None or misplaced_key:
            raise ConfigError(
                f"{setting}: cannot parse {text!r} at character {pos + 1}; write paths "
                "like 'created_at', 'items[*].updated_at' or 'items[0].id'"
            )
        key, index = match.groups()
        if key is not None:
            segments.append(key)
        else:
            segments.append(WILDCARD if index == WILDCARD else int(index))
        pos = match.end()
    if not segments:
        raise ConfigError(f"{setting}: an empty path names nothing; remove it")
    return tuple(segments)


def format_path(path: Path) -> str:
    """``("items", 0, "status")`` -> ``$.items[0].status``; the root is ``$``."""
    out = "$"
    for segment in path:
        out += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return out
