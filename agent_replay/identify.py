"""Identifying arguments: the arguments of a tool call that name what it reads or writes.

When no recording matches a served call exactly, the session falls back to the
oldest unused recording of the same tool and flags the step ``approximate``.
That is right when the difference is incidental (a reworded summary, a
reordered list) and wrong when it names another entity: a read of order
``#W2`` served the recording of ``#W1`` hands the agent another order's data,
and every later step looks normal. No distance on the arguments separates the
two, since both are a one-field change. The suite says which is which, per
tool::

    [defaults.identify]
    get_order_details = ["order_id"]

A call is then only ever served a recording with the same values at those
paths. A call whose identity no unused recording has is ``unrecorded``, and the
policy decides (``strict`` fails the run, ``fallback`` runs the real tool,
``synth`` stubs it). Paths use the ``ignore_fields`` syntax without wildcards:
an identity is a definite value (``order_id``, ``customer.id``,
``items[0].sku``). Values compare as canonical JSON with no normalisation, so
``"sfo"`` and ``"SFO"`` are different entities. A path absent from a request is
part of its identity too (absent is not ``null``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from agent_replay._json import canonical_json
from agent_replay._paths import WILDCARD, Path, format_path, parse_path
from agent_replay.cassette import Cassette, Kind
from agent_replay.errors import ConfigError

type Identify = Mapping[str, tuple[Path, ...]]
"""Per tool name, its parsed identifying paths."""


def label(path: Path) -> str:
    """``order_id``, ``customer.id``: a path as written in a suite."""
    text = format_path(path)
    return text[2:] if text.startswith("$.") else text[1:]


def parse_identify(spec: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[Path, ...]]:
    """Parse ``{tool: [path, ...]}``; raises :class:`ConfigError` naming the bad entry.

    A tool with an empty list has no identifying arguments (a scenario uses that
    to switch off a default).
    """
    parsed: dict[str, tuple[Path, ...]] = {}
    for tool, texts in (spec or {}).items():
        if isinstance(texts, str):
            raise ConfigError(
                f"identify: tool {tool!r}: give a list of argument paths, such as "
                f'["{texts}"], not a string'
            )
        paths: list[Path] = []
        for text in texts:
            try:
                path = parse_path(text, setting="identify")
            except ConfigError as exc:
                raise ConfigError(f"tool {tool!r}: {exc}") from exc
            if WILDCARD in path:
                raise ConfigError(
                    f"identify: tool {tool!r}: {text!r} has a wildcard, but an identifying "
                    "argument names one value; write paths like 'order_id' or 'customer.id'"
                )
            if path not in paths:
                paths.append(path)
        parsed[tool] = tuple(paths)
    return parsed


def _lookup(value: JsonValue, path: Path) -> tuple[bool, JsonValue]:
    for segment in path:
        in_dict = isinstance(value, dict) and isinstance(segment, str) and segment in value
        in_list = isinstance(value, list) and isinstance(segment, int) and segment < len(value)
        if not (in_dict or in_list):
            return False, None
        value = value[segment]  # type: ignore[index]
    return True, value


def identity(request: JsonValue, paths: Sequence[Path]) -> str:
    """The identity of a call with ``request``: its values at ``paths``, as canonical JSON.

    Each path contributes ``[value]`` when present and ``[]`` when absent, so a
    missing argument and an explicit ``null`` differ.
    """
    values: list[JsonValue] = []
    for path in paths:
        found, value = _lookup(request, path)
        values.append([value] if found else [])
    return canonical_json(values)


def values(request: JsonValue, paths: Sequence[Path]) -> dict[str, JsonValue]:
    """``{"order_id": "#W1"}``: the identifying values present in ``request``, by path."""
    out: dict[str, JsonValue] = {}
    for path in paths:
        found, value = _lookup(request, path)
        if found:
            out[label(path)] = value
    return out


def describe(request: JsonValue, paths: Sequence[Path], limit: int = 120) -> str:
    """``order_id="#W1"`` for messages; absent paths read ``(absent)``."""
    parts = []
    for path in paths:
        found, value = _lookup(request, path)
        parts.append(f"{label(path)}={canonical_json(value) if found else '(absent)'}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def check_recorded(cassette: Cassette, identify: Identify) -> None:
    """Fail on an identifying path no recording of its tool has: almost always a typo.

    A tool the cassette never recorded is not checked (a suite's defaults cover
    every scenario); a path that only some recordings have is allowed (an
    optional argument).
    """
    for tool, paths in identify.items():
        requests = [e.request for e in cassette.envelopes if e.kind is Kind.TOOL and e.name == tool]
        if not requests:
            continue
        for path in paths:
            if not any(_lookup(r, path)[0] for r in requests):
                arguments = sorted({k for r in requests if isinstance(r, dict) for k in r})
                raise ConfigError(
                    f"identify: tool {tool!r}: no recorded call has {label(path)!r} "
                    f"(its recorded arguments: {', '.join(arguments) or 'none'}); fix the path "
                    "in the suite's identify table, or name an argument every call sets"
                )
