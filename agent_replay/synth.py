"""Schema-valid stand-in values for the ``synth`` unrecorded-call policy.

When a replay meets a tool call the cassette never saw, ``synth`` answers it
with the smallest value that fits the tool's annotated return type: required
fields only, empty strings and collections, zeros, the first enum member,
``None`` where allowed. The stub is built from the type's JSON schema and then
validated back into the type, so the agent receives a real ``Order`` (say), not
a dict, and a stub that does not validate is reported instead of served.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import JsonValue

from agent_replay.errors import ReplayError

_MAX_DEPTH = 32


class SynthError(ReplayError):
    """No stub can be built for this schema."""


def stub_from_schema(schema: dict[str, Any]) -> JsonValue:
    """The minimal JSON value satisfying ``schema`` (the subset pydantic emits)."""
    return _stub(schema, schema.get("$defs", {}), 0)


def _stub(  # noqa: PLR0911, PLR0912 - one branch per JSON schema construct
    schema: dict[str, Any], defs: dict[str, Any], depth: int
) -> JsonValue:
    if depth > _MAX_DEPTH:
        raise SynthError("the return type is recursive without an empty base case")
    if "$ref" in schema:
        name = str(schema["$ref"]).rsplit("/", 1)[-1]
        if name not in defs:
            raise SynthError(f"unresolvable $ref {schema['$ref']!r}")
        return _stub(defs[name], defs, depth + 1)
    if "default" in schema:
        default: JsonValue = schema["default"]
        return default
    if "const" in schema:
        const: JsonValue = schema["const"]
        return const
    if "enum" in schema:
        first: JsonValue = schema["enum"][0]
        return first
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options: list[dict[str, Any]] = schema[key]
            if any(o.get("type") == "null" for o in options):
                return None
            for option in options:  # the first branch a stub can be built for
                try:
                    return _stub(option, defs, depth + 1)
                except SynthError:
                    continue
            raise SynthError(f"cannot build a stub for any branch of {key}")
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for part in schema["allOf"]:
            merged.update(part)
        return _stub(merged, defs, depth + 1)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    match kind:
        case "object" | None if kind == "object" or "properties" in schema:
            properties: dict[str, Any] = schema.get("properties", {})
            return {
                name: _stub(properties[name], defs, depth + 1)
                for name in schema.get("required", [])
                if name in properties
            }
        case "array":
            items = schema.get("prefixItems")
            if items:
                return [_stub(item, defs, depth + 1) for item in items]
            count = int(schema.get("minItems", 0))
            if count > 1 and schema.get("uniqueItems"):
                raise SynthError(f"cannot build {count} distinct items for a set")
            return [_stub(schema.get("items", {}), defs, depth + 1)] * count
        case "string":
            return _string(schema)
        case "integer":
            return int(_number(schema))
        case "number":
            return _number(schema)
        case "boolean":
            return False
        case "null":
            return None
        case None:
            return None  # an unconstrained schema (``Any``) accepts anything
    raise SynthError(f"cannot build a stub for schema type {kind!r}")


# Smallest valid values for the string formats pydantic emits.
_FORMATS: dict[str, str] = {
    "date-time": "1970-01-01T00:00:00Z",
    "date": "1970-01-01",
    "time": "00:00:00",
    "duration": "PT0S",
    "uuid": "00000000-0000-0000-0000-000000000000",
    "uuid1": "00000000-0000-1000-8000-000000000000",
    "uuid3": "00000000-0000-3000-8000-000000000000",
    "uuid4": "00000000-0000-4000-8000-000000000000",
    "uuid5": "00000000-0000-5000-8000-000000000000",
    "email": "x@example.com",
    "uri": "https://example.com/",
    "ipv4": "0.0.0.0",  # noqa: S104 - a placeholder value, not a bind address
    "ipv6": "::",
}


def _string(schema: dict[str, Any]) -> str:
    if "pattern" in schema:
        raise SynthError(f"cannot build a string matching the pattern {schema['pattern']!r}")
    fmt = schema.get("format")
    if fmt is not None:
        if fmt not in _FORMATS:
            raise SynthError(f"cannot build a string in the {fmt!r} format")
        return _FORMATS[fmt]
    return "x" * int(schema.get("minLength", 0))


def _number(schema: dict[str, Any]) -> float:
    """The value closest to zero that satisfies every bound in ``schema``."""
    inf = float("inf")
    low, high = float(schema.get("minimum", -inf)), float(schema.get("maximum", inf))
    above = float(schema.get("exclusiveMinimum", -inf))
    below = float(schema.get("exclusiveMaximum", inf))
    integer = schema.get("type") == "integer"
    floor, ceiling = max(low, above), min(high, below)
    candidates = [0.0, low, high, above + 1, below - 1, (floor + ceiling) / 2]
    for value in sorted((c for c in candidates if math.isfinite(c)), key=abs):
        fits = low <= value <= high and above < value < below
        if fits and (not integer or value.is_integer()):
            return value
    raise SynthError(f"no {'integer' if integer else 'number'} fits the bounds in {schema}")
