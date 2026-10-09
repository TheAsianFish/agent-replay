"""Schema-valid stand-in values for the ``synth`` unrecorded-call policy.

When a replay meets a tool call the cassette never saw, ``synth`` answers it
with the smallest value that fits the tool's annotated return type: required
fields only, empty strings and collections, zeros, the first enum member,
``None`` where allowed. The stub is built from the type's JSON schema and then
validated back into the type, so the agent receives a real ``Order`` (say), not
a dict, and a stub that does not validate is reported instead of served.
"""

from __future__ import annotations

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
            return _stub(options[0], defs, depth + 1)
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
            return [_stub(schema.get("items", {}), defs, depth + 1)] * schema.get("minItems", 0)
        case "string":
            return "x" * int(schema.get("minLength", 0))
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


def _number(schema: dict[str, Any]) -> float:
    if "minimum" in schema:
        return float(schema["minimum"])
    if "exclusiveMinimum" in schema:
        return float(schema["exclusiveMinimum"]) + 1
    if "maximum" in schema and schema["maximum"] < 0:
        return float(schema["maximum"])
    if "exclusiveMaximum" in schema and schema["exclusiveMaximum"] <= 0:
        return float(schema["exclusiveMaximum"]) - 1
    return 0.0
