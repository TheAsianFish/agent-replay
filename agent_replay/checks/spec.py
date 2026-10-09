"""Build a check from one config table (``[[checks]]`` in a project's config)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent_replay._loading import load_object
from agent_replay.checks.base import Check
from agent_replay.checks.budget import budget
from agent_replay.checks.rule import rule
from agent_replay.checks.schema import arg_schema, output_schema
from agent_replay.checks.tool_sequence import tool_sequence
from agent_replay.checks.tools import forbidden_tools, max_tool_calls, required_tools
from agent_replay.errors import AgentLoadError, ConfigError


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ToolSequenceSpec(_Spec):
    type: Literal["tool_sequence"]
    expected: list[str] | None = None
    mode: Literal["exact", "in_order"] = "exact"


class _RequiredToolsSpec(_Spec):
    type: Literal["required_tools"]
    tools: list[str]


class _ForbiddenToolsSpec(_Spec):
    type: Literal["forbidden_tools"]
    tools: list[str]


class _MaxToolCallsSpec(_Spec):
    type: Literal["max_tool_calls"]
    limit: int = Field(ge=0)
    tool: str | None = None


class _SchemaSources(_Spec):
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_file: str | None = None
    model: str | None = None


class _ArgSchemaSpec(_SchemaSources):
    type: Literal["arg_schema"]
    tool: str
    must_call: bool = False


class _OutputSchemaSpec(_SchemaSources):
    type: Literal["output_schema"]
    parse_json: bool = False


class _RuleSpec(_Spec):
    type: Literal["rule"]
    ref: str
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class _BudgetSpec(_Spec):
    type: Literal["budget"]
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_model_calls: int | None = None
    max_p95_latency_ms: float | None = None


_Anything = Annotated[
    _ToolSequenceSpec
    | _RequiredToolsSpec
    | _ForbiddenToolsSpec
    | _MaxToolCallsSpec
    | _ArgSchemaSpec
    | _OutputSchemaSpec
    | _RuleSpec
    | _BudgetSpec,
    Field(discriminator="type"),
]
_ADAPTER: TypeAdapter[_Anything] = TypeAdapter(_Anything)

CHECK_TYPES = (
    "tool_sequence",
    "required_tools",
    "forbidden_tools",
    "max_tool_calls",
    "arg_schema",
    "output_schema",
    "rule",
    "budget",
)


def _describe(spec: Mapping[str, Any]) -> str:
    return f"check of type {spec.get('type')!r}"


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        if err["type"] == "union_tag_invalid":
            loc = "type"
            msg = f"unknown check type; use one of {', '.join(CHECK_TYPES)}"
        elif err["type"] == "union_tag_not_found":
            loc = "type"
            msg = f"missing; use one of {', '.join(CHECK_TYPES)}"
        else:
            msg = err["msg"]
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def _load(ref: str, what: str) -> object:
    try:
        return load_object(ref)
    except AgentLoadError as exc:
        raise ConfigError(f"{what} {ref!r}: {exc}") from exc


def _schema_source(spec: _SchemaSources, base_dir: Path) -> Mapping[str, Any] | type[BaseModel]:
    given = [
        k
        for k, v in (
            ("schema", spec.schema_),
            ("schema_file", spec.schema_file),
            ("model", spec.model),
        )
        if v is not None
    ]
    if len(given) != 1:
        found = f"got {', '.join(given)}" if given else "got none"
        raise ConfigError(
            f"give exactly one of schema (inline table), schema_file (JSON file) or model "
            f"('module:Class'); {found}"
        )
    if spec.schema_ is not None:
        return spec.schema_
    if spec.schema_file is not None:
        path = base_dir / spec.schema_file
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(
                f"cannot read schema_file {str(path)!r}: {exc.strerror or exc}; paths are "
                f"relative to {str(base_dir)!r}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"schema_file {str(path)!r} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"schema_file {str(path)!r} must contain a JSON object")
        return loaded
    assert spec.model is not None
    obj = _load(spec.model, "cannot load model")
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ConfigError(f"model {spec.model!r} is not a pydantic BaseModel subclass")
    return obj


def build_check(
    spec: Mapping[str, Any],
    *,
    base_dir: Path,
    expected_tools: Sequence[str] | None = None,
) -> Check:
    """Turn one config table into a check, keyed on ``spec["type"]``.

    ``base_dir`` anchors ``schema_file`` paths; ``expected_tools`` (the cassette's
    recorded tools) is the default ``expected`` of ``tool_sequence``. Raises
    :class:`ConfigError` naming the offending field.
    """
    try:
        parsed = _ADAPTER.validate_python(dict(spec))
    except ValidationError as exc:
        raise ConfigError(
            f"invalid {_describe(spec)}: {_validation_message(exc)}. Fix the config table."
        ) from exc
    try:
        return _build(parsed, base_dir, expected_tools)
    except ConfigError as exc:
        raise ConfigError(f"invalid {_describe(spec)}: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid {_describe(spec)}: {exc}") from exc


def _build(parsed: _Anything, base_dir: Path, expected_tools: Sequence[str] | None) -> Check:  # noqa: PLR0911
    if isinstance(parsed, _ToolSequenceSpec):
        expected = parsed.expected if parsed.expected is not None else expected_tools
        if expected is None:
            raise ConfigError(
                "no 'expected' list and no recorded tools to default to; set expected = [...] "
                "or run against a cassette"
            )
        return tool_sequence(expected, mode=parsed.mode)
    if isinstance(parsed, _RequiredToolsSpec):
        return required_tools(parsed.tools)
    if isinstance(parsed, _ForbiddenToolsSpec):
        return forbidden_tools(parsed.tools)
    if isinstance(parsed, _MaxToolCallsSpec):
        return max_tool_calls(parsed.limit, tool=parsed.tool)
    if isinstance(parsed, _ArgSchemaSpec):
        return arg_schema(parsed.tool, _schema_source(parsed, base_dir), must_call=parsed.must_call)
    if isinstance(parsed, _OutputSchemaSpec):
        return output_schema(_schema_source(parsed, base_dir), parse_json=parsed.parse_json)
    if isinstance(parsed, _RuleSpec):
        fn = _load(parsed.ref, "cannot load rule")
        if not callable(fn):
            raise ConfigError(f"ref {parsed.ref!r} is not callable")
        return rule(fn, name=parsed.name, **parsed.args)
    return budget(
        max_cost_usd=parsed.max_cost_usd,
        max_tokens=parsed.max_tokens,
        max_model_calls=parsed.max_model_calls,
        max_p95_latency_ms=parsed.max_p95_latency_ms,
    )
