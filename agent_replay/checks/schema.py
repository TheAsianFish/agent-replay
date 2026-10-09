"""Schema checks: ``arg_schema`` (a tool's arguments) and ``output_schema`` (the final output)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jsonschema.exceptions
from jsonschema.validators import Draft202012Validator
from pydantic import BaseModel, JsonValue, ValidationError

from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

Schema = Mapping[str, Any] | type[BaseModel]


def _path(parts: Iterable[str | int]) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


def _first_error(schema: Schema, value: object) -> str | None:
    """``"<path>: <message>"`` for the first validation error, or ``None`` when valid."""
    if isinstance(schema, type):
        try:
            schema.model_validate(value)
        except ValidationError as exc:
            err = exc.errors()[0]
            return f"{_path(err['loc'])}: {err['msg']}"
        return None
    error = jsonschema.exceptions.best_match(Draft202012Validator(dict(schema)).iter_errors(value))
    if error is None:
        return None
    return f"{_path(error.absolute_path)}: {error.message}"


def prepare_schema(schema: object) -> Schema:
    """Validate a schema argument at construction so a bad one fails fast."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    if not isinstance(schema, Mapping):
        raise TypeError(
            f"schema must be a JSON Schema mapping or a pydantic model class, got {schema!r}"
        )
    try:
        Draft202012Validator.check_schema(dict(schema))
    except jsonschema.exceptions.SchemaError as exc:
        raise ValueError(
            f"invalid JSON Schema at {_path(exc.absolute_path)}: {exc.message}. "
            "Fix the schema (Draft 2020-12)."
        ) from exc
    return schema


def _kind(schema: Schema) -> str:
    return f"model {schema.__name__}" if isinstance(schema, type) else "JSON schema"


@dataclass(frozen=True)
class ArgSchema:
    """Every call to ``tool`` has arguments that satisfy ``schema``.

    ``schema`` is a JSON Schema (Draft 2020-12) mapping or a pydantic model. With
    ``must_call`` a run that never calls the tool fails; otherwise it passes.
    """

    tool: str
    schema: Schema = field(compare=False)
    must_call: bool = False

    @property
    def name(self) -> str:
        return f"arg_schema:{self.tool}"

    def __call__(self, run: RunResult, /) -> CheckResult:
        calls = [s for s in run.trajectory.tool_steps() if s.name == self.tool]
        failures: list[dict[str, JsonValue]] = []
        for s in calls:
            err = _first_error(self.schema, s.request)
            if err is not None:
                failures.append({"step": s.index, "error": err})
        details: dict[str, JsonValue] = {
            "tool": self.tool,
            "calls": len(calls),
            "failures": list(failures),
        }
        if failures:
            shown = "; ".join(f"step {f['step']}: {f['error']}" for f in failures)
            passed = False
            message = f"{self.tool} arguments invalid in {len(failures)} call(s): {shown}"
        elif not calls and self.must_call:
            passed, message = False, f"{self.tool} was never called (must_call is set)"
        elif not calls:
            passed, message = True, f"{self.tool} was not called; nothing to validate"
        else:
            passed = True
            message = f"{len(calls)} call(s) to {self.tool} match the {_kind(self.schema)}"
        return CheckResult(check=self.name, passed=passed, message=message, details=details)


def arg_schema(
    tool: str,
    schema: Mapping[str, Any] | type[BaseModel],
    *,
    must_call: bool = False,
) -> ArgSchema:
    """Build an ``arg_schema`` check. See :class:`ArgSchema`.

    A malformed JSON Schema raises ``ValueError`` here, not at run time.
    """
    return ArgSchema(tool=tool, schema=prepare_schema(schema), must_call=must_call)


@dataclass(frozen=True)
class OutputSchema:
    """The run's final output validates against ``schema``.

    With ``parse_json`` a string output is parsed with ``json.loads`` first.
    """

    schema: Schema = field(compare=False)
    parse_json: bool = False

    @property
    def name(self) -> str:
        return "output_schema"

    def __call__(self, run: RunResult, /) -> CheckResult:
        if run.error is not None:
            return CheckResult(
                check=self.name,
                passed=False,
                message=(
                    f"the run errored ({run.error.type}: {run.error.message}) "
                    "and produced no output to validate"
                ),
            )
        value: object = run.output
        if self.parse_json and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                return CheckResult(
                    check=self.name,
                    passed=False,
                    message=(
                        f"output is not valid JSON: {exc.msg} at line {exc.lineno} "
                        f"column {exc.colno}"
                    ),
                )
        err = _first_error(self.schema, value)
        if err is not None:
            return CheckResult(
                check=self.name,
                passed=False,
                message=f"output does not match the {_kind(self.schema)}: {err}",
                details={"error": err},
            )
        return CheckResult(
            check=self.name, passed=True, message=f"output matches the {_kind(self.schema)}"
        )


def output_schema(
    schema: Mapping[str, Any] | type[BaseModel], *, parse_json: bool = False
) -> OutputSchema:
    """Build an ``output_schema`` check. See :class:`OutputSchema`."""
    return OutputSchema(schema=prepare_schema(schema), parse_json=parse_json)
