"""Tests for the checks library: tools, schema, rule, budget and build_check."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, JsonValue

from agent_replay.cassette import ErrorInfo, Kind, Usage
from agent_replay.checks import (
    arg_schema,
    forbidden_tools,
    max_tool_calls,
    output_schema,
    required_tools,
)
from agent_replay.runner import RunResult
from agent_replay.trajectory import Step, StepStatus, Trajectory


def step(
    index: int,
    name: str,
    kind: Kind = Kind.TOOL,
    *,
    request: JsonValue = None,
    parent: str | None = None,
    usage: Usage | None = None,
    cost: float | None = None,
    latency: float | None = None,
) -> Step:
    return Step(
        index=index,
        kind=kind,
        name=name,
        status=StepStatus.EXACT,
        fingerprint="fp",
        request={} if request is None else request,
        usage=usage,
        cost_usd=cost,
        latency_ms=latency,
        parent_span_id=parent,
    )


def make_run(
    steps: Sequence[Step] = (),
    *,
    output: JsonValue = None,
    error: ErrorInfo | None = None,
) -> RunResult:
    return RunResult(
        mode="full",
        task="t",
        output=output,
        error=error,
        trajectory=Trajectory(steps=tuple(steps)),
    )


def tool_run(*names: str) -> RunResult:
    return make_run([step(i, n) for i, n in enumerate(names)])


# --- required_tools / forbidden_tools / max_tool_calls ---------------------------------------


def test_required_tools_pass_and_fail() -> None:
    ok = required_tools(["a", "b"])(tool_run("b", "x", "a"))
    assert ok.passed
    assert ok.check == "required_tools"
    bad = required_tools(["a", "b", "c"])(tool_run("a"))
    assert not bad.passed
    assert "b, c" in bad.message
    assert bad.details["missing"] == ["b", "c"]


def test_required_tools_empty_trajectory_and_empty_list() -> None:
    assert not required_tools(["a"])(make_run()).passed
    assert required_tools([])(make_run()).passed


def test_required_tools_ignores_nested_and_model_steps() -> None:
    run = make_run(
        [
            step(0, "outer"),
            step(1, "inner", parent="s0"),
            step(2, "a", Kind.MODEL),
        ]
    )
    assert not required_tools(["inner"])(run).passed
    assert not required_tools(["a"])(run).passed
    assert required_tools(["outer"])(run).passed


def test_forbidden_tools_names_tool_and_step_index() -> None:
    run = tool_run("a", "rm", "b", "rm")
    result = forbidden_tools(["rm", "drop"])(run)
    assert not result.passed
    assert "rm at step 1" in result.message
    assert "rm at step 3" in result.message
    assert result.details["calls"] == [{"tool": "rm", "step": 1}, {"tool": "rm", "step": 3}]


def test_forbidden_tools_pass_and_nested_ignored() -> None:
    assert forbidden_tools(["rm"])(tool_run("a")).passed
    assert forbidden_tools(["rm"])(make_run()).passed
    nested = make_run([step(0, "outer"), step(1, "rm", parent="s0")])
    assert forbidden_tools(["rm"])(nested).passed


def test_max_tool_calls_total_and_per_tool() -> None:
    run = tool_run("a", "a", "b")
    assert max_tool_calls(3)(run).passed
    over = max_tool_calls(2)(run)
    assert not over.passed
    assert over.details == {"limit": 2, "count": 3, "tool": None}
    assert "3" in over.message
    assert max_tool_calls(2, tool="a")(run).passed
    assert not max_tool_calls(1, tool="a")(run).passed
    assert max_tool_calls(0)(make_run()).passed


def test_max_tool_calls_ignores_nested() -> None:
    run = make_run([step(0, "outer"), step(1, "inner", parent="s0")])
    assert max_tool_calls(1)(run).passed


def test_tool_checks_reject_bare_string_and_negative_limit() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        required_tools("abc")
    with pytest.raises(TypeError, match="not a single string"):
        forbidden_tools("abc")
    with pytest.raises(ValueError, match=">= 0"):
        max_tool_calls(-1)


# --- arg_schema / output_schema --------------------------------------------------------------

ARGS = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id"],
    "additionalProperties": False,
}


class Args(BaseModel):
    id: int
    tags: list[str] = []


def call(index: int, request: JsonValue, tool: str = "lookup") -> Step:
    return step(index, tool, request=request)


def test_arg_schema_json_schema_pass() -> None:
    run = make_run([call(0, {"id": 1}), call(1, {"id": 2, "tags": ["x"]})])
    result = arg_schema("lookup", ARGS)(run)
    assert result.passed
    assert result.check == "arg_schema:lookup"
    assert result.details["calls"] == 2


def test_arg_schema_reports_each_failing_call_with_path() -> None:
    run = make_run(
        [call(0, {"id": 1}), call(1, {"id": "x"}), call(2, {"id": 3, "tags": [1]}), call(3, {})]
    )
    result = arg_schema("lookup", ARGS)(run)
    assert not result.passed
    failures = result.details["failures"]
    assert isinstance(failures, list)
    assert [f["step"] for f in failures] == [1, 2, 3]  # type: ignore[index,call-overload]
    assert "step 1: $.id" in result.message
    assert "step 2: $.tags[0]" in result.message
    assert "step 3: $: " in result.message
    assert "'id' is a required property" in result.message


def test_arg_schema_pydantic_model() -> None:
    run = make_run([call(0, {"id": 1}), call(1, {"id": "nope"})])
    result = arg_schema("lookup", Args)(run)
    assert not result.passed
    assert "step 1: $.id" in result.message
    assert "step 0" not in result.message
    assert arg_schema("lookup", Args)(make_run([call(0, {"id": 1})])).passed


def test_arg_schema_other_tools_and_nested_calls_ignored() -> None:
    run = make_run([call(0, {"bad": 1}, "other"), step(1, "outer"), call(2, {}, "lookup")])
    nested = make_run([step(0, "outer"), step(1, "lookup", request={}, parent="s0")])
    assert arg_schema("lookup", ARGS)(nested).passed
    assert not arg_schema("lookup", ARGS)(run).passed  # step 2 is top level and invalid


def test_arg_schema_not_called() -> None:
    assert arg_schema("lookup", ARGS)(make_run()).passed
    result = arg_schema("lookup", ARGS, must_call=True)(make_run())
    assert not result.passed
    assert "never called" in result.message


def test_bad_json_schema_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        arg_schema("t", {"type": "not-a-type"})
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        output_schema({"properties": 3})
    with pytest.raises(TypeError, match="JSON Schema mapping or a pydantic model"):
        output_schema(int)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JSON Schema mapping or a pydantic model"):
        output_schema("string")  # type: ignore[arg-type]


OUT = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


class Out(BaseModel):
    answer: str


def test_output_schema_json_schema_and_pydantic() -> None:
    good = make_run(output={"answer": "42"})
    bad = make_run(output={"answer": 42})
    assert output_schema(OUT)(good).passed
    assert output_schema(Out)(good).passed
    failed = output_schema(OUT)(bad)
    assert not failed.passed
    assert "$.answer" in failed.message
    assert "$.answer" in str(failed.details["error"])
    assert not output_schema(Out)(bad).passed
    assert not output_schema(OUT)(make_run(output=None)).passed


def test_output_schema_parse_json() -> None:
    check = output_schema(OUT, parse_json=True)
    assert check(make_run(output='{"answer": "x"}')).passed
    assert check(make_run(output={"answer": "x"})).passed  # non-strings are validated as is
    invalid = check(make_run(output="{not json"))
    assert not invalid.passed
    assert "not valid JSON" in invalid.message
    assert not check(make_run(output='{"answer": 1}')).passed


def test_output_schema_string_without_parse_json_is_a_string() -> None:
    assert not output_schema(OUT)(make_run(output='{"answer": "x"}')).passed
    assert output_schema({"type": "string"})(make_run(output="hi")).passed


def test_output_schema_errored_run_fails_clearly() -> None:
    run = make_run(error=ErrorInfo(type="KeyError", message="'x'"))
    result = output_schema(OUT)(run)
    assert not result.passed
    assert "errored" in result.message
    assert "KeyError" in result.message
