"""Tests for the checks library: tools, schema, rule, budget and build_check."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import JsonValue

from agent_replay.cassette import ErrorInfo, Kind, Usage
from agent_replay.checks import forbidden_tools, max_tool_calls, required_tools
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
