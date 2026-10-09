"""Tests for agent_replay.checks.tool_sequence and RunResult.check."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_replay.cassette import Kind
from agent_replay.checks import tool_sequence
from agent_replay.runner import RunResult
from agent_replay.trajectory import Step, StepStatus, Trajectory


def step(
    index: int, name: str, kind: Kind = Kind.TOOL, status: StepStatus = StepStatus.EXACT
) -> Step:
    return Step(index=index, kind=kind, name=name, status=status, fingerprint="fp", request={})


def run_with(names: Sequence[str]) -> RunResult:
    steps = tuple(step(i, n) for i, n in enumerate(names))
    return RunResult(mode="full", task="t", trajectory=Trajectory(steps=steps))


def test_exact_pass() -> None:
    result = tool_sequence(["a", "b"])(run_with(["a", "b"]))
    assert result.passed
    assert result.check == "tool_sequence"
    assert "diff" not in result.details


def test_exact_fail_missing_tool() -> None:
    result = tool_sequence(["a", "b", "c"])(run_with(["a", "c"]))
    assert not result.passed
    assert "missing b" in result.message
    assert "tool call #2 differs" in result.message
    assert "- b" in result.details["diff"]  # type: ignore[operator]


def test_exact_fail_extra_tool() -> None:
    result = tool_sequence(["a"])(run_with(["a", "x"]))
    assert not result.passed
    assert "unexpected x" in result.message
    assert "+ x" in result.details["diff"]  # type: ignore[operator]


def test_reordered_fails_exact() -> None:
    assert not tool_sequence(["a", "b"])(run_with(["b", "a"])).passed


def test_in_order_allows_interleaved_calls() -> None:
    check = tool_sequence(["a", "c"], mode="in_order")
    assert check(run_with(["a", "b", "c", "d"])).passed


def test_in_order_fails_when_swapped() -> None:
    result = tool_sequence(["a", "b"], mode="in_order")(run_with(["b", "a"]))
    assert not result.passed
    assert "missing" in result.message
    assert "unexpected" not in result.message


def test_empty_expected() -> None:
    assert tool_sequence([], mode="in_order")(run_with(["a"])).passed
    assert tool_sequence([], mode="in_order")(run_with([])).passed
    assert tool_sequence([])(run_with([])).passed
    assert not tool_sequence([])(run_with(["a"])).passed


def test_model_steps_ignored_and_unrecorded_tools_counted() -> None:
    steps = (
        step(0, "anthropic.messages.create", Kind.MODEL),
        step(1, "a"),
        step(2, "anthropic.messages.create", Kind.MODEL),
        step(3, "b", status=StepStatus.UNRECORDED),
    )
    run = RunResult(mode="full", task="t", trajectory=Trajectory(steps=steps))
    assert tool_sequence(["a", "b"])(run).passed


def test_bare_string_rejected() -> None:
    with pytest.raises(TypeError):
        tool_sequence("lookup_order")


def test_result_details() -> None:
    result = tool_sequence(["a"], mode="in_order")(run_with(["a", "b"]))
    assert result.details["mode"] == "in_order"
    assert result.details["expected"] == ["a"]
    assert result.details["actual"] == ["a", "b"]
    assert tool_sequence(["a"]).name == "tool_sequence"


def test_run_result_check_and_ok() -> None:
    run = run_with(["a"])
    results = run.check(
        tool_sequence(["a"]), tool_sequence(["b"]), tool_sequence([], mode="in_order")
    )
    assert [r.passed for r in results] == [True, False, True]
    assert run.check() == []
    assert run.ok


names = st.lists(st.sampled_from(["a", "b", "c"]), max_size=12)


@given(data=st.data())
def test_exact_and_subsequence_properties(data: st.DataObject) -> None:
    actual = data.draw(names)
    mask = data.draw(st.lists(st.booleans(), min_size=len(actual), max_size=len(actual)))
    sub = [n for n, keep in zip(actual, mask, strict=True) if keep]
    run = run_with(actual)
    assert tool_sequence(actual)(run).passed
    assert tool_sequence(sub, mode="in_order")(run).passed
