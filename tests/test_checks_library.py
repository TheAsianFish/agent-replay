"""Tests for the checks library: tools, schema, rule, budget and build_check."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue

from agent_replay.baseline import ScenarioBaseline
from agent_replay.cassette import ErrorInfo, Kind, Usage
from agent_replay.checks import (
    ArgSchema,
    Budget,
    CheckResult,
    ForbiddenTools,
    MaxToolCalls,
    OutputSchema,
    RequiredTools,
    Rule,
    ToolSequence,
    arg_schema,
    budget,
    build_check,
    forbidden_tools,
    max_tool_calls,
    output_schema,
    required_tools,
    rule,
)
from agent_replay.checks.budget import percentile_nearest_rank
from agent_replay.errors import ConfigError, ReplayError
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


# --- rule -------------------------------------------------------------------------------------


def has_output(run: RunResult) -> bool:
    return run.output is not None


def with_message(run: RunResult, *, needle: str = "x") -> tuple[bool, str]:
    ok = needle in str(run.output)
    return ok, f"{'found' if ok else 'missing'} {needle}"


def full_result(run: RunResult) -> CheckResult:
    return CheckResult(check="mine", passed=True, message="custom", details={"k": 1})


def boom(run: RunResult) -> bool:
    raise RuntimeError("kaput")


def test_rule_bool() -> None:
    check = rule(has_output)
    assert check.name == "rule:has_output"
    assert check(make_run(output="x")).passed
    failed = check(make_run())
    assert not failed.passed
    assert failed.check == "rule:has_output"
    assert failed.message


def test_rule_tuple_with_kwargs_and_name() -> None:
    check = rule(with_message, name="needle", needle="abc")
    assert check.name == "needle"
    ok = check(make_run(output="xabcx"))
    assert ok.passed
    assert ok.message == "found abc"
    bad = check(make_run(output="zzz"))
    assert not bad.passed
    assert bad.message == "missing abc"
    assert rule(with_message)(make_run(output="x")).passed  # default kwarg


def test_rule_checkresult_passes_through() -> None:
    result = rule(full_result)(make_run())
    assert result.check == "mine"
    assert result.details == {"k": 1}


def test_rule_raising_fails_instead_of_crashing() -> None:
    result = rule(boom)(make_run())
    assert not result.passed
    assert "RuntimeError" in result.message
    assert "kaput" in result.message
    assert result.check == "rule:boom"


def test_rule_bad_return_shape_fails_with_hint() -> None:
    result = rule(lambda run: "yes")(make_run())
    assert not result.passed
    assert "bool" in result.message
    assert rule(lambda run: None, name="n").name == "n"


def test_rule_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        rule("nope")  # type: ignore[arg-type]


# --- budget -----------------------------------------------------------------------------------


def test_percentile_nearest_rank() -> None:
    assert percentile_nearest_rank([], 95) is None
    assert percentile_nearest_rank([7.0], 95) == 7.0
    assert percentile_nearest_rank([9.0, 1.0], 95) == 9.0
    values = [float(v) for v in range(20, 0, -1)]  # 1..20 shuffled order
    assert percentile_nearest_rank(values, 95) == 19.0  # ceil(0.95 * 20) = 19th
    assert percentile_nearest_rank(values, 100) == 20.0
    assert percentile_nearest_rank(values, 5) == 1.0


def model_step(index: int, **kw: float | Usage | None) -> Step:
    return step(index, "m", Kind.MODEL, **kw)  # type: ignore[arg-type]


def test_budget_requires_a_limit_and_rejects_negative() -> None:
    with pytest.raises(ValueError, match="at least one limit"):
        budget()
    with pytest.raises(ValueError, match=">= 0"):
        budget(max_tokens=-1)


def test_budget_within_limits_and_details() -> None:
    usage = Usage(input_tokens=100, output_tokens=50)
    run = make_run(
        [
            model_step(0, usage=usage, cost=0.01, latency=100.0),
            step(1, "tool"),
            model_step(2, usage=usage, cost=0.02, latency=300.0),
        ]
    )
    result = budget(max_cost_usd=0.05, max_tokens=400, max_model_calls=2, max_p95_latency_ms=300)(
        run
    )
    assert result.passed, result.message
    assert result.check == "budget"
    assert result.details == {
        "cost_usd": 0.03,
        "max_cost_usd": 0.05,
        "tokens": 300,
        "max_tokens": 400,
        "model_calls": 2,
        "max_model_calls": 2,
        "p95_latency_ms": 300.0,
        "max_p95_latency_ms": 300,
    }


def test_budget_message_names_each_exceeded_limit() -> None:
    usage = Usage(input_tokens=1000)
    run = make_run([model_step(i, usage=usage, cost=0.5, latency=50.0 * (i + 1)) for i in range(3)])
    result = budget(max_cost_usd=1.0, max_tokens=2000, max_model_calls=2, max_p95_latency_ms=200)(
        run
    )
    assert not result.passed
    assert "cost_usd 1.5 > max_cost_usd 1" in result.message
    assert "tokens 3000 > max_tokens 2000" in result.message
    assert "model_calls 3 > max_model_calls 2" in result.message
    assert "p95_latency_ms 150" not in result.message  # 150 <= 200: not exceeded
    assert "p95_latency_ms" not in result.message


def test_budget_p95_latency_exceeded_and_skips_missing_latency() -> None:
    run = make_run([model_step(0, latency=10.0), model_step(1), model_step(2, latency=900.0)])
    result = budget(max_p95_latency_ms=500)(run)
    assert not result.passed
    assert "p95_latency_ms 900 > max_p95_latency_ms 500" in result.message
    assert result.details["model_calls"] == 3


def test_budget_p95_on_one_two_and_twenty_samples() -> None:
    one = make_run([model_step(0, latency=42.0)])
    assert budget(max_p95_latency_ms=100)(one).details["p95_latency_ms"] == 42.0
    two = make_run([model_step(0, latency=10.0), model_step(1, latency=20.0)])
    assert budget(max_p95_latency_ms=100)(two).details["p95_latency_ms"] == 20.0
    twenty = make_run([model_step(i, latency=float(i + 1)) for i in range(20)])
    assert budget(max_p95_latency_ms=100)(twenty).details["p95_latency_ms"] == 19.0


def test_budget_empty_run_and_no_latency_samples() -> None:
    result = budget(max_cost_usd=0.0, max_p95_latency_ms=1)(make_run())
    assert result.passed
    assert result.details["p95_latency_ms"] is None
    assert result.details["model_calls"] == 0


# --- budget: relative limits against a baseline ------------------------------------------------


def base(
    cost: float = 0.1, tokens: int = 1000, calls: int = 2, p95: float | None = 200.0
) -> ScenarioBaseline:
    return ScenarioBaseline(
        cost_usd=cost, tokens=tokens, model_calls=calls, p95_latency_ms=p95, runs=1
    )


def priced_run(cost: float, tokens: int = 1000, latency: float | None = 200.0) -> RunResult:
    """Two model calls splitting ``cost`` and ``tokens`` evenly."""
    usage = Usage(input_tokens=tokens // 2)
    return make_run([model_step(i, usage=usage, cost=cost / 2, latency=latency) for i in range(2)])


def test_relative_limit_passes_within_and_reports_deltas() -> None:
    result = budget(max_cost_increase_pct=20, baseline=base())(priced_run(0.11))
    assert result.passed, result.message
    assert result.message == "within budget (vs baseline: cost_usd +10.0%)"
    assert result.details["baseline_cost_usd"] == 0.1
    assert result.details["max_cost_increase_pct"] == 20
    assert result.details["cost_usd_delta"] == pytest.approx(0.01)
    assert result.details["cost_usd_delta_pct"] == pytest.approx(10.0)
    # Metrics without a relative limit still get their deltas, but no verdict.
    assert result.details["tokens_delta"] == 0
    assert result.details["max_tokens_increase_pct"] is None
    assert result.details["skipped"] == []


def test_relative_limit_fails_with_the_percentage_over_baseline() -> None:
    result = budget(max_cost_increase_pct=20, baseline=base())(priced_run(0.13))
    assert not result.passed
    assert result.message == (
        "over budget: cost_usd 0.1300 is +30.0% over baseline 0.1000 (limit +20%)"
    )


def test_relative_limit_equality_passes_and_one_more_fails() -> None:
    # 1000 tokens * 1.1 = 1100 exactly at the limit, even through float rounding.
    limit = budget(max_tokens_increase_pct=10, baseline=base())
    assert limit(priced_run(0.1, tokens=1100)).passed
    assert not limit(priced_run(0.1, tokens=1102)).passed
    # 0.07 * 1.1 is 0.07700000000000001 in floating point; 0.077 still passes.
    assert budget(max_cost_increase_pct=10, baseline=base(cost=0.07))(priced_run(0.077)).passed
    assert budget(max_cost_increase_pct=0, baseline=base())(priced_run(0.1)).passed
    assert not budget(max_cost_increase_pct=0, baseline=base())(priced_run(0.1002)).passed


def test_a_decrease_always_passes() -> None:
    result = budget(max_tokens_increase_pct=0, baseline=base(tokens=2000))(priced_run(0.1))
    assert result.passed
    assert "tokens -50.0%" in result.message


def test_zero_baseline_allows_only_zero() -> None:
    zero = base(cost=0.0)
    over = budget(max_cost_increase_pct=50, baseline=zero)(priced_run(0.0002))
    assert not over.passed
    assert "cost_usd 0.0002 is over baseline 0" in over.message
    assert "zero baseline" in over.message
    assert over.details["cost_usd_delta_pct"] is None
    assert over.details["cost_usd_delta"] == pytest.approx(0.0002)
    same = budget(max_cost_increase_pct=50, baseline=zero)(priced_run(0.0))
    assert same.passed
    assert "cost_usd = baseline 0" in same.message


def test_missing_latency_skips_the_relative_limit_and_says_so() -> None:
    no_run_latency = budget(max_p95_latency_increase_pct=10, baseline=base())(
        priced_run(0.1, latency=None)
    )
    assert no_run_latency.passed
    assert "skipped max_p95_latency_increase_pct (no latency samples in the run)" in (
        no_run_latency.message
    )
    assert no_run_latency.details["skipped"] == [
        "max_p95_latency_increase_pct (no latency samples in the run)"
    ]
    assert no_run_latency.details["p95_latency_ms_delta"] is None
    no_base_latency = budget(
        max_p95_latency_increase_pct=10, max_cost_increase_pct=10, baseline=base(p95=None)
    )(priced_run(0.5))
    assert not no_base_latency.passed  # the cost limit still applies
    assert "skipped max_p95_latency_increase_pct (no latency samples in the baseline)" in (
        no_base_latency.message
    )


def test_latency_relative_limit() -> None:
    check = budget(max_p95_latency_increase_pct=50, baseline=base(p95=200.0))
    assert check(priced_run(0.1, latency=300.0)).passed
    failed = check(priced_run(0.1, latency=301.0))
    assert not failed.passed
    assert "p95_latency_ms 301 is +50.5% over baseline 200 (limit +50%)" in failed.message


def test_absolute_and_relative_limits_together() -> None:
    check = budget(
        max_cost_usd=0.2,
        max_model_calls=2,
        max_cost_increase_pct=20,
        max_model_calls_increase_pct=0,
        baseline=base(),
    )
    ok = check(priced_run(0.105))
    assert ok.passed
    assert ok.message == (
        "within budget (cost_usd, model_calls; vs baseline: cost_usd +5.0%, model_calls +0.0%)"
    )
    both = check(priced_run(0.25))
    assert not both.passed
    assert "cost_usd 0.25 > max_cost_usd 0.2" in both.message
    assert "cost_usd 0.2500 is +150.0% over baseline 0.1000" in both.message


def test_relative_limits_validation() -> None:
    with pytest.raises(ValueError, match=r"need a baseline.*replay baseline SUITE -o FILE"):
        budget(max_cost_increase_pct=10)
    with pytest.raises(ValueError, match=">= 0"):
        budget(max_tokens_increase_pct=-1, baseline=base())
    with pytest.raises(ValueError, match="at least one limit"):
        budget(baseline=base())
    # A baseline with only absolute limits is fine: the deltas are reported.
    result = budget(max_tokens=5000, baseline=base())(priced_run(0.1))
    assert result.passed
    assert result.details["cost_usd_delta_pct"] == 0.0


# --- build_check ------------------------------------------------------------------------------


class SpecArgs(BaseModel):
    id: int


def spec_rule(run: RunResult, *, needle: str = "") -> bool:
    return needle in str(run.output)


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> str:
    """A module the specs can reference as ``check_helpers_fixture:...``."""
    module = types.ModuleType("check_helpers_fixture")
    module.SpecArgs = SpecArgs  # type: ignore[attr-defined]
    module.spec_rule = spec_rule  # type: ignore[attr-defined]
    module.not_a_model = 3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "check_helpers_fixture", module)
    return "check_helpers_fixture"


def test_build_tool_sequence_explicit_and_default(tmp_path: Path) -> None:
    explicit = build_check(
        {"type": "tool_sequence", "expected": ["a", "b"], "mode": "in_order"}, base_dir=tmp_path
    )
    assert isinstance(explicit, ToolSequence)
    assert explicit.expected == ("a", "b")
    assert explicit.mode == "in_order"
    default = build_check({"type": "tool_sequence"}, base_dir=tmp_path, expected_tools=["x"])
    assert isinstance(default, ToolSequence)
    assert default.expected == ("x",)
    assert default.mode == "exact"
    assert default(tool_run("x")).passed


def test_build_tool_sequence_without_expected_or_recording(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no 'expected' list"):
        build_check({"type": "tool_sequence"}, base_dir=tmp_path)


def test_build_tool_checks(tmp_path: Path) -> None:
    required = build_check({"type": "required_tools", "tools": ["a"]}, base_dir=tmp_path)
    forbidden = build_check({"type": "forbidden_tools", "tools": ["rm"]}, base_dir=tmp_path)
    limit = build_check({"type": "max_tool_calls", "limit": 1, "tool": "a"}, base_dir=tmp_path)
    assert isinstance(required, RequiredTools)
    assert isinstance(forbidden, ForbiddenTools)
    assert isinstance(limit, MaxToolCalls)
    assert limit.tool == "a"
    assert required(tool_run("a")).passed
    assert not forbidden(tool_run("rm")).passed
    assert not limit(tool_run("a", "a")).passed


def test_build_budget(tmp_path: Path) -> None:
    check = build_check(
        {"type": "budget", "max_cost_usd": 0.5, "max_model_calls": 3}, base_dir=tmp_path
    )
    assert isinstance(check, Budget)
    assert check.max_cost_usd == 0.5
    assert check.max_model_calls == 3
    with pytest.raises(ConfigError, match="at least one limit"):
        build_check({"type": "budget"}, base_dir=tmp_path)


def test_build_budget_with_relative_limits(tmp_path: Path) -> None:
    spec = {
        "type": "budget",
        "max_cost_increase_pct": 25,
        "max_tokens_increase_pct": 10.5,
        "max_model_calls_increase_pct": 0,
        "max_p95_latency_increase_pct": 50,
    }
    check = build_check(spec, base_dir=tmp_path, baseline=base())
    assert isinstance(check, Budget)
    assert check.baseline == base()
    assert (
        check.max_cost_increase_pct,
        check.max_tokens_increase_pct,
        check.max_model_calls_increase_pct,
        check.max_p95_latency_increase_pct,
    ) == (25, 10.5, 0, 50)
    assert check(priced_run(0.1)).passed
    with pytest.raises(ConfigError, match=r"invalid check of type 'budget': relative limits"):
        build_check(spec, base_dir=tmp_path)
    with pytest.raises(ConfigError, match=">= 0"):
        build_check({"type": "budget", "max_cost_increase_pct": -5}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="max_cost_increase"):
        build_check({"type": "budget", "max_cost_increase": 5}, base_dir=tmp_path)


def test_build_arg_schema_from_inline_file_and_model(tmp_path: Path, helpers: str) -> None:
    run = make_run([call(0, {"id": 1}), call(1, {"id": "x"})])
    inline = build_check(
        {"type": "arg_schema", "tool": "lookup", "schema": ARGS}, base_dir=tmp_path
    )
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "args.json").write_text(json.dumps(ARGS))
    from_file = build_check(
        {"type": "arg_schema", "tool": "lookup", "schema_file": "schemas/args.json"},
        base_dir=tmp_path,
    )
    model = build_check(
        {"type": "arg_schema", "tool": "lookup", "model": f"{helpers}:SpecArgs", "must_call": True},
        base_dir=tmp_path,
    )
    for check in (inline, from_file, model):
        assert isinstance(check, ArgSchema)
        result = check(run)
        assert not result.passed
        assert "step 1" in result.message
    assert isinstance(model, ArgSchema)
    assert model.must_call


def test_build_output_schema_sources(tmp_path: Path, helpers: str) -> None:
    (tmp_path / "out.json").write_text(json.dumps(OUT))
    run = make_run(output='{"answer": "x"}')
    for source in ({"schema": OUT}, {"schema_file": "out.json"}):
        check = build_check(
            {"type": "output_schema", "parse_json": True, **source}, base_dir=tmp_path
        )
        assert isinstance(check, OutputSchema)
        assert check.parse_json
        assert check(run).passed
        assert not output_schema(OUT)(run).passed
    model = build_check(
        {"type": "output_schema", "model": f"{helpers}:SpecArgs"}, base_dir=tmp_path
    )
    assert not model(make_run(output={"answer": "x"})).passed


def test_build_schema_file_is_relative_to_base_dir_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "project"
    base.mkdir()
    (base / "s.json").write_text(json.dumps(OUT))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert build_check({"type": "output_schema", "schema_file": "s.json"}, base_dir=base)
    with pytest.raises(ConfigError, match="cannot read schema_file"):
        build_check({"type": "output_schema", "schema_file": "s.json"}, base_dir=elsewhere)


def test_build_schema_file_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(ConfigError, match="not valid JSON"):
        build_check({"type": "output_schema", "schema_file": "bad.json"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="JSON object"):
        build_check({"type": "output_schema", "schema_file": "list.json"}, base_dir=tmp_path)


def test_build_schema_source_must_be_exactly_one(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="exactly one of schema"):
        build_check({"type": "output_schema"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="got schema, model"):
        build_check(
            {"type": "arg_schema", "tool": "t", "schema": OUT, "model": "m:C"}, base_dir=tmp_path
        )


def test_build_bad_inline_schema_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid JSON Schema"):
        build_check({"type": "output_schema", "schema": {"type": "bogus"}}, base_dir=tmp_path)


def test_build_model_reference_errors(tmp_path: Path, helpers: str) -> None:
    with pytest.raises(ConfigError, match="not a pydantic BaseModel"):
        build_check({"type": "output_schema", "model": f"{helpers}:not_a_model"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="cannot load model"):
        build_check({"type": "output_schema", "model": "no_such_module_xyz:C"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="module:function"):
        build_check({"type": "output_schema", "model": "nocolon"}, base_dir=tmp_path)


def test_build_rule(tmp_path: Path, helpers: str) -> None:
    check = build_check(
        {"type": "rule", "ref": f"{helpers}:spec_rule", "args": {"needle": "hi"}, "name": "greets"},
        base_dir=tmp_path,
    )
    assert isinstance(check, Rule)
    assert check.name == "greets"
    assert check(make_run(output="hi there")).passed
    assert not check(make_run(output="bye")).passed
    default_name = build_check({"type": "rule", "ref": f"{helpers}:spec_rule"}, base_dir=tmp_path)
    assert default_name.name == "rule:spec_rule"


def test_build_rule_reference_errors(tmp_path: Path, helpers: str) -> None:
    with pytest.raises(ConfigError, match="cannot load rule"):
        build_check({"type": "rule", "ref": f"{helpers}:missing"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="not callable"):
        build_check({"type": "rule", "ref": f"{helpers}:not_a_model"}, base_dir=tmp_path)


def test_build_unknown_and_missing_type(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown check type") as info:
        build_check({"type": "nope"}, base_dir=tmp_path)
    assert "tool_sequence" in str(info.value)
    with pytest.raises(ConfigError, match="type: missing"):
        build_check({"tools": ["a"]}, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("spec", "field"),
    [
        ({"type": "required_tools", "tool": ["a"]}, "tool"),
        ({"type": "forbidden_tools", "tools": ["a"], "extra": 1}, "extra"),
        ({"type": "max_tool_calls", "limit": 1, "tools": "a"}, "tools"),
        ({"type": "tool_sequence", "expectd": ["a"]}, "expectd"),
        ({"type": "budget", "max_cost": 1}, "max_cost"),
        ({"type": "rule", "ref": "m:f", "arg": {}}, "arg"),
        ({"type": "arg_schema", "tool": "t", "schema": OUT, "must_cal": True}, "must_cal"),
        ({"type": "output_schema", "schema": OUT, "parse_jsn": True}, "parse_jsn"),
    ],
)
def test_build_typo_names_the_field(tmp_path: Path, spec: dict[str, object], field: str) -> None:
    with pytest.raises(ConfigError) as info:
        build_check(spec, base_dir=tmp_path)
    assert field in str(info.value)
    assert f"check of type {spec['type']!r}" in str(info.value)


def test_build_wrong_value_types_and_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="limit"):
        build_check({"type": "max_tool_calls", "limit": -1}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="limit"):
        build_check({"type": "max_tool_calls"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="tools"):
        build_check({"type": "required_tools", "tools": "abc"}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="mode"):
        build_check({"type": "tool_sequence", "expected": [], "mode": "loose"}, base_dir=tmp_path)


def test_config_error_is_a_replay_error() -> None:
    assert issubclass(ConfigError, ReplayError)
