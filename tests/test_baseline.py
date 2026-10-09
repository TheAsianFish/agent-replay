"""Tests for agent_replay.baseline: measuring runs, aggregating them, the baseline file."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_replay.baseline import (
    BaselineFile,
    RunMetrics,
    ScenarioBaseline,
    aggregate,
    measure,
)
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ConfigError
from agent_replay.runner import RunResult
from agent_replay.trajectory import Step, StepStatus, Trajectory

CREATED = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def step(
    index: int,
    kind: Kind = Kind.MODEL,
    *,
    usage: Usage | None = None,
    cost: float | None = None,
    latency: float | None = None,
) -> Step:
    return Step(
        index=index,
        kind=kind,
        name="m" if kind is Kind.MODEL else "tool",
        status=StepStatus.EXACT,
        fingerprint="fp",
        request={},
        usage=usage,
        cost_usd=cost,
        latency_ms=latency,
    )


def run_of(*steps: Step) -> RunResult:
    return RunResult(mode="full", task="t", trajectory=Trajectory(steps=steps))


def metrics(
    cost: float = 0.0, tokens: int = 0, calls: int = 0, p95: float | None = None
) -> RunMetrics:
    return RunMetrics(cost_usd=cost, tokens=tokens, model_calls=calls, p95_latency_ms=p95)


# --- measure ------------------------------------------------------------------------------


def test_measure_counts_model_steps_cost_tokens_and_p95() -> None:
    usage = Usage(input_tokens=100, output_tokens=50, cache_read_input_tokens=10)
    run = run_of(
        step(0, usage=usage, cost=0.01, latency=100.0),
        step(1, Kind.TOOL, latency=5000.0),  # tool latency is not model latency
        step(2, usage=usage, cost=0.02, latency=300.0),
        step(3, usage=usage, cost=0.005),  # no latency: left out of the p95
    )
    assert measure(run) == RunMetrics(
        cost_usd=0.035, tokens=480, model_calls=3, p95_latency_ms=300.0
    )


def test_measure_an_empty_run() -> None:
    assert measure(run_of()) == metrics()


# --- aggregate ----------------------------------------------------------------------------


def test_aggregate_one_run_is_that_run() -> None:
    assert aggregate([metrics(0.5, 1000, 3, 250.0)]) == ScenarioBaseline(
        cost_usd=0.5, tokens=1000, model_calls=3, p95_latency_ms=250.0, runs=1
    )


def test_aggregate_takes_the_median_of_each_metric_independently() -> None:
    runs = [metrics(0.3, 10, 5, 900.0), metrics(0.1, 30, 4, 100.0), metrics(0.2, 20, 6, 500.0)]
    assert aggregate(runs) == ScenarioBaseline(
        cost_usd=0.2, tokens=20, model_calls=5, p95_latency_ms=500.0, runs=3
    )


def test_aggregate_median_ignores_one_outlying_run() -> None:
    # One slow, dear run must not set the bar: the mean would (0.4, 1000, 700 ms).
    runs = [metrics(0.1, 100, 2, 100.0), metrics(0.1, 100, 2, 100.0), metrics(1.0, 2800, 2, 1900.0)]
    assert aggregate(runs) == ScenarioBaseline(
        cost_usd=0.1, tokens=100, model_calls=2, p95_latency_ms=100.0, runs=3
    )


def test_aggregate_even_runs_rounds_integer_medians_half_up() -> None:
    runs = [metrics(0.1, 10, 3, 100.0), metrics(0.2, 11, 4, 200.0)]
    baseline = aggregate(runs)
    assert baseline.cost_usd == pytest.approx(0.15)
    assert baseline.tokens == 11  # median 10.5
    assert baseline.model_calls == 4  # median 3.5
    assert baseline.p95_latency_ms == 150.0
    assert baseline.runs == 2


def test_aggregate_latency_over_the_runs_that_measured_one() -> None:
    runs = [metrics(p95=None), metrics(p95=100.0), metrics(p95=300.0)]
    assert aggregate(runs).p95_latency_ms == 200.0
    assert aggregate([metrics(), metrics()]).p95_latency_ms is None


def test_aggregate_needs_a_run() -> None:
    with pytest.raises(ValueError, match="at least one"):
        aggregate([])


# --- the file -----------------------------------------------------------------------------


def make_file(**scenarios: ScenarioBaseline) -> BaselineFile:
    return BaselineFile(
        suite="replay.toml",
        created_at=CREATED,
        git_sha="abc123",
        mode="full",
        policy="strict",
        scenarios=scenarios,
    )


def test_dump_is_stable_and_round_trips(tmp_path: Path) -> None:
    b = ScenarioBaseline(cost_usd=0.1, tokens=10, model_calls=1, p95_latency_ms=None, runs=1)
    baseline = make_file(zeta=b, alpha=b)
    path = tmp_path / "nested" / "baseline.json"
    baseline.dump(path)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert text.startswith('{\n  "schema_version": 1,\n')
    assert list(json.loads(text)["scenarios"]) == ["alpha", "zeta"]
    assert BaselineFile.load(path) == baseline
    baseline.dump(path)
    assert path.read_text(encoding="utf-8") == text


def test_load_errors_name_the_fix(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"does not exist; record one with `replay baseline"):
        BaselineFile.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"not valid JSON.*replay baseline SUITE -o PATH"):
        BaselineFile.load(bad)
    data = json.loads(make_file().dumps())
    for change, message in [
        ({"schema_version": 2}, "schema_version"),
        ({"scenarios": {"s": {"cost_usd": 1}}}, r"scenarios\.s\.tokens: Field required"),
        ({"extra": 1}, "extra: Extra inputs"),
    ]:
        bad.write_text(json.dumps({**data, **change}), encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            BaselineFile.load(bad)
    with pytest.raises(ConfigError, match="cannot read"):
        BaselineFile.load(tmp_path)  # a directory
