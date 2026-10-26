"""Tests for ``replay flake``: repeated runs, gates, ordering, overshoot, reports and the CLI.

``agent_replay.stats`` (the SPRT) is not implemented yet, so early stopping is
tested with a scripted ``StatsImpl`` whose gate decides on a fixed rule; its
intervals come from the baseline. Default runs use the baseline (fixed-k
majority, Wald interval).
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import threading
import types
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError
from structlog.testing import capture_logs
from typer.testing import CliRunner

from agent_replay import flake, stats_baseline
from agent_replay.checks import CheckResult
from agent_replay.ci import Overrides, RunSink, ScenarioReport
from agent_replay.cli import app
from agent_replay.errors import ConfigError
from agent_replay.flake import (
    FlakeReport,
    FlakeSettings,
    RateLimiter,
    ScenarioFlake,
    WilsonGate,
    _repeat,
    _summarise,
    classify,
    describe,
    flake_suite,
    gate_summary,
    make_gate,
    run_seed,
    to_json,
    to_markdown,
)
from agent_replay.runner import make_client
from agent_replay.stats import Decision
from agent_replay.store import Store
from examples.retail_agent import tools as retail_tools
from examples.retail_agent.agent import RetailAgent
from examples.retail_agent.flaky_model import SLIP_RATES

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay.toml"
SUITE_ARG = "examples/retail_agent/replay.toml"
FLAKY = "examples.retail_agent.flaky_model:make_client"
FAKE = "examples.retail_agent.fake_model:make_client"
LIVE = Overrides(mode="model-live", client=FLAKY)
SUBSET = ["cancel_pending_laptop", "exchange_waterproof_boots", "transfer_swapped_refunds"]
BOOTS = "exchange_waterproof_boots"
STARTED = datetime(2026, 10, 10, tzinfo=UTC)
ENV = {"COLUMNS": "200"}

runner = CliRunner()


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("REPLAY_STATS_IMPL", raising=False)
    retail_tools.reset_db()
    yield
    retail_tools.reset_db()


def invoke(*args: str):
    return runner.invoke(app, list(args), env=ENV)


def slips(name: str, index: int, seed: int = 0) -> bool:
    """Whether the simulated flaky model slips on run ``index`` of ``name`` (its own rule)."""
    draw = random.Random(run_seed(seed, name, index)).random()
    return draw < SLIP_RATES[name]


# --- a scripted stats implementation ------------------------------------------------

Rule = Callable[[int, int, int], Decision]
"""``rule(runs, passes, max_runs)`` -> the gate's decision after that many runs."""


def after(k: int, verdict: Decision | None = None) -> Rule:
    """Decide after ``k`` runs: ``verdict``, or pass iff every run so far passed."""

    def rule(runs: int, passes: int, _max_runs: int) -> Decision:
        if runs < k:
            return "continue"
        return verdict or ("pass" if passes == runs else "fail")

    return rule


def fail_on_third_failure(runs: int, passes: int, max_runs: int) -> Decision:
    if runs - passes >= 3:
        return "fail"
    return "pass" if runs >= max_runs else "continue"


class ScriptedGate:
    """A test double for ``SequentialGate``: decides by a fixed rule, records what it saw."""

    def __init__(self, rule: Rule, max_runs: int, on_decide: Callable[[], None]) -> None:
        self._rule = rule
        self._max_runs = max_runs
        self._on_decide = on_decide
        self.seen: list[bool] = []
        self._decision: Decision = "continue"

    @property
    def runs(self) -> int:
        return len(self.seen)

    @property
    def passes(self) -> int:
        return sum(self.seen)

    @property
    def decision(self) -> Decision:
        return self._decision

    def update(self, passed: bool) -> Decision:
        if self._decision != "continue":
            raise RuntimeError("fed after deciding")
        self.seen.append(bool(passed))
        self._decision = self._rule(self.runs, self.passes, self._max_runs)
        if self._decision != "continue":
            self._on_decide()
        return self._decision


class ScriptedImpl:
    """``StatsImpl`` whose gates follow ``rule``; everything else is the baseline."""

    def __init__(self, rule: Rule) -> None:
        self._rule = rule
        self.gates: list[ScriptedGate] = []
        self.decided = threading.Event()

    def SequentialGate(
        self,
        p_fail: float,
        p_pass: float,
        *,
        alpha: float = 0.05,
        beta: float = 0.05,
        max_runs: int = 20,
    ) -> ScriptedGate:
        gate = ScriptedGate(self._rule, max_runs, self.decided.set)
        self.gates.append(gate)
        return gate

    cohens_kappa = staticmethod(stats_baseline.cohens_kappa)
    bootstrap_ci = staticmethod(stats_baseline.bootstrap_ci)
    wilson_interval = staticmethod(stats_baseline.wilson_interval)


def report_for(index: int, *, passed: bool, ran: bool = True, cost: float = 0.25):
    checks = () if passed else (CheckResult(check="tool_sequence", passed=False, message="x"),)
    return ScenarioReport(
        name="s",
        cassette="c.jsonl",
        agent="a:b",
        mode="model-live",
        policy="strict",
        passed=passed and ran,
        ran=ran,
        checks=checks,
        cost_usd=cost,
    )


class RecordingPool(ThreadPoolExecutor):
    """Records the gate's decision at every submit (submits happen on the calling thread)."""

    def __init__(self, workers: int) -> None:
        super().__init__(max_workers=workers)
        self.gate: Any = None
        self.decision_at_submit: list[str] = []

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        self.decision_at_submit.append(self.gate.decision)
        return super().submit(fn, *args, **kwargs)


# --- 1. classification and broken scenarios -----------------------------------------


@pytest.mark.parametrize(
    ("passes", "runs", "expected"),
    [(5, 5, "passing"), (1, 1, "passing"), (0, 5, "failing"), (3, 5, "flaky"), (0, 0, "broken")],
)
def test_classify(passes: int, runs: int, expected: str) -> None:
    assert classify(passes, runs) == expected


MISSING_SUITE = """
[defaults]
agent = "examples.retail_agent.agent:run"

[[scenario]]
name = "gone"
cassette = "missing.jsonl"
"""


@pytest.fixture
def missing_suite(tmp_path: Path) -> Path:
    path = tmp_path / "replay.toml"
    path.write_text(MISSING_SUITE, encoding="utf-8")
    return path


def test_a_scenario_that_cannot_run_is_broken_and_stops(missing_suite: Path) -> None:
    report = flake_suite(missing_suite, settings=FlakeSettings(concurrency=1), overrides=LIVE)

    [s] = report.scenarios
    assert s.classification == "broken"
    assert s.decision == "continue"
    assert not s.passed
    assert not report.passed
    assert s.runs == 0
    assert s.executed == 1
    assert s.pass_rate is None
    assert s.error is not None
    assert "missing.jsonl" in s.error.message
    assert report.broken == [s]
    assert describe(s) == f"broken: {s.error.type}: {s.error.message}"


def test_a_broken_scenario_with_concurrency_counts_nothing(missing_suite: Path) -> None:
    report = flake_suite(missing_suite, settings=FlakeSettings(concurrency=4), overrides=LIVE)

    [s] = report.scenarios
    assert s.classification == "broken"
    assert s.runs == 0
    assert 1 <= s.executed <= 4
    assert all(not o.counted for o in s.outcomes[1:])
    assert s.failures == {}


def test_cli_exits_2_for_a_broken_scenario(missing_suite: Path) -> None:
    result = invoke("flake", str(missing_suite), "--no-store", "--concurrency", "1")
    assert result.exit_code == 2, result.output
    assert "gone" in result.output
    assert "broken" in result.output
    assert "could not run" in result.output


# --- 2. the retail suite with the simulated flaky model -------------------------------


@pytest.fixture(scope="module")
def flaky_report() -> FlakeReport:
    retail_tools.reset_db()
    return flake_suite(
        SUITE, settings=FlakeSettings(concurrency=8), overrides=LIVE, scenarios=SUBSET
    )


def by_name(report: FlakeReport) -> dict[str, ScenarioFlake]:
    return {s.name: s for s in report.scenarios}


def test_flaky_model_classifications_match_the_slip_rates(flaky_report: FlakeReport) -> None:
    assert flaky_report.stats_impl == "baseline"
    s = by_name(flaky_report)
    assert [x.name for x in flaky_report.scenarios] == SUBSET  # suite order

    assert SLIP_RATES["cancel_pending_laptop"] == 0.0
    cancel = s["cancel_pending_laptop"]
    assert (cancel.classification, cancel.decision) == ("passing", "pass")
    assert (cancel.passes, cancel.runs, cancel.executed) == (20, 20, 20)
    assert cancel.failures == {}

    assert SLIP_RATES["transfer_swapped_refunds"] == 1.0
    transfer = s["transfer_swapped_refunds"]
    assert (transfer.classification, transfer.decision) == ("failing", "fail")
    assert (transfer.passes, transfer.runs) == (0, 20)
    assert transfer.failures["error:UnrecordedCallError"] == 20
    assert transfer.failures["tool_sequence"] == 20

    boots = s[BOOTS]
    assert boots.classification == "flaky"
    assert 0 < boots.passes < boots.runs == 20
    fails = boots.runs - boots.passes
    assert boots.failures["error:UnrecordedCallError"] == fails
    assert boots.failures["tool_sequence"] == fails
    # The baseline gate is a majority vote at (0.6 + 0.9) / 2.
    assert boots.decision == ("pass" if boots.passes / 20 >= 0.75 else "fail")
    assert boots.interval == stats_baseline.wilson_interval(boots.passes, 20)

    assert flaky_report.flaky == [boots]
    assert {x.name for x in flaky_report.failed} >= {"transfer_swapped_refunds"}


def test_each_run_slips_exactly_when_the_models_seeded_draw_says(
    flaky_report: FlakeReport,
) -> None:
    for s in flaky_report.scenarios:
        for o in s.outcomes:
            assert o.seed == run_seed(0, s.name, o.index)
            assert o.passed == (not slips(s.name, o.index)), (s.name, o.index)
        assert [o.index for o in s.outcomes] == list(range(20))


def test_outcomes_are_the_same_at_concurrency_1_and_8(flaky_report: FlakeReport) -> None:
    serial = flake_suite(
        SUITE, settings=FlakeSettings(concurrency=1), overrides=LIVE, scenarios=[BOOTS]
    )
    [one] = serial.scenarios
    eight = by_name(flaky_report)[BOOTS]

    def key(s: ScenarioFlake) -> list[tuple[int, int, bool, tuple[str, ...]]]:
        return [(o.index, o.seed, o.passed, o.failed_checks) for o in s.outcomes]

    assert key(one) == key(eight)
    assert (one.passes, one.runs, one.decision, one.failures) == (
        eight.passes,
        eight.runs,
        eight.decision,
        eight.failures,
    )


def test_a_different_base_seed_changes_the_draws() -> None:
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(seed=7, max_runs=10, concurrency=4),
        overrides=LIVE,
        scenarios=[BOOTS],
    )
    [s] = report.scenarios
    assert [o.passed for o in s.outcomes] == [not slips(BOOTS, i, seed=7) for i in range(10)]


# --- 3. the gate is fed in run-index order ----------------------------------------------


def test_the_gate_is_fed_in_index_order_when_runs_finish_out_of_order() -> None:
    # Pairs finish in reverse: run 1 before run 0, run 3 before run 2, ...
    finished: list[int] = []
    lock = threading.Lock()
    events = {i: threading.Event() for i in range(6)}
    pattern = [True, False, False, True, True, False]

    def one_run(index: int, seed: int):
        if index % 2 == 0:
            assert events[index + 1].wait(5), f"run {index + 1} never finished"
        with lock:
            finished.append(index)
        events[index].set()
        return report_for(index, passed=pattern[index]), None, STARTED

    impl = ScriptedImpl(after(6))
    settings = FlakeSettings(max_runs=6, concurrency=2)
    gate = make_gate(settings, impl)  # type: ignore[arg-type]
    with ThreadPoolExecutor(max_workers=2) as pool:
        _first, outcomes, broken = _repeat(
            one_run, "s", settings, gate, pool=pool, limiter=None, sink=RunSink()
        )

    assert broken is None
    assert finished == [1, 0, 3, 2, 5, 4]
    assert impl.gates[-1].seen == pattern
    assert sorted(outcomes) == list(range(6))
    assert all(o.counted for o in outcomes.values())


# --- 4. early stop and overshoot --------------------------------------------------------


@pytest.mark.parametrize(("k", "c"), [(1, 4), (2, 4), (2, 3), (3, 5)])
def test_runs_in_flight_at_the_decision_are_not_counted(k: int, c: int) -> None:
    impl = ScriptedImpl(after(k, "pass"))
    settings = FlakeSettings(max_runs=20, concurrency=c)
    gate = make_gate(settings, impl)  # type: ignore[arg-type]
    released: list[bool] = []

    def one_run(index: int, seed: int):
        if index >= k:  # cannot finish before the gate decides
            released.append(impl.decided.wait(5))
            return report_for(index, passed=False), None, STARTED
        return report_for(index, passed=True), None, STARTED

    pool = RecordingPool(c)
    pool.gate = gate
    with pool:
        first, outcomes, broken = _repeat(
            one_run, "s", settings, gate, pool=pool, limiter=None, sink=RunSink()
        )

    assert broken is None
    assert all(released)
    # No run is launched once the gate has decided.
    assert pool.decision_at_submit == ["continue"] * len(pool.decision_at_submit)
    executed = len(outcomes)
    assert c <= executed <= k + c - 1
    assert executed == len(pool.decision_at_submit)
    assert {i for i, o in outcomes.items() if o.counted} == set(range(k))

    s = _summarise(first, settings, stats_baseline, gate, outcomes, broken)
    assert s.decision == "pass"
    assert (s.runs, s.passes, s.pass_rate) == (k, k, 1.0)
    assert s.classification == "passing"  # the failing extra runs do not count
    assert s.interval == stats_baseline.wilson_interval(k, k)
    assert s.failures == {}
    assert s.executed == executed
    assert s.cost_usd == pytest.approx(0.25 * executed)
    assert [o.counted for o in s.outcomes] == [True] * k + [False] * (executed - k)
    assert executed > k
    assert describe(s).endswith(f", {executed - k} more in flight")


def test_early_stop_with_concurrency_1_runs_exactly_k() -> None:
    impl = ScriptedImpl(after(3))
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=1),
        overrides=LIVE,
        scenarios=[BOOTS],
        impl=impl,  # type: ignore[arg-type]
    )
    [s] = report.scenarios
    assert report.stats_impl == "custom"
    assert (s.runs, s.executed) == (3, 3)
    assert all(o.counted for o in s.outcomes)
    assert impl.gates[-1].seen == [not slips(BOOTS, i) for i in range(3)]


def test_early_stop_with_concurrency_excludes_the_overshoot() -> None:
    impl = ScriptedImpl(after(2))
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=5, max_runs=10),
        overrides=LIVE,
        scenarios=[BOOTS],
        impl=impl,  # type: ignore[arg-type]
    )
    [s] = report.scenarios
    assert s.runs == 2
    assert 5 <= s.executed <= 2 + 5 - 1
    counted = [o for o in s.outcomes if o.counted]
    assert [o.index for o in counted] == [0, 1]
    assert s.passes == sum(o.passed for o in counted)
    assert s.interval == stats_baseline.wilson_interval(s.passes, 2)
    assert report.runs == s.executed


def test_a_scripted_fail_on_the_third_failure_stops_there() -> None:
    third = [i for i in range(20) if slips(BOOTS, i)][2]
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=3),
        overrides=LIVE,
        scenarios=[BOOTS],
        impl=ScriptedImpl(fail_on_third_failure),  # type: ignore[arg-type]
    )
    [s] = report.scenarios
    assert s.decision == "fail"
    assert s.runs == third + 1
    assert s.runs - s.passes == 3
    assert s.failures["error:UnrecordedCallError"] == 3
    assert s.failures["tool_sequence"] == 3


# --- 5. the wilson gate -----------------------------------------------------------------


def test_wilson_gate_runs_max_runs_and_compares_the_lower_bound() -> None:
    settings = FlakeSettings(gate="wilson", max_runs=10, min_pass_rate=0.8, concurrency=4)
    report = flake_suite(
        SUITE, settings=settings, overrides=LIVE, scenarios=["cancel_pending_laptop", BOOTS]
    )
    decisions = set()
    for s in report.scenarios:
        assert s.runs == s.executed == 10
        lo, _ = stats_baseline.wilson_interval(s.passes, 10, confidence=0.95)
        assert s.decision == ("pass" if lo >= 0.8 else "fail")
        decisions.add(s.decision)
    assert decisions == {"pass", "fail"}
    assert by_name(report)["cancel_pending_laptop"].passed
    assert "10 runs, pass when the 95% Wald lower bound is at least 0.8" in to_markdown(report)


def test_wilson_gate_boundary_is_inclusive() -> None:
    lo, _ = stats_baseline.wilson_interval(19, 20, confidence=0.9)
    assert 0 < lo < 1

    def decide(min_pass_rate: float) -> Decision:
        gate = WilsonGate(stats_baseline, runs=20, min_pass_rate=min_pass_rate, confidence=0.9)
        decisions = [gate.update(i != 7) for i in range(20)]
        assert decisions[:-1] == ["continue"] * 19
        assert (gate.runs, gate.passes) == (20, 19)
        with pytest.raises(RuntimeError):
            gate.update(True)
        return decisions[-1]

    assert decide(lo) == "pass"
    assert decide(math.nextafter(lo, 1.0)) == "fail"


def test_make_gate_picks_the_gate() -> None:
    assert isinstance(make_gate(FlakeSettings(gate="wilson"), stats_baseline), WilsonGate)
    assert isinstance(make_gate(FlakeSettings(), stats_baseline), stats_baseline.SequentialGate)


# --- 6. settings validation -------------------------------------------------------------


def test_p_fail_must_be_below_p_pass_for_sprt() -> None:
    with pytest.raises(ValidationError, match="p_fail"):
        FlakeSettings(p_fail=0.9, p_pass=0.9)
    with pytest.raises(ValidationError, match="p_fail"):
        FlakeSettings(p_fail=0.95, p_pass=0.9)
    FlakeSettings(gate="wilson", p_fail=0.95, p_pass=0.9)  # unused by the wilson gate


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        FlakeSettings(concurrency=0)


def test_cli_rejects_p_fail_above_p_pass() -> None:
    result = invoke("flake", SUITE_ARG, "--no-store", "--p-fail", "0.95")
    assert result.exit_code == 2, result.output
    assert "p_fail" in result.output
    assert "p_pass" in result.output


def test_cli_rejects_concurrency_0() -> None:
    result = invoke("flake", SUITE_ARG, "--no-store", "--concurrency", "0")
    assert result.exit_code == 2, result.output
    assert "--concurrency" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def test_cli_names_the_option_of_a_bad_setting() -> None:
    result = invoke("flake", SUITE_ARG, "--no-store", "--confidence", "1.5")
    assert result.exit_code == 2, result.output
    assert "--confidence" in result.output


def test_an_unimplemented_core_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REPLAY_STATS_IMPL=core`` before ``stats.py`` is written must exit 2 with the
    fix named, not 1 with a traceback (1 means "a scenario failed the gate")."""
    monkeypatch.setenv("REPLAY_STATS_IMPL", "core")
    result = invoke("flake", SUITE_ARG, "--no-store", "--client", FAKE)
    assert result.exit_code == 2, result.output
    assert "REPLAY_STATS_IMPL" in result.output
    assert "Traceback" not in result.output


def test_unknown_scenario_names_the_known_ones() -> None:
    with pytest.raises(ConfigError, match="nope"):
        flake_suite(SUITE, overrides=LIVE, scenarios=["nope"])
    result = invoke("flake", SUITE_ARG, "--no-store", "-s", "nope")
    assert result.exit_code == 2, result.output
    assert "nope" in result.output
    for name in SLIP_RATES:
        assert name in result.output


# --- 7. the rate limiter ---------------------------------------------------------------


class FakeTime:
    def __init__(self, start: float = 100.0, *, advance: bool = True) -> None:
        self.now = start
        self.advance = advance
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.advance:
            self.now += seconds


def test_rate_limiter_spaces_starts() -> None:
    t = FakeTime()
    limiter = RateLimiter(30, clock=t.clock, sleep=t.sleep)  # one start every 2 s

    limiter.acquire()
    assert t.sleeps == []
    limiter.acquire()
    assert t.sleeps == [pytest.approx(2.0)]
    t.now += 0.5
    limiter.acquire()
    assert t.sleeps[-1] == pytest.approx(1.5)
    t.now += 10  # long enough since the last start: no wait
    limiter.acquire()
    assert len(t.sleeps) == 2
    limiter.acquire()
    assert t.sleeps[-1] == pytest.approx(2.0)


def test_rate_limiter_queues_back_to_back_callers() -> None:
    t = FakeTime(advance=False)  # every caller arrives at the same instant
    limiter = RateLimiter(120, clock=t.clock, sleep=t.sleep)
    for _ in range(4):
        limiter.acquire()
    assert t.sleeps == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(1.5)]


@pytest.mark.parametrize("rate", [0, -1, -0.5])
def test_rate_limiter_rejects_a_non_positive_rate(rate: float) -> None:
    with pytest.raises(ConfigError, match="positive"):
        RateLimiter(rate)


def test_runs_per_minute_builds_the_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``--runs-per-minute`` setting, not only an injected limiter, spaces the starts."""
    t = FakeTime(advance=False)
    built: list[float] = []

    class Recording(RateLimiter):
        def __init__(self, per_minute: float, **kwargs: Any) -> None:
            built.append(per_minute)
            super().__init__(per_minute, clock=t.clock, sleep=t.sleep)

    monkeypatch.setattr(flake, "RateLimiter", Recording)
    flake_suite(
        SUITE,
        settings=FlakeSettings(max_runs=3, concurrency=2, runs_per_minute=60),
        overrides=LIVE,
        scenarios=["cancel_pending_laptop"],
    )
    assert built == [60]
    assert t.sleeps == [pytest.approx(1.0), pytest.approx(2.0)]


def test_flake_suite_uses_the_limiter_for_every_start() -> None:
    t = FakeTime(advance=False)
    flake_suite(
        SUITE,
        settings=FlakeSettings(max_runs=3, concurrency=2),
        overrides=LIVE,
        scenarios=["cancel_pending_laptop"],
        limiter=RateLimiter(60, clock=t.clock, sleep=t.sleep),
    )
    assert t.sleeps == [pytest.approx(1.0), pytest.approx(2.0)]


# --- 8. seeds ---------------------------------------------------------------------------


def test_run_seed_is_stable_and_distinct() -> None:
    assert run_seed(0, "a", 0) == 3002079388
    assert run_seed(1, BOOTS, 3) == 3309653958
    seeds = {run_seed(0, name, i) for name in SLIP_RATES for i in range(20)}
    assert len(seeds) == len(SLIP_RATES) * 20
    assert run_seed(0, BOOTS, 0) != run_seed(1, BOOTS, 0)


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("flake_factories_fixture")

    def seeded(seed: int = -1) -> object:
        return ("seeded", seed)

    def unseeded() -> object:
        return ("unseeded",)

    def positional(seed: int = -1, /) -> object:
        return ("positional", seed)

    module.seeded = seeded  # type: ignore[attr-defined]
    module.unseeded = unseeded  # type: ignore[attr-defined]
    module.positional = positional  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flake_factories_fixture", module)
    return "flake_factories_fixture"


def test_make_client_passes_seed_only_to_factories_that_take_it(factories: str) -> None:
    assert make_client(f"{factories}:seeded", seed=42) == ("seeded", 42)
    assert make_client(f"{factories}:seeded") == ("seeded", -1)
    assert make_client(f"{factories}:unseeded", seed=42) == ("unseeded",)
    # a positional-only ``seed`` cannot be passed by keyword, so it is not passed at all
    assert make_client(f"{factories}:positional", seed=42) == ("positional", -1)


# --- 9. the sink ------------------------------------------------------------------------


def test_sink_sees_every_executed_run_on_the_calling_thread() -> None:
    calls: dict[str, list[Any]] = {"start": [], "save": [], "finish": []}
    caller = threading.get_ident()

    def save(report: ScenarioReport, result: object, started: datetime) -> str:
        calls["save"].append((threading.get_ident(), report.name, result is not None, started))
        return f"run-{len(calls['save'])}"

    sink = RunSink(
        start=lambda sha: calls["start"].append(sha) or "suite-1",
        save=save,
        finish=lambda *args: calls["finish"].append(args),
    )
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=4, max_runs=6),
        overrides=LIVE,
        scenarios=["cancel_pending_laptop", "transfer_swapped_refunds"],
        sink=sink,
        impl=ScriptedImpl(after(1)),  # type: ignore[arg-type]
        clock=lambda: STARTED,
    )

    assert report.suite_id == "suite-1"
    assert calls["start"] == [report.git_sha]
    assert calls["finish"] == [(False, 2, 1)]
    assert len(calls["save"]) == report.runs
    assert report.runs >= 8  # four in flight per scenario when the gate decided
    assert {ident for ident, *_ in calls["save"]} == {caller}
    assert all(has_result and started == STARTED for _, _, has_result, started in calls["save"])
    outcomes = [o for s in report.scenarios for o in s.outcomes]
    assert any(not o.counted for o in outcomes)
    assert sorted(o.run_id for o in outcomes) == sorted(f"run-{i + 1}" for i in range(report.runs))


def test_cli_stores_every_run(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'r.db'}"
    result = invoke(
        "flake",
        SUITE_ARG,
        "-s",
        "cancel_pending_laptop",
        "--client",
        FAKE,
        "--max-runs",
        "3",
        "--concurrency",
        "2",
        "--db",
        db,
    )
    assert result.exit_code == 0, result.output
    assert "stored as suite" in result.output
    with Store.open(db) as store:
        runs = store.list_runs(limit=50)
        suites = store.list_suites()
    assert len(runs) == 3
    assert {r.scenario for r in runs} == {"cancel_pending_laptop"}
    assert all(r.passed for r in runs)
    assert len(suites) == 1


# --- 10. reports --------------------------------------------------------------------------


def test_to_json_round_trips(flaky_report: FlakeReport) -> None:
    data = json.loads(to_json(flaky_report))
    assert data["failed"] == [s.name for s in flaky_report.failed]
    assert data["flaky"] == [BOOTS]
    assert data["broken"] == []
    assert data["runs"] == 60
    assert data["cost_usd"] == flaky_report.cost_usd
    assert data["stats_impl"] == "baseline"
    assert data["settings"]["gate"] == "sprt"
    assert [s["name"] for s in data["scenarios"]] == SUBSET
    assert [s["passed"] for s in data["scenarios"]] == [s.passed for s in flaky_report.scenarios]
    assert len(data["scenarios"][0]["outcomes"]) == 20


def test_markdown_names_the_baseline_gate_and_lists_failures(flaky_report: FlakeReport) -> None:
    md = to_markdown(flaky_report)
    assert "(the baseline stand-in for the SPRT)" in md
    assert "Wald" in md
    assert "`REPLAY_STATS_IMPL=baseline`" in md
    rows = [line for line in md.splitlines() if line.startswith("| `")]
    assert len(rows) == len(SUBSET)
    for name, row in zip(SUBSET, rows, strict=True):
        assert f"`{name}`" in row
    assert "**tool_sequence** failed 20/20 runs" in md
    assert "**error:UnrecordedCallError** failed 20/20 runs" in md
    boots = by_name(flaky_report)[BOOTS]
    fails = boots.runs - boots.passes
    assert f"**tool_sequence** failed {fails}/20 runs" in md


def test_interval_uses_the_settings_confidence() -> None:
    narrow = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=8, confidence=0.5),
        overrides=LIVE,
        scenarios=[BOOTS],
    )
    wide = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=8, confidence=0.99),
        overrides=LIVE,
        scenarios=[BOOTS],
    )
    n_lo, n_hi = narrow.scenarios[0].interval
    w_lo, w_hi = wide.scenarios[0].interval
    assert narrow.scenarios[0].passes == wide.scenarios[0].passes
    assert w_hi - w_lo > n_hi - n_lo
    assert "50% Wald" in to_markdown(narrow)


def test_markdown_shows_runs_in_flight_when_the_gate_decided() -> None:
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=4, max_runs=8),
        overrides=LIVE,
        scenarios=["cancel_pending_laptop"],
        impl=ScriptedImpl(after(1)),  # type: ignore[arg-type]
    )
    scenario = report.scenarios[0]
    assert scenario.runs == 1
    assert scenario.executed > 1
    extra = scenario.executed - scenario.runs
    assert f"| 1 (+{extra}) |" in to_markdown(report)
    assert f", {extra} more in flight" in describe(scenario)


def test_a_report_over_an_unpriced_model_says_unpriced_not_zero(
    flaky_report: FlakeReport,
) -> None:
    # The flaky model answers as `fake:<model>`, which has no price.
    assert flaky_report.cost_usd == 0
    assert flaky_report.unpriced_calls > 0
    assert all(
        s.unpriced_calls == sum(o.unpriced_calls for o in s.outcomes)
        for s in flaky_report.scenarios
    )
    md = to_markdown(flaky_report)
    assert "$0.0000" not in md
    assert f"{flaky_report.runs} runs in" in md
    assert "s, unpriced. Gate:" in md
    rows = [line for line in md.splitlines() if line.startswith("| `")]
    assert all(row.endswith("| unpriced |") for row in rows)
    data = json.loads(to_json(flaky_report))
    assert data["unpriced_calls"] == flaky_report.unpriced_calls
    assert data["scenarios"][0]["unpriced_calls"] == flaky_report.scenarios[0].unpriced_calls
    assert data["scenarios"][0]["outcomes"][0]["unpriced_calls"] > 0


def test_markdown_renders_a_partly_priced_cost(flaky_report: FlakeReport) -> None:
    first, *rest = flaky_report.scenarios
    priced = first.model_copy(update={"cost_usd": 0.0123, "unpriced_calls": 3})
    fully = rest[0].model_copy(update={"cost_usd": 0.5, "unpriced_calls": 0})
    report = flaky_report.model_copy(update={"scenarios": (priced, fully)})
    md = to_markdown(report)
    assert f"`{first.name}`" in md
    assert "| $0.0123 + 3 unpriced calls |" in md
    assert "| $0.5000 |" in md
    assert "s, $0.5123 + 3 unpriced calls. Gate:" in md


def test_old_json_reports_without_unpriced_calls_still_load(flaky_report: FlakeReport) -> None:
    data = json.loads(to_json(flaky_report))
    del data["unpriced_calls"]
    for scenario in data["scenarios"]:
        del scenario["unpriced_calls"]
        for outcome in scenario["outcomes"]:
            del outcome["unpriced_calls"]
    loaded = FlakeReport.model_validate(data)
    assert loaded.unpriced_calls == 0


def test_gate_summary_depends_on_the_implementation() -> None:
    settings = FlakeSettings()
    assert "baseline stand-in for the SPRT" in gate_summary(settings, "baseline")
    assert "pass rate of at least 0.75" in gate_summary(settings, "baseline")
    core = gate_summary(settings, "core")
    assert core.startswith("SPRT p_fail=0.6, p_pass=0.9")
    assert "stand-in" not in core


def test_describe_flaky(flaky_report: FlakeReport) -> None:
    boots = by_name(flaky_report)[BOOTS]
    lo, hi = boots.interval
    assert describe(boots) == (
        f"flaky {boots.passes}/20 ({boots.passes / 20:.0%}), interval [{lo:.2f}, {hi:.2f}]"
    )


def test_markdown_for_a_broken_scenario(missing_suite: Path) -> None:
    report = flake_suite(missing_suite, settings=FlakeSettings(concurrency=1), overrides=LIVE)
    md = to_markdown(report)
    assert "broken" in md
    assert "could not run" in md
    assert json.loads(to_json(report))["broken"] == ["gone"]


# --- 11. the CLI ---------------------------------------------------------------------------


def test_cli_passes_with_the_fake_model_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out_json, out_md = tmp_path / "out" / "flake.json", tmp_path / "out" / "flake.md"
    result = invoke(
        "flake",
        SUITE_ARG,
        "-s",
        "cancel_pending_laptop",
        "-s",
        "lookup_latest_order_total",
        "--client",
        FAKE,
        "--max-runs",
        "3",
        "--no-store",
        "--json",
        str(out_json),
        "--markdown",
        str(out_md),
    )
    assert result.exit_code == 0, result.output
    assert "2/2 scenario(s) passed the gate" in result.output
    assert "6 runs, unpriced; gate:" in result.output  # the fake model has no price
    assert "$0.0000" not in result.output
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["cost_usd"] == 0
    assert data["unpriced_calls"] > 0
    assert [s["classification"] for s in data["scenarios"]] == ["passing", "passing"]
    assert data["runs"] == 6
    markdown = out_md.read_text(encoding="utf-8")
    assert "2/2 scenarios passed the gate" in markdown
    assert summary.read_text(encoding="utf-8") == "before\n" + markdown


def test_cli_exits_1_when_a_scenario_fails_the_gate() -> None:
    result = invoke(
        "flake",
        SUITE_ARG,
        "-s",
        "transfer_swapped_refunds",
        "--client",
        FLAKY,
        "--max-runs",
        "3",
        "--no-store",
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "failing 0/3" in result.output
    assert "tool_sequence: failed 3/3 runs" in result.output


def test_a_slow_first_run_does_not_let_later_runs_pile_up() -> None:
    # Finished runs waiting behind a slow run 0 still occupy the window, so the gate's
    # decision after run 0 wastes at most concurrency - 1 runs, not max_runs - 1.
    go = threading.Event()

    def one_run(index: int, seed: int):
        if index == 0:
            go.wait(5)
        elif index == 1:
            go.set()
        report = ScenarioReport(
            name="s", cassette="c", agent="a", mode="model-live", policy="strict", passed=True
        )
        return report, None, datetime.now(UTC)

    settings = FlakeSettings(max_runs=20, concurrency=2)
    gate = WilsonGate(stats_baseline, runs=1, min_pass_rate=0.5, confidence=0.95)
    with ThreadPoolExecutor(2) as pool:
        _, outcomes, _ = _repeat(
            one_run, "s", settings, gate, pool=pool, limiter=None, sink=RunSink()
        )
    assert gate.decision == "pass"
    assert sorted(outcomes) == [0, 1]
    assert [o.counted for o in outcomes.values()] == [True, False]


# --- review fixes: shared state, plain thread pools, crashing runs -----------------


class _PlainPoolAgent(RetailAgent):
    """Runs a turn's tools on a plain ThreadPoolExecutor: its workers carry no session."""

    def handle_tool_uses(self, blocks):
        with ThreadPoolExecutor(2) as pool:
            return list(pool.map(self.execute, blocks))


class _OverlappingPlainPoolAgent(_PlainPoolAgent):
    """Waits until four runs are inside their sessions, so the calls are ambiguous."""

    barrier = threading.Barrier(4, timeout=10)

    def run(self, client, task):
        self.barrier.wait()
        return super().run(client, task)


plain_pool_run = _PlainPoolAgent().run
overlapping_plain_pool_run = _OverlappingPlainPoolAgent().run
PLAIN = Overrides(mode="model-live", client=FAKE, agent=f"{__name__}:plain_pool_run")


def test_a_plain_thread_pool_agent_passes_one_run_at_a_time() -> None:
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=1, max_runs=3),
        overrides=PLAIN,
        scenarios=["cancel_pending_laptop"],
    )
    [s] = report.scenarios
    assert s.classification == "passing"
    assert (s.passes, s.runs) == (3, 3)


def test_unattributable_calls_under_concurrency_are_broken_not_failing() -> None:
    # Concurrent sessions cannot tell whose plain-pool worker made a call; that is the
    # agent's threading, so the scenario is broken (exit 2), never reported as a flake.
    overrides = PLAIN.model_copy(update={"agent": f"{__name__}:overlapping_plain_pool_run"})
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=4, max_runs=8),
        overrides=overrides,
        scenarios=["cancel_pending_laptop"],
    )
    [s] = report.scenarios
    assert s.classification == "broken"
    assert s.decision == "continue"
    assert (s.runs, s.executed) == (0, 4)
    assert s.error is not None
    assert s.error.type == "UnattributedCallError"
    assert "ContextThreadPoolExecutor" in s.error.message
    assert "--concurrency 1" in s.error.message
    assert report.broken == [s]


def test_runs_that_execute_tools_default_to_one_at_a_time() -> None:
    settings = FlakeSettings(max_runs=6)
    tools_live = flake_suite(
        SUITE,
        settings=settings,
        overrides=Overrides(mode="tools-live"),
        scenarios=["return_damaged_bicycle"],
    )
    served = flake_suite(SUITE, settings=settings, overrides=LIVE, scenarios=[BOOTS])
    fallback = flake_suite(
        SUITE,
        settings=settings,
        overrides=Overrides(mode="model-live", policy="fallback", client=FAKE),
        scenarios=["cancel_pending_laptop"],
    )
    assert [s.concurrency for s in tools_live.scenarios] == [1]
    assert [s.concurrency for s in fallback.scenarios] == [1]
    assert [s.concurrency for s in served.scenarios] == [4]
    # One at a time, the retail database is never shared, so tools-live is stable.
    [s] = tools_live.scenarios
    assert (s.classification, s.passes, s.runs) == ("passing", 6, 6)


def test_explicit_concurrency_with_live_tools_warns() -> None:
    with capture_logs() as logs:
        report = flake_suite(
            SUITE,
            settings=FlakeSettings(max_runs=1, concurrency=3),
            overrides=Overrides(mode="tools-live"),
            scenarios=["lookup_latest_order_total"],
        )
    assert report.scenarios[0].concurrency == 3
    [warning] = [e for e in logs if e["event"] == "flake.shared_tools"]
    assert warning["log_level"] == "warning"
    assert "--concurrency 1" in warning["hint"]


def test_a_run_that_crashes_is_broken_and_the_others_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = flake.run_scenario
    crash_seed = run_seed(0, BOOTS, 2)

    def crashing(*args, seed=None, **kwargs):
        if seed == crash_seed:
            raise KeyError("boom in a worker")
        return real(*args, seed=seed, **kwargs)

    monkeypatch.setattr(flake, "run_scenario", crashing)
    saved: list[int] = []
    finished: list[tuple[bool, int, int]] = []
    sink = RunSink(
        save=lambda report, result, started: saved.append(1) or None,
        finish=lambda passed, scenarios, failed: finished.append((passed, scenarios, failed)),
    )
    report = flake_suite(
        SUITE,
        settings=FlakeSettings(concurrency=1, max_runs=5),
        overrides=LIVE,
        scenarios=[BOOTS],
        sink=sink,
    )
    [s] = report.scenarios
    assert s.classification == "broken"
    assert s.error is not None
    assert s.error.type == "KeyError"
    assert "boom in a worker" in s.error.message
    assert s.runs == 2
    assert s.executed == 3
    assert len(saved) == 3
    assert finished == [(False, 1, 1)]


def test_replay_flake_reports_the_verdict_when_storing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing(self: Store, report: FlakeReport) -> str:
        raise OperationalError("INSERT", {}, Exception("database is locked"))

    monkeypatch.setattr(Store, "save_flake", failing)
    db = f"sqlite:///{tmp_path / 'r.db'}"
    result = runner.invoke(
        app,
        [
            "flake",
            SUITE_ARG,
            "-s",
            "cancel_pending_laptop",
            "--max-runs",
            "2",
            "--client",
            FLAKY,
            "--db",
            db,
        ],
        env=ENV,
    )
    assert result.exit_code == 2, result.output
    assert "cancel_pending_laptop (" in result.output
    assert "scenario(s) passed the gate" in result.output
    assert "could not store the flake report" in result.output
    assert "database is locked" in result.output
