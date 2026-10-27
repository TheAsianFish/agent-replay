"""``replay flake``: repeat a suite's scenarios and gate on their estimated pass rate.

One run of a ``model-live`` scenario says little: the model samples, so a
scenario that passed once may fail one run in three. This module replays every
scenario of a suite (see :mod:`agent_replay.ci`) repeatedly and treats the
outcomes as Bernoulli trials:

* **Gate.** ``sprt`` (default) feeds the outcomes, in run order, to the
  statistics implementation's :class:`~agent_replay.stats.SequentialGate`,
  which stops as soon as the evidence separates "pass rate ``p_pass``" from
  "pass rate ``p_fail``" (or at ``max_runs``). ``wilson`` runs exactly
  ``max_runs`` times and passes when the lower bound of the pass rate's
  interval is at least ``min_pass_rate``. Both go through
  :func:`agent_replay.stats_impl.load`, so ``REPLAY_STATS_IMPL`` decides
  whether the naive baseline or the core methods run.
* **Classification.** Over the runs the gate counted, a scenario is
  ``passing`` (every run passed), ``failing`` (none did) or ``flaky`` (some
  did). ``broken`` means a run could not happen at all (missing cassette,
  unimportable agent, offline judge cache miss); it stops that scenario.
* **Concurrency.** Up to ``concurrency`` runs of a scenario are in flight at
  once, on threads (each run has its own replay session). Outcomes are fed to
  the gate in run-index order, not completion order: a failing run often ends
  sooner than a passing one, and feeding by completion would bias the test
  towards failures. No run starts more than ``concurrency`` places ahead of the
  last outcome fed to the gate, so at most ``concurrency - 1`` runs are wasted
  when it decides; those still finish and are reported with ``counted=False``.
* **Shared state.** A run that executes tools for real (``tools-live``, or
  the ``fallback`` policy) may race another run on whatever the tools share,
  so such scenarios default to one run at a time; a larger ``concurrency`` is
  honoured with a warning. A call that cannot be attributed to one run (an
  agent starting plain threads while several runs are active) makes the
  scenario ``broken``, not ``failing``: the agent's threading, not its
  behaviour, is at fault.
* **Rate limit.** ``runs_per_minute`` spaces run starts evenly. Each run makes
  several model calls; the Anthropic SDK retries rate-limited calls itself.

Each run gets a seed derived from ``seed``, the scenario name and the run
index; a client factory that takes a ``seed`` parameter receives it, so a
simulated flaky model gives the same outcomes on every invocation.
"""

from __future__ import annotations

import functools
import json
import threading
import time
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_replay import stats_impl
from agent_replay.cassette import ErrorInfo
from agent_replay.ci import (
    Baselines,
    Defaults,
    Overrides,
    RunSink,
    Scenario,
    ScenarioReport,
    git_sha,
    load_baselines,
    load_suite,
    run_scenario,
)
from agent_replay.errors import ConfigError, UnattributedCallError
from agent_replay.pricing import format_cost
from agent_replay.runner import RunResult
from agent_replay.session import Policy, ReplayMode
from agent_replay.stats import Decision
from agent_replay.stats_impl import Gate, StatsImpl

log = structlog.get_logger("agent_replay")

GateKind = Literal["sprt", "wilson"]
Classification = Literal["passing", "flaky", "failing", "broken"]


class FlakeSettings(BaseModel):
    """How often to repeat each scenario and how to decide on the outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate: GateKind = "sprt"
    p_fail: float = Field(0.6, gt=0, lt=1)
    """``sprt``: the pass rate of a scenario that should fail the gate."""
    p_pass: float = Field(0.9, gt=0, lt=1)
    """``sprt``: the pass rate of a scenario that should pass it."""
    alpha: float = Field(0.05, gt=0, lt=1)
    """``sprt``: the chance of passing a scenario whose pass rate is ``p_fail``."""
    beta: float = Field(0.05, gt=0, lt=1)
    """``sprt``: the chance of failing a scenario whose pass rate is ``p_pass``."""
    max_runs: int = Field(20, ge=1)
    """``sprt``: the most runs per scenario; ``wilson``: exactly this many."""
    min_pass_rate: float = Field(0.8, gt=0, lt=1)
    """``wilson``: the interval's lower bound must reach this."""
    confidence: float = Field(0.95, gt=0, lt=1)
    """Confidence of the reported interval (and of the ``wilson`` gate)."""
    concurrency: int | None = Field(None, ge=1)
    """Runs of one scenario in flight at once; ``None``: :func:`default_concurrency`."""
    runs_per_minute: float | None = Field(None, gt=0)
    seed: int = 0

    @model_validator(mode="after")
    def _ordered(self) -> FlakeSettings:
        if self.gate == "sprt" and not self.p_fail < self.p_pass:
            raise ValueError(
                f"p_fail ({self.p_fail}) must be below p_pass ({self.p_pass}): p_fail is the "
                "pass rate of a failing scenario, p_pass of a healthy one"
            )
        return self


class RunOutcome(BaseModel):
    """One repeated run of a scenario."""

    model_config = ConfigDict(frozen=True)

    index: int
    seed: int
    passed: bool
    ran: bool = True
    counted: bool = True
    """False for a run that was in flight when the gate decided."""
    error: ErrorInfo | None = None
    failed_checks: tuple[str, ...] = ()
    steps: dict[str, int] = {}
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    """Model calls left out of ``cost_usd`` because their model has no price."""
    tokens: int = 0
    duration_ms: float = 0.0
    run_id: str | None = None


class ScenarioFlake(BaseModel):
    """A scenario's repeated runs, its estimated pass rate and the gate's decision."""

    model_config = ConfigDict(frozen=True)

    name: str
    cassette: str
    agent: str
    mode: ReplayMode
    policy: Policy
    classification: Classification
    decision: Decision
    """``"pass"`` or ``"fail"``; ``"continue"`` only for a broken scenario."""
    runs: int
    """Runs the gate counted."""
    passes: int
    pass_rate: float | None
    interval: tuple[float, float]
    """Interval on the pass rate at the settings' confidence (Wilson in ``core``, Wald in
    ``baseline``), over the counted runs."""
    failures: dict[str, int] = {}
    """Over the counted runs: how many failed each check, or ended in each error type
    (``error:Type``)."""
    executed: int
    """Runs started, including those in flight when the gate decided."""
    concurrency: int = 1
    """Runs of this scenario that were allowed in flight at once."""
    cost_usd: float = 0.0
    """Over every executed run."""
    unpriced_calls: int = 0
    """Model calls over every executed run that ``cost_usd`` leaves out (no price)."""
    error: ErrorInfo | None = None
    """Why a broken scenario could not run."""
    outcomes: tuple[RunOutcome, ...] = ()

    @property
    def passed(self) -> bool:
        return self.decision == "pass"


class FlakeReport(BaseModel):
    """Every scenario of one ``replay flake`` invocation."""

    model_config = ConfigDict(frozen=True)

    suite: str
    stats_impl: str
    settings: FlakeSettings
    passed: bool
    started_at: datetime
    duration_ms: float
    git_sha: str | None = None
    suite_id: str | None = None
    scenarios: tuple[ScenarioFlake, ...]

    @property
    def failed(self) -> list[ScenarioFlake]:
        return [s for s in self.scenarios if not s.passed]

    @property
    def broken(self) -> list[ScenarioFlake]:
        return [s for s in self.scenarios if s.classification == "broken"]

    @property
    def flaky(self) -> list[ScenarioFlake]:
        return [s for s in self.scenarios if s.classification == "flaky"]

    @property
    def runs(self) -> int:
        """Runs executed over the whole suite (counted or not)."""
        return sum(s.executed for s in self.scenarios)

    @property
    def cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.scenarios), 8)

    @property
    def unpriced_calls(self) -> int:
        return sum(s.unpriced_calls for s in self.scenarios)


# --- gates ----------------------------------------------------------------------


class WilsonGate:
    """Exactly ``runs`` runs, then pass when the interval's lower bound reaches
    ``min_pass_rate``. Satisfies :class:`agent_replay.stats_impl.Gate`."""

    def __init__(
        self, impl: StatsImpl, *, runs: int, min_pass_rate: float, confidence: float
    ) -> None:
        self._impl = impl
        self._max_runs = runs
        self._min_pass_rate = min_pass_rate
        self._confidence = confidence
        self._runs = 0
        self._passes = 0
        self._decision: Decision = "continue"

    @property
    def runs(self) -> int:
        return self._runs

    @property
    def passes(self) -> int:
        return self._passes

    @property
    def decision(self) -> Decision:
        return self._decision

    def update(self, passed: bool) -> Decision:
        if self._decision != "continue":
            raise RuntimeError(f"this gate already decided {self._decision!r}")
        self._runs += 1
        self._passes += bool(passed)
        if self._runs >= self._max_runs:
            lo, _ = self._impl.wilson_interval(
                self._passes, self._runs, confidence=self._confidence
            )
            self._decision = "pass" if lo >= self._min_pass_rate else "fail"
        return self._decision


def make_gate(settings: FlakeSettings, impl: StatsImpl) -> Gate:
    """A fresh gate for one scenario's sequence of runs."""
    if settings.gate == "wilson":
        return WilsonGate(
            impl,
            runs=settings.max_runs,
            min_pass_rate=settings.min_pass_rate,
            confidence=settings.confidence,
        )
    try:
        return impl.SequentialGate(
            settings.p_fail,
            settings.p_pass,
            alpha=settings.alpha,
            beta=settings.beta,
            max_runs=settings.max_runs,
        )
    except NotImplementedError as exc:  # the core is still a specification
        raise ConfigError(
            f"the selected statistics implementation has no SequentialGate yet: {exc}"
        ) from exc


class RateLimiter:
    """Spaces calls to :meth:`acquire` at least ``60 / per_minute`` seconds apart.

    The first call returns at once; there is no burst allowance. Thread-safe.
    """

    def __init__(
        self,
        per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute <= 0:
            raise ConfigError(f"runs_per_minute must be positive, got {per_minute}")
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = now if self._next is None else max(now, self._next)
            self._next = start + self._interval
        if start > now:
            self._sleep(start - now)


# --- the engine -----------------------------------------------------------------


DEFAULT_CONCURRENCY = 4


def tools_may_run(mode: ReplayMode, policy: Policy) -> bool:
    """Whether a replay in ``mode`` with ``policy`` can execute real tool code."""
    return mode == "tools-live" or policy == "fallback"


def default_concurrency(mode: ReplayMode, policy: Policy) -> int:
    """:data:`DEFAULT_CONCURRENCY` when every tool call is served (or stubbed), else 1:
    tools that run for real may share state between concurrent runs."""
    return 1 if tools_may_run(mode, policy) else DEFAULT_CONCURRENCY


def run_seed(seed: int, scenario: str, index: int) -> int:
    """The seed for run ``index`` of ``scenario``: stable across processes and platforms."""
    return zlib.crc32(f"{seed}:{scenario}:{index}".encode())


def classify(passes: int, runs: int) -> Classification:
    if runs == 0:
        return "broken"
    if passes == runs:
        return "passing"
    return "failing" if passes == 0 else "flaky"


def _outcome(index: int, seed: int, report: ScenarioReport) -> RunOutcome:
    return RunOutcome(
        index=index,
        seed=seed,
        passed=report.passed,
        ran=report.ran,
        error=report.error,
        failed_checks=tuple(c.check for c in report.failed_checks),
        steps=dict(report.steps.by_status),
        cost_usd=report.cost_usd,
        unpriced_calls=report.unpriced_calls,
        tokens=report.tokens,
        duration_ms=report.duration_ms,
        run_id=report.run_id,
    )


def _failure_counts(outcomes: Sequence[RunOutcome]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for o in outcomes:
        if o.error is not None:
            counts[f"error:{o.error.type}"] += 1
        counts.update(set(o.failed_checks))
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def flake_suite(
    path: Path,
    *,
    settings: FlakeSettings | None = None,
    overrides: Overrides | None = None,
    scenarios: Sequence[str] | None = None,
    sink: RunSink | None = None,
    impl: StatsImpl | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timer: Callable[[], float] = time.perf_counter,
    limiter: RateLimiter | None = None,
) -> FlakeReport:
    """Replay the scenarios of the suite at ``path`` repeatedly and gate each on its pass rate.

    ``scenarios`` restricts the run to those names (in suite order). Scenarios
    run one after another; the runs of one scenario run ``settings.concurrency``
    at a time (by default :func:`default_concurrency` of its mode and policy).
    ``impl`` defaults to :func:`agent_replay.stats_impl.load`.
    ``sink.save`` is called once per executed run, from the calling thread.

    Raises :class:`ConfigError` for a broken suite or baseline file, or an unknown
    scenario name.
    """
    settings = settings or FlakeSettings()
    overrides = overrides or Overrides()
    sink = sink or RunSink()
    impl = impl or stats_impl.load()
    impl_name = next((n for n in stats_impl.IMPLS if stats_impl.load(n) is impl), "custom")
    suite = load_suite(path)
    if scenarios:
        known = {s.name for s in suite.scenario}
        unknown = [n for n in scenarios if n not in known]
        if unknown:
            raise ConfigError(
                f"unknown scenario(s) {', '.join(unknown)}; {path} has: {', '.join(sorted(known))}"
            )
    chosen = [s for s in suite.scenario if not scenarios or s.name in scenarios]
    base_dir = path.parent
    baselines = load_baselines(suite, overrides, base_dir)
    if limiter is None and settings.runs_per_minute is not None:
        limiter = RateLimiter(settings.runs_per_minute)
    make_gate(settings, impl)  # fail on invalid gate parameters before anything runs

    started_at = clock()
    started = timer()
    sha = git_sha(base_dir)
    suite_id = sink.start(sha)
    log.info(
        "flake.start",
        suite=str(path),
        scenarios=len(chosen),
        gate=settings.gate,
        stats_impl=impl_name,
        concurrency=settings.concurrency or "per scenario",
    )
    results: list[ScenarioFlake] = []
    plans = {s.name: _plan(s, suite.defaults, overrides, settings) for s in chosen}
    with ThreadPoolExecutor(
        max_workers=max(c for _, _, c in plans.values()), thread_name_prefix="replay-flake"
    ) as pool:
        for scenario in chosen:
            mode, policy, concurrency = plans[scenario.name]

            one_run = functools.partial(
                _run_once,
                scenario,
                suite.defaults,
                overrides,
                base_dir,
                baselines,
                mode,
                policy,
                clock=clock,
                timer=timer,
            )
            gate = make_gate(settings, impl)
            first, outcomes, broken = _repeat(
                one_run,
                scenario.name,
                settings,
                gate,
                pool=pool,
                limiter=limiter,
                sink=sink,
                concurrency=concurrency,
            )
            summary = _summarise(first, settings, impl, gate, outcomes, broken)
            results.append(summary.model_copy(update={"concurrency": concurrency}))
            log.info(
                "flake.scenario",
                name=scenario.name,
                classification=results[-1].classification,
                decision=results[-1].decision,
                runs=results[-1].runs,
                executed=results[-1].executed,
            )

    report = FlakeReport(
        suite=str(path),
        stats_impl=impl_name,
        settings=settings,
        passed=all(s.passed for s in results),
        started_at=started_at,
        duration_ms=round((timer() - started) * 1000, 3),
        git_sha=sha,
        suite_id=suite_id,
        scenarios=tuple(results),
    )
    sink.finish(report.passed, len(results), len(report.failed))
    log.info("flake.done", passed=report.passed, runs=report.runs, flaky=len(report.flaky))
    return report


def _plan(
    scenario: Scenario, defaults: Defaults, overrides: Overrides, settings: FlakeSettings
) -> tuple[ReplayMode, Policy, int]:
    """The scenario's mode and policy (resolved as :func:`run_scenario` does) and how many
    of its runs may be in flight at once."""
    mode = overrides.mode or scenario.mode or defaults.mode
    policy = overrides.policy or scenario.policy or defaults.policy
    if settings.concurrency is None:
        return mode, policy, default_concurrency(mode, policy)
    if settings.concurrency > 1 and tools_may_run(mode, policy):
        log.warning(
            "flake.shared_tools",
            name=scenario.name,
            mode=mode,
            policy=policy,
            concurrency=settings.concurrency,
            hint="tools run for real in this mode; if they share state (a database, files, "
            "globals) concurrent runs can interfere. Use --concurrency 1 unless they are "
            "independent",
        )
    return mode, policy, settings.concurrency


_Done = tuple[ScenarioReport, RunResult | None, datetime]
"""A finished run: its report, its result (``None`` if it could not run) and its start."""


def _run_once(
    scenario: Scenario,
    defaults: Defaults,
    overrides: Overrides,
    base_dir: Path,
    baselines: Baselines | None,
    mode: ReplayMode,
    policy: Policy,
    index: int,
    seed: int,
    *,
    clock: Callable[[], datetime],
    timer: Callable[[], float],
) -> _Done:
    """One run of ``scenario``. An unexpected exception becomes a report that did not run,
    so a bug in one run cannot lose the others' results."""
    started = clock()
    try:
        report, result = run_scenario(
            scenario, defaults, overrides, base_dir, timer, baselines=baselines, seed=seed
        )
    except Exception as exc:
        log.exception("flake.run_crashed", name=scenario.name, index=index)
        report = ScenarioReport(
            name=scenario.name,
            cassette=scenario.cassette,
            agent=overrides.agent or scenario.agent or defaults.agent or "?",
            mode=mode,
            policy=policy,
            passed=False,
            ran=False,
            error=ErrorInfo.from_exception(exc),
        )
        result = None
    return report, result, started


def _repeat(
    one_run: Callable[[int, int], _Done],
    name: str,
    settings: FlakeSettings,
    gate: Gate,
    *,
    pool: ThreadPoolExecutor,
    limiter: RateLimiter | None,
    sink: RunSink,
    concurrency: int | None = None,
) -> tuple[ScenarioReport, dict[int, RunOutcome], ScenarioReport | None]:
    """Run ``one_run(index, seed)`` until ``gate`` decides, ``concurrency`` at a time
    (default: ``settings.concurrency``, else :data:`DEFAULT_CONCURRENCY`).

    Returns run 0's report, every executed run's outcome by index, and the
    report of the run that could not run (or made an unattributable call), if any.
    """
    window = concurrency or settings.concurrency or DEFAULT_CONCURRENCY
    in_flight: dict[Future[_Done], int] = {}
    done: dict[int, _Done] = {}
    seeds: dict[int, int] = {}
    outcomes: dict[int, RunOutcome] = {}
    broken: ScenarioReport | None = None
    next_index = 0

    def record(index: int, *, counted: bool) -> None:
        report, result, run_started = done[index]
        run_id = sink.save(report, result, run_started)
        outcomes[index] = _outcome(index, seeds[index], report).model_copy(
            update={"run_id": run_id, "counted": counted}
        )

    while True:
        while (
            gate.decision == "continue"
            and broken is None
            # At most `concurrency` runs ahead of the gate, finished or not: a slow early
            # run must not let later ones pile up past a decision it is about to cause.
            and next_index < len(outcomes) + window
            and next_index < settings.max_runs
        ):
            if limiter is not None:
                limiter.acquire()
            seeds[next_index] = run_seed(settings.seed, name, next_index)
            in_flight[pool.submit(one_run, next_index, seeds[next_index])] = next_index
            next_index += 1
        if not in_flight:
            break
        finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in finished:
            done[in_flight.pop(future)] = future.result()
        # Feed the gate strictly in run order, whatever order the runs finished in.
        while len(outcomes) in done and gate.decision == "continue" and broken is None:
            report = done[len(outcomes)][0]
            usable = report.ran and not _unattributed(report)
            record(len(outcomes), counted=usable)
            if usable:
                gate.update(report.passed)
            else:
                broken = report
    for index in sorted(set(done) - set(outcomes)):
        record(index, counted=False)
    return done[0][0], outcomes, broken


def _unattributed(report: ScenarioReport) -> bool:
    return report.error is not None and report.error.type == UnattributedCallError.__name__


def _summarise(
    first: ScenarioReport,
    settings: FlakeSettings,
    impl: StatsImpl,
    gate: Gate,
    outcomes: dict[int, RunOutcome],
    broken: ScenarioReport | None,
) -> ScenarioFlake:
    """``first`` is run 0's report, which names the cassette, agent, mode and policy."""
    ordered = tuple(outcomes[i] for i in sorted(outcomes))
    counted = [o for o in ordered if o.counted]
    passes = sum(o.passed for o in counted)
    runs = len(counted)
    return ScenarioFlake(
        name=first.name,
        cassette=first.cassette,
        agent=first.agent,
        mode=first.mode,
        policy=first.policy,
        classification="broken" if broken is not None else classify(passes, runs),
        decision="continue" if broken is not None else gate.decision,
        runs=runs,
        passes=passes,
        pass_rate=round(passes / runs, 6) if runs else None,
        interval=impl.wilson_interval(passes, runs, confidence=settings.confidence),
        failures=_failure_counts(counted),
        executed=len(ordered),
        cost_usd=round(sum(o.cost_usd for o in ordered), 8),
        unpriced_calls=sum(o.unpriced_calls for o in ordered),
        error=broken.error if broken is not None else None,
        outcomes=ordered,
    )


# --- reports ----------------------------------------------------------------------


def to_json(report: FlakeReport) -> str:
    data = report.model_dump(mode="json")
    for scenario, model in zip(data["scenarios"], report.scenarios, strict=True):
        scenario["passed"] = model.passed
    data["failed"] = [s.name for s in report.failed]
    data["flaky"] = [s.name for s in report.flaky]
    data["broken"] = [s.name for s in report.broken]
    data["runs"] = report.runs
    data["cost_usd"] = report.cost_usd
    data["unpriced_calls"] = report.unpriced_calls
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


_ICONS: dict[Classification, str] = {
    "passing": "✅",
    "flaky": "⚠️",
    "failing": "❌",
    "broken": "💥",
}


def interval_name(impl_name: str) -> str:
    """What the reported interval is: Wilson from ``core``, Wald from the ``baseline``."""
    return {"core": "Wilson", "baseline": "Wald"}.get(impl_name, impl_name)


def gate_summary(settings: FlakeSettings, impl_name: str) -> str:
    """The gate as it actually ran: the ``baseline`` implementation's ``SequentialGate`` is a
    fixed-k majority vote, not an SPRT."""
    if settings.gate == "wilson":
        return (
            f"{settings.max_runs} runs, pass when the {settings.confidence:.0%} "
            f"{interval_name(impl_name)} lower bound is at least {settings.min_pass_rate}"
        )
    if impl_name == "baseline":
        threshold = (settings.p_fail + settings.p_pass) / 2
        return (
            f"fixed {settings.max_runs} runs, pass on a pass rate of at least {threshold:g} "
            "(the baseline stand-in for the SPRT)"
        )
    return (
        f"SPRT p_fail={settings.p_fail}, p_pass={settings.p_pass}, alpha={settings.alpha}, "
        f"beta={settings.beta}, at most {settings.max_runs} runs"
    )


def describe(s: ScenarioFlake) -> str:
    """One line per scenario: ``flaky 14/20 (70%), 95% [0.48, 0.85]``."""
    if s.classification == "broken":
        reason = f"{s.error.type}: {s.error.message}" if s.error is not None else "could not run"
        return f"broken: {reason}"
    lo, hi = s.interval
    rate = f"{s.passes}/{s.runs} ({(s.pass_rate or 0):.0%})"
    extra = f", {s.executed - s.runs} more in flight" if s.executed > s.runs else ""
    return f"{s.classification} {rate}, interval [{lo:.2f}, {hi:.2f}]{extra}"


def to_markdown(report: FlakeReport) -> str:
    """A PR-comment-sized summary: one row per scenario, failure counts for the rest."""
    total = len(report.scenarios)
    icon = "✅" if report.passed else "❌"
    lines = [
        f"## {icon} Replay flake: {total - len(report.failed)}/{total} scenarios passed the gate",
        "",
        f"Suite `{report.suite}`"
        + (f" at `{report.git_sha[:12]}`" if report.git_sha else "")
        + f": {report.runs} runs in {report.duration_ms / 1000:.1f}s, "
        f"{format_cost(report.cost_usd, report.unpriced_calls)}. "
        f"Gate: {gate_summary(report.settings, report.stats_impl)}. "
        f"Interval: {report.settings.confidence:.0%} {interval_name(report.stats_impl)} "
        f"(`REPLAY_STATS_IMPL={report.stats_impl}`).",
        "",
        "| Scenario | Mode | Gate | Class | Passed | Interval | Runs | Cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report.scenarios:
        gate = {"pass": "✅ pass", "fail": "❌ fail"}.get(s.decision, "—")
        if s.classification == "broken":
            passed, interval = "—", "—"
        else:
            passed = f"{s.passes}/{s.runs}"
            interval = f"[{s.interval[0]:.2f}, {s.interval[1]:.2f}]"
        runs = f"{s.runs}" + (f" (+{s.executed - s.runs})" if s.executed > s.runs else "")
        cost = format_cost(s.cost_usd, s.unpriced_calls)
        lines.append(
            f"| `{_cell(s.name)}` | {s.mode} | {gate} | {_ICONS[s.classification]} "
            f"{s.classification} | {passed} | {interval} | {runs} | {cost} |"
        )
    noted = [s for s in report.scenarios if s.failures or s.error is not None]
    for s in noted:
        lines += ["", f"### {_ICONS[s.classification]} `{_cell(s.name)}`", ""]
        if s.error is not None and s.classification == "broken":
            lines.append(f"- could not run: `{s.error.type}`: {_cell(s.error.message)}")
        for name, count in s.failures.items():
            lines.append(f"- **{_cell(name)}** failed {count}/{s.runs} runs")
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
