"""``replay ci``: replay a suite of cassettes, run their checks, gate on the result.

A suite is a TOML file (``replay.toml``) listing scenarios. Each scenario is a
cassette, the agent to replay it against, a cut-point mode, an unrecorded-call
policy and a list of checks::

    [defaults]
    agent = "examples.retail_agent.agent:run"
    mode = "full"
    checks = [{ type = "tool_sequence" }]

    [[scenario]]
    name = "cancel_pending_laptop"
    cassette = "cassettes/cancel_pending_laptop.jsonl"
    checks = [{ type = "required_tools", tools = ["cancel_pending_order"] }]

Paths are relative to the suite file. A scenario's checks are the defaults'
checks followed by its own. A scenario passes when the agent finished without
error, every check passed (advisory checks are reported but never fail it) and
(with ``fail_on_approximate``) no step was served approximately and (with
``fail_on_tool_drift``) every tool that ran live returned what it returned when
recorded (:mod:`agent_replay.drift`; ``ignore_fields`` maps a tool to result
paths left out of that comparison)::

    [defaults]
    fail_on_tool_drift = true
    [defaults.ignore_fields]
    get_order_details = ["fulfillments[*].tracking_id"]

``identify`` maps a tool to the arguments that name the entity it reads or
writes (:mod:`agent_replay.identify`). A served call is then never given a
recording made for another entity: with no unused recording of the same
identity it is unrecorded and the policy decides. In ``tools-live`` with
``fail_on_tool_drift``, a live call for an entity no recording has is drift
(``identity_unrecorded``)::

    [defaults.identify]
    get_order_details = ["order_id"]

The suite passes when every scenario does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_replay.baseline import BaselineFile, RunMetrics, ScenarioBaseline, aggregate, measure
from agent_replay.cassette import Cassette, ErrorInfo
from agent_replay.checks.base import Check, CheckResult
from agent_replay.checks.budget import RELATIVE_LIMITS
from agent_replay.checks.spec import build_check
from agent_replay.drift import describe, parse_path, tool_drift
from agent_replay.errors import ConfigError, JudgeCacheMiss, MeasurementError, ReplayError
from agent_replay.identify import parse_identify
from agent_replay.pricing import format_cost
from agent_replay.runner import RunResult, load_agent, make_client, replay
from agent_replay.session import Policy, ReplayMode
from agent_replay.trajectory import StepStatus

log = structlog.get_logger("agent_replay")


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Defaults(_Config):
    agent: str | None = None
    """``module:function``; else the agent named in each cassette."""
    mode: ReplayMode = "full"
    policy: Policy = "strict"
    client: str | None = None
    """``module:factory`` returning a live model client (needed by ``model-live``)."""
    fail_on_approximate: bool = False
    fail_on_tool_drift: bool = False
    """In ``tools-live``, fail when a live tool's result differs from its recording."""
    ignore_fields: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Per tool, result paths the drift comparison skips (values that legitimately vary)."""
    identify: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Per tool, the argument paths that identify the entity a call reads or writes."""
    baseline: str | None = None
    """Baseline file (``replay baseline``) that budget checks' relative limits compare
    against, relative to the suite file."""
    checks: tuple[dict[str, Any], ...] = ()


class Scenario(_Config):
    name: str
    cassette: str
    agent: str | None = None
    mode: ReplayMode | None = None
    policy: Policy | None = None
    client: str | None = None
    task: str | None = None
    fail_on_approximate: bool | None = None
    fail_on_tool_drift: bool | None = None
    ignore_fields: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Added to the defaults' paths for the same tool."""
    identify: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Replaces the defaults' paths for the same tool (``[]`` switches it off)."""
    checks: tuple[dict[str, Any], ...] = ()


class Suite(_Config):
    defaults: Defaults = Defaults()
    scenario: tuple[Scenario, ...] = Field(min_length=1)


class Overrides(_Config):
    """Command-line settings that replace the suite's for every scenario."""

    agent: str | None = None
    mode: ReplayMode | None = None
    policy: Policy | None = None
    client: str | None = None
    fail_on_approximate: bool | None = None
    """Fail every scenario on a served step whose request differs from the recording."""
    fail_on_tool_drift: bool | None = None
    """Fail every scenario on a live tool result that differs from the recording."""
    identify: dict[str, tuple[str, ...]] | None = None
    """Replaces every scenario's identify table (``{}``: no identifying arguments)."""
    cassette_dir: Path | None = None
    """Read every scenario's cassette (by file name) from this directory instead."""
    baseline: Path | None = None
    """Use this baseline file (relative to the working directory) instead of the suite's."""


def load_suite(path: Path) -> Suite:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"suite file {path} does not exist; pass the path to a replay.toml"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    try:
        suite = Suite.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc
    names = [s.name for s in suite.scenario]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"{path}: scenario names must be unique; repeated: {duplicates}")
    for ignore_fields in (suite.defaults.ignore_fields, *(s.ignore_fields for s in suite.scenario)):
        for tool, paths in ignore_fields.items():
            for field in paths:
                try:
                    parse_path(field)
                except ConfigError as exc:
                    raise ConfigError(f"{path}: tool {tool!r}: {exc}") from exc
    for identify in (suite.defaults.identify, *(s.identify for s in suite.scenario)):
        try:
            parse_identify(identify)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    return suite


# --- results --------------------------------------------------------------------


class StepCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    by_status: dict[str, int]


class ScenarioReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    cassette: str
    agent: str
    mode: ReplayMode
    policy: Policy
    passed: bool
    ran: bool = True
    """False when the scenario never ran (missing cassette, unimportable agent, bad
    check, no client): a broken suite rather than a regression."""
    error: ErrorInfo | None = None
    """Why the scenario could not run, or the exception the agent ended with."""
    checks: tuple[CheckResult, ...] = ()
    steps: StepCounts = StepCounts(total=0, by_status={})
    unconsumed: tuple[str, ...] = ()
    """Names of recorded calls the replay never made."""
    output: JsonValue = None
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    """Model calls left out of ``cost_usd`` because their model has no price."""
    tokens: int = 0
    duration_ms: float = 0.0
    run_id: str | None = None
    """Where the run was stored (``replay runs ID``), when a store was used."""
    baseline: ScenarioBaseline | None = None
    """The scenario's entry in the suite's baseline file, when it has one."""

    @property
    def failed_checks(self) -> list[CheckResult]:
        """Checks that failed the scenario (advisory failures are not among them)."""
        return [c for c in self.checks if c.blocking]

    @property
    def advisory_failures(self) -> list[CheckResult]:
        """Checks that failed but are advisory, so the scenario still passed them."""
        return [c for c in self.checks if not c.passed and c.advisory]


class SuiteReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    suite: str
    passed: bool
    started_at: datetime
    duration_ms: float
    git_sha: str | None = None
    suite_id: str | None = None
    scenarios: tuple[ScenarioReport, ...]

    @property
    def failed(self) -> list[ScenarioReport]:
        return [s for s in self.scenarios if not s.passed]

    @property
    def broken(self) -> list[ScenarioReport]:
        """Scenarios that could not run at all; ``replay ci`` exits 2 for these."""
        return [s for s in self.scenarios if not s.ran]


@dataclass(frozen=True)
class RunSink:
    """Where :func:`run_suite` reports as it goes (the CLI plugs the run store in here).

    ``start(git_sha)`` returns a suite id; ``save(report, result, started_at)``
    returns a run id (``result`` is ``None`` when the scenario could not run);
    ``finish(passed, scenarios, failed)`` closes the suite.
    """

    start: Callable[[str | None], str | None] = lambda _sha: None
    save: Callable[[ScenarioReport, RunResult | None, datetime], str | None] = (
        lambda _report, _result, _started: None
    )
    finish: Callable[[bool, int, int], None] = lambda _passed, _scenarios, _failed: None


def run_suite(
    path: Path,
    *,
    overrides: Overrides | None = None,
    sink: RunSink | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timer: Callable[[], float] = time.perf_counter,
) -> SuiteReport:
    """Replay every scenario of the suite at ``path`` and check it.

    Raises :class:`ConfigError` for a broken suite file, or a missing or broken
    baseline file (``baseline`` in ``[defaults]``, or ``overrides.baseline``),
    which is read once for the whole suite. A scenario that cannot
    run (missing cassette, unimportable agent, bad check) is reported with
    ``ran=False`` and the reason; the other scenarios still run, and the suite
    counts as broken rather than regressed.
    """
    suite = load_suite(path)
    overrides = overrides or Overrides()
    sink = sink or RunSink()
    base_dir = path.parent
    baselines = load_baselines(suite, overrides, base_dir)
    started_at = clock()
    started = timer()
    sha = git_sha(base_dir)
    suite_id = sink.start(sha)
    log.info("ci.start", suite=str(path), scenarios=len(suite.scenario))
    reports: list[ScenarioReport] = []
    for scenario in suite.scenario:
        scenario_started = clock()
        outcome, result = run_scenario(
            scenario, suite.defaults, overrides, base_dir, timer, baselines=baselines
        )
        run_id = sink.save(outcome, result, scenario_started)
        reports.append(outcome.model_copy(update={"run_id": run_id}))
        log.info("ci.scenario", name=scenario.name, passed=outcome.passed)
    report = SuiteReport(
        suite=str(path),
        passed=all(r.passed for r in reports),
        started_at=started_at,
        duration_ms=round((timer() - started) * 1000, 3),
        git_sha=sha,
        suite_id=suite_id,
        scenarios=tuple(reports),
    )
    sink.finish(report.passed, len(report.scenarios), len(report.failed))
    log.info("ci.done", passed=report.passed, failed=len(report.failed))
    return report


@dataclass(frozen=True)
class Baselines:
    """A suite's baseline file, read once per command."""

    path: Path
    file: BaselineFile


def load_baselines(suite: Suite, overrides: Overrides, base_dir: Path) -> Baselines | None:
    """The baseline file ``overrides.baseline`` or the suite's ``baseline`` names, if any.

    Raises :class:`ConfigError` when the file is missing or broken.
    """
    if overrides.baseline is not None:
        path = overrides.baseline
    elif suite.defaults.baseline is not None:
        path = base_dir / suite.defaults.baseline
    else:
        return None
    return Baselines(path, BaselineFile.load(path))


def run_scenario(
    scenario: Scenario,
    defaults: Defaults,
    overrides: Overrides,
    base_dir: Path,
    timer: Callable[[], float],
    *,
    baselines: Baselines | None = None,
    with_checks: bool = True,
    seed: int | None = None,
) -> tuple[ScenarioReport, RunResult | None]:
    """Resolve, replay and (``with_checks``) check one scenario.

    Returns ``(report, None)`` when the scenario could not run at all. ``seed``
    goes to a client factory that takes one (see :func:`make_client`).
    """
    baseline = baselines.file.scenarios.get(scenario.name) if baselines is not None else None
    mode = overrides.mode or scenario.mode or defaults.mode
    policy = overrides.policy or scenario.policy or defaults.policy
    cassette_path = (
        overrides.cassette_dir / Path(scenario.cassette).name
        if overrides.cassette_dir is not None
        else base_dir / scenario.cassette
    )
    agent_ref = overrides.agent or scenario.agent or defaults.agent
    started = timer()

    def failed(exc: Exception, agent: str) -> tuple[ScenarioReport, None]:
        log.warning("ci.scenario_error", name=scenario.name, error=str(exc))
        return ScenarioReport(
            name=scenario.name,
            cassette=str(cassette_path),
            agent=agent,
            mode=mode,
            policy=policy,
            passed=False,
            ran=False,
            error=ErrorInfo.from_exception(exc),
            duration_ms=round((timer() - started) * 1000, 3),
            baseline=baseline,
        ), None

    try:
        cassette = Cassette.load(cassette_path)
        agent_ref = agent_ref or cassette.header.agent
        if agent_ref is None:
            raise ConfigError(
                f"scenario {scenario.name!r}: the cassette does not name its agent; "
                "set agent = 'module:function' in the scenario or [defaults]"
            )
        agent = load_agent(agent_ref)
        checks = _checks(scenario, defaults, base_dir, cassette, baselines) if with_checks else []
        client_ref = overrides.client or scenario.client or defaults.client
        client = make_client(client_ref, seed=seed) if client_ref is not None else None
        identify = _identify(scenario, defaults, overrides)
        result = replay(
            cassette,
            agent,
            mode=mode,
            policy=policy,
            task=scenario.task,
            client=client,
            close_client=client is not None,
            identify=identify,
        )
    except ReplayError as exc:
        return failed(exc, agent_ref or "?")

    try:
        check_results = tuple(result.check(*checks))
    except JudgeCacheMiss as exc:
        return failed(exc, agent_ref)
    strict = with_checks and _first_set(
        overrides.fail_on_approximate, scenario.fail_on_approximate, defaults.fail_on_approximate
    )
    approximate = result.trajectory.count(StepStatus.APPROXIMATE)
    if strict and approximate:
        check_results += (
            CheckResult(
                check="no_approximate_steps",
                passed=False,
                message=f"{approximate} step(s) were served approximately: the request "
                "differed from the recording",
            ),
        )
    drift = with_checks and _first_set(
        overrides.fail_on_tool_drift, scenario.fail_on_tool_drift, defaults.fail_on_tool_drift
    )
    if drift and mode == "tools-live":
        check_results += (
            _drift_check(result, cassette, _ignore_fields(scenario, defaults), identify),
        )
    elif drift:
        log.warning(
            "ci.tool_drift_skipped",
            name=scenario.name,
            mode=mode,
            reason="fail_on_tool_drift compares live tool results; only tools-live runs "
            "the tools, so replay this scenario with --mode tools-live",
        )
    by_status: dict[str, int] = {}
    for step in result.trajectory.steps:
        by_status[step.status.value] = by_status.get(step.status.value, 0) + 1
    report = ScenarioReport(
        name=scenario.name,
        cassette=str(cassette_path),
        agent=agent_ref,
        mode=mode,
        policy=policy,
        passed=result.ok and not any(c.blocking for c in check_results),
        error=result.error,
        checks=check_results,
        steps=StepCounts(total=len(result.trajectory.steps), by_status=by_status),
        unconsumed=tuple(e.name for e in result.unconsumed),
        output=result.output,
        cost_usd=result.cost_usd,
        unpriced_calls=result.unpriced_calls,
        tokens=result.tokens,
        duration_ms=round((timer() - started) * 1000, 3),
        baseline=baseline,
    )
    return report, result


def _ignore_fields(scenario: Scenario, defaults: Defaults) -> dict[str, tuple[str, ...]]:
    merged = dict(defaults.ignore_fields)
    for tool, paths in scenario.ignore_fields.items():
        merged[tool] = (*merged.get(tool, ()), *paths)
    return merged


def _identify(
    scenario: Scenario, defaults: Defaults, overrides: Overrides
) -> dict[str, tuple[str, ...]]:
    if overrides.identify is not None:
        return dict(overrides.identify)
    return {**defaults.identify, **scenario.identify}


def _drift_check(
    result: RunResult,
    cassette: Cassette,
    ignore_fields: dict[str, tuple[str, ...]],
    identify: dict[str, tuple[str, ...]],
) -> CheckResult:
    """``no_tool_drift``: every live tool result equals its recording and every identified
    call names an entity the recording has (see :mod:`drift`)."""
    report = tool_drift(result.trajectory, cassette, ignore_fields, identify)
    return CheckResult(
        check="no_tool_drift",
        passed=report.ok,
        message=describe(report),
        details=report.model_dump(mode="json"),
    )


def _first_set(*values: bool | None) -> bool:
    """The first value that is not ``None`` (the last one never is)."""
    return next(v for v in values if v is not None)


def _checks(
    scenario: Scenario,
    defaults: Defaults,
    base_dir: Path,
    cassette: Cassette,
    baselines: Baselines | None,
) -> list[Check]:
    checks: list[Check] = []
    expected = cassette.tool_names()
    baseline = baselines.file.scenarios.get(scenario.name) if baselines is not None else None
    for i, spec in enumerate((*defaults.checks, *scenario.checks)):
        where = f"scenario {scenario.name!r}, check #{i + 1}"
        relative = [k for k in RELATIVE_LIMITS if spec.get(k) is not None]
        if spec.get("type") == "budget" and relative and baseline is None:
            if baselines is None:
                raise ConfigError(
                    f"{where}: budget limits {', '.join(relative)} compare against a "
                    "baseline, and the suite has none; record one with `replay baseline "
                    'SUITE -o FILE` and set baseline = "FILE" in [defaults] (or pass '
                    "--baseline FILE)"
                )
            raise ConfigError(
                f"{where}: budget limits {', '.join(relative)} compare against a baseline, "
                f"and {baselines.path} has no entry for this scenario; re-record it with "
                f"`replay baseline SUITE -o {baselines.path}`"
            )
        try:
            checks.append(
                build_check(spec, base_dir=base_dir, expected_tools=expected, baseline=baseline)
            )
        except ReplayError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return checks


def measure_suite(
    path: Path,
    *,
    overrides: Overrides | None = None,
    runs: int = 1,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timer: Callable[[], float] = time.perf_counter,
) -> BaselineFile:
    """Replay every scenario ``runs`` times, without checks, and aggregate a baseline.

    A baseline is a measurement, so no check runs (this also lets a suite whose
    budget checks need a baseline record its first one). Scenarios resolve
    exactly as in :func:`run_suite`; the suite's own baseline file is not read.
    Raises :class:`ConfigError` for a broken suite or a scenario that cannot
    run, and :class:`MeasurementError` when the agent ends any run in an error:
    nothing is measured from a failing run.
    """
    if runs < 1:
        raise ConfigError(f"runs must be at least 1, got {runs}")
    suite = load_suite(path)
    overrides = overrides or Overrides()
    base_dir = path.parent
    created_at = clock()
    log.info("baseline.start", suite=str(path), scenarios=len(suite.scenario), runs=runs)
    measured: dict[str, list[RunMetrics]] = {}
    broken: list[str] = []
    errored: list[str] = []
    for scenario in suite.scenario:
        for _ in range(runs):
            report, result = run_scenario(
                scenario, suite.defaults, overrides, base_dir, timer, with_checks=False
            )
            if result is None or result.error is not None:
                assert report.error is not None
                problem = f"{scenario.name} ({report.error.type}: {report.error.message})"
                (broken if result is None else errored).append(problem)
                break
            measured.setdefault(scenario.name, []).append(measure(result))
    if broken:
        raise ConfigError(
            "cannot measure a baseline: scenario(s) could not run: " + "; ".join(broken)
        )
    if errored:
        raise MeasurementError(
            "cannot measure a baseline: the agent failed in " + "; ".join(errored) + ". A "
            "baseline is measured from runs that finish; fix the agent or the scenario first"
        )
    log.info("baseline.done", scenarios=len(measured))
    return BaselineFile(
        suite=str(path),
        created_at=created_at,
        git_sha=git_sha(base_dir),
        mode=overrides.mode or suite.defaults.mode,
        policy=overrides.policy or suite.defaults.policy,
        scenarios={name: aggregate(m) for name, m in measured.items()},
    )


def git_sha(cwd: Path) -> str | None:
    """The commit being tested: ``$GITHUB_SHA`` when it is a commit of the checkout at
    ``cwd`` (in Actions it is), else ``git rev-parse HEAD`` there, else ``$GITHUB_SHA``
    as given, else ``None``.

    A workflow that runs this tool on another repository's checkout exports a
    ``GITHUB_SHA`` that means nothing here; the check keeps it out of reports.
    """
    actions = os.environ.get("GITHUB_SHA") or None
    if actions is not None and _git(cwd, "cat-file", "-e", f"{actions}^{{commit}}") is not None:
        return actions
    return _git(cwd, "rev-parse", "HEAD") or actions


def _git(cwd: Path, *args: str) -> str | None:
    """stdout of ``git ARGS`` in ``cwd``, or ``None`` when git is missing or fails."""
    try:
        out = subprocess.run(  # noqa: S603 - arguments are literals from this module
            ["git", *args],  # noqa: S607 - git from PATH
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# --- reports ----------------------------------------------------------------------


def to_json(report: SuiteReport) -> str:
    data = report.model_dump(mode="json")
    for scenario, model in zip(data["scenarios"], report.scenarios, strict=True):
        scenario["failed_checks"] = [c.check for c in model.failed_checks]
        scenario["advisory_failures"] = [c.check for c in model.advisory_failures]
    data["failed"] = [s.name for s in report.failed]
    data["broken"] = [s.name for s in report.broken]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _fence(lines: Sequence[object]) -> str:
    """A code fence longer than any backtick run in ``lines``, so none of them closes it.

    The diff holds tool names and arguments from the agent's run; a line of three
    backticks would otherwise end the block and render the rest as Markdown, in a
    comment the Action posts under its own name.
    """
    longest = max((len(run) for line in lines for run in re.findall(r"`+", str(line))), default=0)
    return "`" * max(3, longest + 1)


def _cost(s: ScenarioReport) -> str:
    """The cost cell: ``$0.0461``, with the change against the baseline when there is one.

    No percentage when either side has unpriced model calls: their cost is unknown."""
    cost = format_cost(s.cost_usd, s.unpriced_calls)
    if s.baseline is None or not s.ran:
        return cost
    if s.baseline.unpriced_calls:
        baseline = format_cost(s.baseline.cost_usd, s.baseline.unpriced_calls)
        return f"{cost} (baseline {baseline})"
    if s.unpriced_calls:
        return f"{cost} (baseline {format_cost(s.baseline.cost_usd)})"
    if s.baseline.cost_usd == 0:
        return f"{cost} (baseline $0)"
    return f"{cost} ({(s.cost_usd - s.baseline.cost_usd) / s.baseline.cost_usd * 100:+.1f}%)"


def to_markdown(report: SuiteReport) -> str:
    """A PR-comment-sized summary: one table row per scenario, details for failures."""
    total = len(report.scenarios)
    failed = report.failed
    verdict = "passed" if report.passed else "failed"
    icon = "✅" if report.passed else "❌"
    lines = [
        f"## {icon} Replay: {total - len(failed)}/{total} scenarios passed",
        "",
        f"Suite `{report.suite}` {verdict}"
        + (f" at `{report.git_sha[:12]}`" if report.git_sha else "")
        + f" in {report.duration_ms / 1000:.2f}s.",
        "",
        "| Scenario | Mode | Result | Checks | Steps | Cost |",
        "|---|---|---|---|---|---|",
    ]
    for s in report.scenarios:
        passed_checks = sum(1 for c in s.checks if c.passed)
        advisory_note = f" (+{len(s.advisory_failures)} advisory)" if s.advisory_failures else ""
        steps = ", ".join(f"{n} {status}" for status, n in sorted(s.steps.by_status.items()))
        result = "✅ pass" if s.passed else ("❌ fail" if s.ran else "⚠️ did not run")
        lines.append(
            f"| `{_cell(s.name)}` | {s.mode} | {result} "
            f"| {passed_checks}/{len(s.checks)}{advisory_note} | {_cell(steps) or '—'} "
            f"| {_cost(s)} |"
        )
    advisory = [s for s in report.scenarios if s.advisory_failures]
    if advisory:
        lines += ["", "Advisory (reported, not gating):", ""]
        for s in advisory:
            for c in s.advisory_failures:
                lines.append(f"- `{_cell(s.name)}` **{c.check}**: {_cell(c.message)}")
    for s in failed:
        lines += ["", f"### {'❌' if s.ran else '⚠️'} `{s.name}`", ""]
        lines.append(f"Cassette `{s.cassette}`, agent `{s.agent}`, {s.mode} / {s.policy}.")
        if s.error is not None:
            lines += ["", f"- **run**: `{s.error.type}`: {_cell(s.error.message)}"]
        for c in s.failed_checks:
            lines.append(f"- **{c.check}**: {_cell(c.message)}")
            diff = c.details.get("diff")
            if isinstance(diff, list) and diff:
                fence = _fence(diff)
                lines += ["", f"  {fence}diff", *(f"  {line}" for line in diff), f"  {fence}"]
        if s.unconsumed:
            lines.append(f"- recorded but never made: {', '.join(s.unconsumed)}")
    return "\n".join(lines) + "\n"
