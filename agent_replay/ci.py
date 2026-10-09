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
error, every check passed and (with ``fail_on_approximate``) no step was served
approximately. The suite passes when every scenario does.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_replay.cassette import Cassette, ErrorInfo
from agent_replay.checks.base import Check, CheckResult
from agent_replay.checks.spec import build_check
from agent_replay.errors import ConfigError, ReplayError
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
    cassette_dir: Path | None = None
    """Read every scenario's cassette (by file name) from this directory instead."""


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
    tokens: int = 0
    duration_ms: float = 0.0
    run_id: str | None = None
    """Where the run was stored (``replay runs ID``), when a store was used."""

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


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
    returns a run id (``result`` is ``None`` when the scenario could not run).
    """

    start: Callable[[str | None], str | None] = lambda _sha: None
    save: Callable[[ScenarioReport, RunResult | None, datetime], str | None] = (
        lambda _report, _result, _started: None
    )
    finish: Callable[[SuiteReport], None] = lambda _report: None


def run_suite(
    path: Path,
    *,
    overrides: Overrides | None = None,
    sink: RunSink | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    timer: Callable[[], float] = time.perf_counter,
) -> SuiteReport:
    """Replay every scenario of the suite at ``path`` and check it.

    Raises :class:`ConfigError` for a broken suite file. A scenario that cannot
    run (missing cassette, unimportable agent, bad check) is reported with
    ``ran=False`` and the reason; the other scenarios still run, and the suite
    counts as broken rather than regressed.
    """
    suite = load_suite(path)
    overrides = overrides or Overrides()
    sink = sink or RunSink()
    base_dir = path.parent
    started_at = clock()
    started = timer()
    sha = git_sha(base_dir)
    suite_id = sink.start(sha)
    log.info("ci.start", suite=str(path), scenarios=len(suite.scenario))
    reports: list[ScenarioReport] = []
    for scenario in suite.scenario:
        scenario_started = clock()
        outcome, result = _run_scenario(scenario, suite.defaults, overrides, base_dir, timer)
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
    sink.finish(report)
    log.info("ci.done", passed=report.passed, failed=len(report.failed))
    return report


def _run_scenario(
    scenario: Scenario,
    defaults: Defaults,
    overrides: Overrides,
    base_dir: Path,
    timer: Callable[[], float],
) -> tuple[ScenarioReport, RunResult | None]:
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
        checks = _checks(scenario, defaults, base_dir, cassette)
        client_ref = overrides.client or scenario.client or defaults.client
        client = make_client(client_ref) if client_ref is not None else None
        result = replay(
            cassette, agent, mode=mode, policy=policy, task=scenario.task, client=client
        )
    except ReplayError as exc:
        return failed(exc, agent_ref or "?")

    check_results = tuple(result.check(*checks))
    strict = (
        scenario.fail_on_approximate
        if scenario.fail_on_approximate is not None
        else defaults.fail_on_approximate
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
    by_status: dict[str, int] = {}
    for step in result.trajectory.steps:
        by_status[step.status.value] = by_status.get(step.status.value, 0) + 1
    report = ScenarioReport(
        name=scenario.name,
        cassette=str(cassette_path),
        agent=agent_ref,
        mode=mode,
        policy=policy,
        passed=result.ok and all(c.passed for c in check_results),
        error=result.error,
        checks=check_results,
        steps=StepCounts(total=len(result.trajectory.steps), by_status=by_status),
        unconsumed=tuple(e.name for e in result.unconsumed),
        output=result.output,
        cost_usd=result.cost_usd,
        tokens=result.tokens,
        duration_ms=round((timer() - started) * 1000, 3),
    )
    return report, result


def _checks(
    scenario: Scenario, defaults: Defaults, base_dir: Path, cassette: Cassette
) -> list[Check]:
    checks: list[Check] = []
    expected = cassette.tool_names()
    for i, spec in enumerate((*defaults.checks, *scenario.checks)):
        try:
            checks.append(build_check(spec, base_dir=base_dir, expected_tools=expected))
        except ReplayError as exc:
            raise ConfigError(f"scenario {scenario.name!r}, check #{i + 1}: {exc}") from exc
    return checks


def git_sha(cwd: Path) -> str | None:
    """The commit being tested: ``$GITHUB_SHA`` in Actions, else ``git rev-parse HEAD``."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, fixed arguments
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# --- reports ----------------------------------------------------------------------


def to_json(report: SuiteReport) -> str:
    data = report.model_dump(mode="json")
    for scenario, model in zip(data["scenarios"], report.scenarios, strict=True):
        scenario["failed_checks"] = [c.check for c in model.failed_checks]
    data["failed"] = [s.name for s in report.failed]
    data["broken"] = [s.name for s in report.broken]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


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
        steps = ", ".join(f"{n} {status}" for status, n in sorted(s.steps.by_status.items()))
        result = "✅ pass" if s.passed else ("❌ fail" if s.ran else "⚠️ did not run")
        lines.append(
            f"| `{_cell(s.name)}` | {s.mode} | {result} "
            f"| {passed_checks}/{len(s.checks)} | {_cell(steps) or '—'} | ${s.cost_usd:.4f} |"
        )
    for s in failed:
        lines += ["", f"### {'❌' if s.ran else '⚠️'} `{s.name}`", ""]
        lines.append(f"Cassette `{s.cassette}`, agent `{s.agent}`, {s.mode} / {s.policy}.")
        if s.error is not None:
            lines += ["", f"- **run**: `{s.error.type}`: {_cell(s.error.message)}"]
        for c in s.failed_checks:
            lines.append(f"- **{c.check}**: {_cell(c.message)}")
            diff = c.details.get("diff")
            if isinstance(diff, list) and diff:
                lines += ["", "  ```diff", *(f"  {line}" for line in diff), "  ```"]
        if s.unconsumed:
            lines.append(f"- recorded but never made: {', '.join(s.unconsumed)}")
    return "\n".join(lines) + "\n"
