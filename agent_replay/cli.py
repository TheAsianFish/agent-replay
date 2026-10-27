"""The ``replay`` command line.

Exit codes: 0 = everything passed, 1 = a check failed or the agent errored
(a regression), 2 = bad input to the command itself (missing cassette or
suite, unimportable agent, invalid config, missing credentials). In ``replay
ci`` a scenario that cannot run (or, offline, has no cached judge verdict) is
reported and the others still run, and the command exits 2: a broken suite,
not a regression.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import JsonValue, ValidationError
from rich.markup import escape
from rich.table import Table
from sqlalchemy.exc import SQLAlchemyError

from agent_replay import __version__
from agent_replay._cli import DB_OPTION as _DB_OPTION
from agent_replay._cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, console
from agent_replay._cli import fail as _fail
from agent_replay._json import canonical_json, to_jsonable
from agent_replay._logging import configure_logging
from agent_replay.calibrate import commands_calibrate, commands_golden
from agent_replay.cassette import Cassette, Kind
from agent_replay.checks import CheckResult, tool_sequence
from agent_replay.checks.tool_sequence import SequenceMode
from agent_replay.ci import (
    Overrides,
    RunSink,
    ScenarioReport,
    measure_suite,
    run_suite,
)
from agent_replay.ci import to_json as report_json
from agent_replay.ci import to_markdown as report_markdown
from agent_replay.diff import (
    DiffStep,
    cost_text,
    describe_change,
    diff_steps,
    steps_from_cassette,
    steps_from_record,
    summary,
)
from agent_replay.diff import to_json as diff_json
from agent_replay.diff import to_markdown as diff_markdown
from agent_replay.errors import MeasurementError, ReplayError, StoreError
from agent_replay.flake import (
    FlakeReport,
    FlakeSettings,
    describe,
    flake_suite,
    gate_summary,
    interval_name,
)
from agent_replay.flake import to_json as flake_json
from agent_replay.flake import to_markdown as flake_markdown
from agent_replay.judge.calibration import CalibrationReport
from agent_replay.pricing import format_cost, prices
from agent_replay.record import wrap
from agent_replay.runner import RunResult, load_agent, make_client, record, replay
from agent_replay.session import Policy as PolicyName
from agent_replay.session import ReplayMode
from agent_replay.store import Store
from agent_replay.trajectory import Step, StepStatus, Trajectory

app = typer.Typer(
    name="replay",
    help="Record an LLM agent's run into a cassette and replay it as a regression test.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
_STATUS_STYLE = {
    StepStatus.LIVE: "cyan",
    StepStatus.EXACT: "green",
    StepStatus.APPROXIMATE: "yellow",
    StepStatus.UNRECORDED: "bold red",
    StepStatus.FALLBACK: "magenta",
    StepStatus.SYNTH: "magenta",
}


class Mode(StrEnum):
    full = "full"
    model_live = "model-live"
    tools_live = "tools-live"


class Policy(StrEnum):
    strict = "strict"
    fallback = "fallback"
    synth = "synth"


class SeqMode(StrEnum):
    exact = "exact"
    in_order = "in_order"


def _version(value: bool) -> None:
    if value:
        console.print(f"agent-replay {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    configure_logging()
    try:
        prices()  # a bad REPLAY_PRICES_FILE fails here, not midway through a recording
    except ReplayError as exc:
        raise _fail(str(exc)) from exc


def _preview(value: JsonValue, limit: int = 60) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _steps_table(trajectory: Trajectory) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("#", "kind", "name", "status", "request"):
        table.add_column(column)
    for step in trajectory.steps:
        style = _STATUS_STYLE[step.status]
        request = step.request
        if step.kind.value == "model" and isinstance(request, dict):
            messages = request.get("messages")
            request = f"{len(messages)} message(s)" if isinstance(messages, list) else "…"
        table.add_row(
            str(step.index),
            step.kind.value,
            escape(step.name),
            f"[{style}]{step.status.value}[/]",
            escape(_preview(request)),
        )
    return table


def _make_client(spec: str) -> object:
    try:
        return make_client(spec)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc


@app.command("record")
def record_cmd(
    agent: Annotated[str, typer.Argument(help="The agent, as module:function(client, task).")],
    task: Annotated[str, typer.Option("--task", "-t", help="Task text passed to the agent.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Cassette file to write.")],
    client: Annotated[
        str,
        typer.Option(
            help="'anthropic' for the real API (needs ANTHROPIC_API_KEY), or a "
            "module:function returning a client."
        ),
    ] = "anthropic",
    force: Annotated[bool, typer.Option(help="Overwrite an existing cassette.")] = False,
) -> None:
    """Run AGENT for real and record every model and tool call into a cassette."""
    if output.exists() and not force:
        raise _fail(f"{output} already exists; pass --force to overwrite it")
    try:
        agent_fn = load_agent(agent)
        model_client = wrap(_make_client(client))
    except (ReplayError, TypeError) as exc:
        raise _fail(str(exc)) from exc
    result, cassette = record(agent_fn, task, client=model_client, agent_ref=agent)
    try:
        cassette.dump(output)
    except OSError as exc:
        raise _fail(
            f"cannot write the cassette to {output}: {exc}; pick a writable --output path"
        ) from exc
    console.print(_steps_table(result.trajectory))
    cost = format_cost(result.cost_usd, result.unpriced_calls)
    tokens = sum(
        (e.usage.input_tokens + e.usage.output_tokens) if e.usage else 0 for e in cassette.envelopes
    )
    console.print(
        f"Recorded {len(cassette.envelopes)} envelope(s) to [bold]{escape(str(output))}[/] "
        f"({tokens} tokens, {cost})"
    )
    if result.error is not None:
        console.print(
            f"[bold red]agent raised[/] {escape(result.error.type)}: {escape(result.error.message)}"
        )
        raise typer.Exit(EXIT_FAILED)
    console.print(f"Output: {escape(str(result.output))}")


def _report(
    path: Path, agent: str, result: RunResult, checks: list[CheckResult], passed: bool
) -> dict[str, Any]:
    return {
        "cassette": str(path),
        "mode": result.mode,
        "agent": agent,
        "task": result.task,
        "passed": passed,
        "error": to_jsonable(result.error),
        "output": result.output,
        "steps": [
            {"index": s.index, "kind": s.kind.value, "name": s.name, "status": s.status.value}
            for s in result.trajectory.steps
        ],
        "unconsumed": [e.seq for e in result.unconsumed],
        "checks": [c.model_dump(mode="json") for c in checks],
    }


def _print_check(check: CheckResult) -> None:
    if check.passed:
        mark = "[bold green]PASS[/]"
    elif check.advisory:
        mark = "[bold yellow]ADVISORY FAIL[/]"
    else:
        mark = "[bold red]FAIL[/]"
    console.print(f"{mark} {check.check}: {escape(check.message)}")
    diff = check.details.get("diff")
    if not check.passed and isinstance(diff, list):
        for line in diff:
            style = {"-": "red", "+": "green"}.get(str(line)[:1], "dim")
            console.print(f"    [{style}]{escape(str(line))}[/]")


@app.command("run")
def run_cmd(
    cassette_path: Annotated[Path, typer.Argument(metavar="CASSETTE", help="Cassette to replay.")],
    mode: Annotated[
        Mode,
        typer.Option(
            help="Cut point: full serves model and tools from the cassette; model-live runs "
            "the model for real against recorded tools (needs --client); tools-live runs the "
            "tools for real against recorded model replies."
        ),
    ] = Mode.full,
    policy: Annotated[
        Policy,
        typer.Option(
            help="For calls the cassette never recorded: strict fails, fallback performs "
            "them for real, synth stubs tool results from their return type."
        ),
    ] = Policy.strict,
    client: Annotated[
        str | None,
        typer.Option(
            help="Live model client for model-live (or fallback): 'anthropic' or module:factory."
        ),
    ] = None,
    agent: Annotated[
        str | None,
        typer.Option(help="Agent to replay (module:function); defaults to the recorded one."),
    ] = None,
    expect: Annotated[
        str | None,
        typer.Option(help="Expected tool names, comma-separated; defaults to the recording."),
    ] = None,
    sequence_mode: Annotated[
        SeqMode, typer.Option(help="exact, or in_order (extra calls allowed).")
    ] = SeqMode.exact,
    strict: Annotated[
        bool, typer.Option(help="Also fail when any step was served approximately.")
    ] = False,
    task: Annotated[
        str | None, typer.Option(help="Task text; defaults to the one in the cassette.")
    ] = None,
    save: Annotated[
        Path | None, typer.Option(help="Write the replayed run as a cassette of its own.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report.")] = False,
) -> None:
    """Replay CASSETTE against the agent's current code and run the checks."""
    replay_mode: ReplayMode = mode.value
    replay_policy: PolicyName = policy.value
    try:
        cassette = Cassette.load(cassette_path)
        agent_ref = agent or cassette.header.agent
        if agent_ref is None:
            raise ReplayError("the cassette does not name its agent; pass --agent module:function")
        agent_fn = load_agent(agent_ref)
        model_client = _make_client(client) if client is not None else None
        result = replay(
            cassette,
            agent_fn,
            mode=replay_mode,
            policy=replay_policy,
            task=task,
            client=model_client,
            save_to=save,
        )
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    except OSError as exc:
        raise _fail(f"cannot write {save}: {exc}; pick a writable --save path") from exc

    expected = (
        [name.strip() for name in expect.split(",") if name.strip()]
        if expect is not None
        else cassette.tool_names()
    )
    seq_mode: SequenceMode = sequence_mode.value
    checks = result.check(tool_sequence(expected, mode=seq_mode))
    approx = result.trajectory.count(StepStatus.APPROXIMATE)
    passed = result.ok and not any(c.blocking for c in checks) and not (strict and approx)

    if json_output:
        print(json.dumps(_report(cassette_path, agent_ref, result, checks, passed), indent=2))
        raise typer.Exit(EXIT_OK if passed else EXIT_FAILED)

    console.print(
        f"Replaying [bold]{escape(str(cassette_path))}[/] (mode {mode.value}, policy "
        f"{policy.value}) against [bold]{escape(agent_ref)}[/]"
    )
    console.print(_steps_table(result.trajectory))
    if approx:
        note = (
            f"{approx} step(s) were served approximately: the request differed from the recording"
        )
        console.print(f"[bold red]FAIL[/] strict: {note}" if strict else f"[yellow]{note}[/]")
    if result.unconsumed:
        names = ", ".join(e.name for e in result.unconsumed)
        console.print(f"[yellow]{len(result.unconsumed)} recorded call(s) never made: {names}[/]")
    if result.error is not None:
        console.print(
            f"[bold red]FAIL[/] run: agent raised {escape(result.error.type)}: "
            f"{escape(result.error.message)}"
        )
    for check in checks:
        _print_check(check)
    if save is not None:
        console.print(f"Saved the replayed run to [bold]{escape(str(save))}[/]")
    console.print("[bold green]passed[/]" if passed else "[bold red]failed[/]")
    raise typer.Exit(EXIT_OK if passed else EXIT_FAILED)


def _print_scenario(scenario: ScenarioReport) -> None:
    if scenario.passed:
        mark = "[bold green]PASS[/]"
    elif scenario.ran:
        mark = "[bold red]FAIL[/]"
    else:
        mark = "[bold yellow]ERROR[/]"
    steps = ", ".join(f"{n} {s}" for s, n in sorted(scenario.steps.by_status.items()))
    console.print(
        f"{mark} {escape(scenario.name)} ({scenario.mode}; {escape(steps) or 'no steps'})"
    )
    if scenario.error is not None:
        console.print(
            f"    [red]run:[/] {escape(scenario.error.type)}: {escape(scenario.error.message)}"
        )
    for check in scenario.failed_checks:
        console.print(f"    [red]{escape(check.check)}:[/] {escape(check.message)}")
    for check in scenario.advisory_failures:
        console.print(f"    [yellow]{escape(check.check)} (advisory):[/] {escape(check.message)}")


def _store_sink(store: Store, suite_path: Path) -> RunSink:
    def save(report: ScenarioReport, result: RunResult | None, started_at: datetime) -> str | None:
        if result is None:
            return None
        return store.save_run(
            result,
            report.checks,
            scenario=report.name,
            cassette_path=report.cassette,
            agent=report.agent,
            passed=report.passed,
            started_at=started_at,
            duration_ms=report.duration_ms,
            suite_id=suite_id[0],
        )

    suite_id: list[str | None] = [None]

    def start(git_sha: str | None) -> str:
        new_id = store.start_suite(config_path=str(suite_path), git_sha=git_sha)
        suite_id[0] = new_id
        return new_id

    def finish(passed: bool, scenarios: int, failed: int) -> None:
        if suite_id[0] is not None:
            store.finish_suite(
                suite_id[0], passed=passed, scenario_count=scenarios, failed_count=failed
            )

    return RunSink(start=start, save=save, finish=finish)


def _write(path: Path, text: str, what: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise _fail(f"cannot write the {what} report to {path}: {exc}") from exc


def _append_step_summary(markdown: str) -> None:
    """Append ``markdown`` to ``$GITHUB_STEP_SUMMARY`` when it is set (GitHub Actions)."""
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        try:
            with Path(summary).open("a", encoding="utf-8") as fh:
                fh.write(markdown)
        except OSError as exc:  # the verdict matters more than the summary
            console.print(f"[yellow]warning:[/] cannot append to GITHUB_STEP_SUMMARY: {exc}")


@app.command("ci")
def ci_cmd(
    suite_path: Annotated[
        Path, typer.Argument(metavar="SUITE", help="Suite file (replay.toml) listing scenarios.")
    ],
    agent: Annotated[
        str | None,
        typer.Option(help="Replay every scenario against this agent (module:function)."),
    ] = None,
    mode: Annotated[Mode | None, typer.Option(help="Override every scenario's mode.")] = None,
    policy: Annotated[Policy | None, typer.Option(help="Override every scenario's policy.")] = None,
    client: Annotated[
        str | None,
        typer.Option(help="Live model client for model-live: 'anthropic' or module:factory."),
    ] = None,
    cassette_dir: Annotated[
        Path | None,
        typer.Option(help="Read each scenario's cassette (by file name) from this directory."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option(
            help="Baseline file (from `replay baseline`) for budget checks' relative limits; "
            "replaces the suite's baseline setting."
        ),
    ] = None,
    fail_on_tool_drift: Annotated[
        bool | None,
        typer.Option(
            "--fail-on-tool-drift/--no-fail-on-tool-drift",
            help="In tools-live, fail a scenario when a live tool's result differs from its "
            "recording (no_tool_drift); replaces the suite's fail_on_tool_drift.",
            show_default=False,
        ),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the JSON report here.")
    ] = None,
    markdown_path: Annotated[
        Path | None, typer.Option("--markdown", help="Write the Markdown summary here.")
    ] = None,
    use_store: Annotated[
        bool, typer.Option("--store/--no-store", help="Save every run in the run store.")
    ] = True,
    db: Annotated[
        str | None,
        typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db."),
    ] = None,
) -> None:
    """Replay every scenario in SUITE, run its checks and fail on any regression.

    Exit 0 when every scenario passed, 1 when any failed, 2 when the suite
    itself is broken (invalid TOML, or a scenario that could not run at all:
    missing cassette, unimportable agent, bad check, no client). Writes the
    Markdown summary to $GITHUB_STEP_SUMMARY too when that is set (GitHub
    Actions).
    """
    overrides = Overrides(
        agent=agent,
        mode=mode.value if mode else None,
        policy=policy.value if policy else None,
        client=client,
        fail_on_tool_drift=fail_on_tool_drift,
        cassette_dir=cassette_dir,
        baseline=baseline,
    )
    try:
        store = Store.open(db) if use_store else None
        sink = _store_sink(store, suite_path) if store is not None else RunSink()
        try:
            report = run_suite(suite_path, overrides=overrides, sink=sink)
        finally:
            if store is not None:
                store.close()
    except ReplayError as exc:
        raise _fail(str(exc)) from exc

    for scenario in report.scenarios:
        _print_scenario(scenario)
    markdown = report_markdown(report)
    if json_path is not None:
        _write(json_path, report_json(report), "JSON")
    if markdown_path is not None:
        _write(markdown_path, markdown, "Markdown")
    _append_step_summary(markdown)
    total, failed = len(report.scenarios), len(report.failed)
    verdict = "[bold green]passed[/]" if report.passed else "[bold red]failed[/]"
    console.print(f"{total - failed}/{total} scenario(s) passed: {verdict}")
    if report.suite_id is not None:
        console.print(f"[dim]stored as suite {report.suite_id[:12]}; see `replay runs`[/]")
    if report.broken:
        console.print(
            f"[bold yellow]{len(report.broken)} scenario(s) could not run; fix the suite[/]"
        )
        raise typer.Exit(EXIT_USAGE)
    raise typer.Exit(EXIT_OK if report.passed else EXIT_FAILED)


@app.command("baseline")
def baseline_cmd(
    suite_path: Annotated[
        Path, typer.Argument(metavar="SUITE", help="Suite file (replay.toml) listing scenarios.")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Baseline file to write.")],
    runs: Annotated[
        int, typer.Option(min=1, help="Replays per scenario; the baseline is their median.")
    ] = 1,
    agent: Annotated[
        str | None,
        typer.Option(help="Replay every scenario against this agent (module:function)."),
    ] = None,
    mode: Annotated[Mode | None, typer.Option(help="Override every scenario's mode.")] = None,
    policy: Annotated[Policy | None, typer.Option(help="Override every scenario's policy.")] = None,
    client: Annotated[
        str | None,
        typer.Option(help="Live model client for model-live: 'anthropic' or module:factory."),
    ] = None,
    cassette_dir: Annotated[
        Path | None,
        typer.Option(help="Read each scenario's cassette (by file name) from this directory."),
    ] = None,
) -> None:
    """Measure every scenario in SUITE and write a cost and latency baseline.

    Replays each scenario --runs times without running checks and records the
    median cost, tokens, model calls and p95 model latency per scenario, for
    budget checks' relative limits (max_cost_increase_pct and friends). Exit 2
    when a scenario cannot run, 1 when the agent fails a run (nothing is
    written).
    """
    overrides = Overrides(
        agent=agent,
        mode=mode.value if mode else None,
        policy=policy.value if policy else None,
        client=client,
        cassette_dir=cassette_dir,
    )
    try:
        baseline = measure_suite(suite_path, overrides=overrides, runs=runs)
    except MeasurementError as exc:
        console.print(f"[bold red]failed:[/] {escape(str(exc))}")
        raise typer.Exit(EXIT_FAILED) from exc
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    try:
        baseline.dump(output)
    except OSError as exc:
        raise _fail(
            f"cannot write the baseline to {output}: {exc}; pick a writable -o path"
        ) from exc
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("scenario", "runs", "cost", "tokens", "model calls", "p95 latency"):
        table.add_column(column)
    for name, b in sorted(baseline.scenarios.items()):
        p95 = f"{b.p95_latency_ms:.0f} ms" if b.p95_latency_ms is not None else "—"
        table.add_row(
            escape(name),
            str(b.runs),
            format_cost(b.cost_usd, b.unpriced_calls),
            str(b.tokens),
            str(b.model_calls),
            p95,
        )
    console.print(table)
    console.print(
        f"Wrote the baseline of {len(baseline.scenarios)} scenario(s) to "
        f"[bold]{escape(str(output))}[/]"
    )


class GateChoice(StrEnum):
    sprt = "sprt"
    wilson = "wilson"


_FLAKE_STYLE = {"passing": "green", "flaky": "yellow", "failing": "red", "broken": "yellow"}


@app.command("flake")
def flake_cmd(
    suite_path: Annotated[
        Path, typer.Argument(metavar="SUITE", help="Suite file (replay.toml) listing scenarios.")
    ],
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", "-s", help="Only this scenario (repeatable)."),
    ] = None,
    gate: Annotated[
        GateChoice,
        typer.Option(help="sprt: stop as soon as the verdict is clear; wilson: fixed runs."),
    ] = GateChoice.sprt,
    max_runs: Annotated[
        int, typer.Option(min=1, help="Most runs per scenario (wilson: exactly this many).")
    ] = 20,
    p_fail: Annotated[
        float, typer.Option(help="sprt: pass rate of a scenario that should fail the gate.")
    ] = 0.6,
    p_pass: Annotated[
        float, typer.Option(help="sprt: pass rate of a scenario that should pass the gate.")
    ] = 0.9,
    alpha: Annotated[
        float, typer.Option(help="sprt: chance of passing a scenario at p_fail.")
    ] = 0.05,
    beta: Annotated[float, typer.Option(help="sprt: chance of failing a scenario at p_pass.")] = (
        0.05
    ),
    min_pass_rate: Annotated[
        float, typer.Option(help="wilson: the interval's lower bound must reach this.")
    ] = 0.8,
    confidence: Annotated[float, typer.Option(help="Confidence of the reported interval.")] = 0.95,
    concurrency: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Runs of one scenario in flight at once. Default: 4 when every tool call is "
            "served, 1 when tools run for real (tools-live, --policy fallback).",
        ),
    ] = None,
    runs_per_minute: Annotated[
        float | None, typer.Option(help="Start at most this many runs a minute.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Base seed for client factories that take a seed.")] = 0,
    agent: Annotated[
        str | None,
        typer.Option(help="Replay every scenario against this agent (module:function)."),
    ] = None,
    mode: Annotated[
        Mode, typer.Option(help="Cut point; model-live is where sampling makes runs differ.")
    ] = Mode.model_live,
    policy: Annotated[Policy | None, typer.Option(help="Override every scenario's policy.")] = None,
    client: Annotated[
        str | None,
        typer.Option(help="Live model client for model-live: 'anthropic' or module:factory."),
    ] = None,
    cassette_dir: Annotated[
        Path | None,
        typer.Option(help="Read each scenario's cassette (by file name) from this directory."),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option(help="Baseline file for budget checks' relative limits."),
    ] = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the JSON report here.")
    ] = None,
    markdown_path: Annotated[
        Path | None, typer.Option("--markdown", help="Write the Markdown summary here.")
    ] = None,
    use_store: Annotated[
        bool, typer.Option("--store/--no-store", help="Save every run in the run store.")
    ] = True,
    db: Annotated[
        str | None,
        typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db."),
    ] = None,
) -> None:
    """Replay every scenario in SUITE repeatedly and gate on its estimated pass rate.

    Each scenario is classified passing, flaky or failing from its runs and
    passes when the gate says so. The statistics come from REPLAY_STATS_IMPL
    (baseline: fixed-k majority and Wald intervals; core: SPRT and Wilson).
    Exit 0 when every scenario passed the gate, 1 when any failed, 2 when the
    suite is broken or a scenario could not run.
    """
    try:
        settings = FlakeSettings(
            gate=gate.value,
            p_fail=p_fail,
            p_pass=p_pass,
            alpha=alpha,
            beta=beta,
            max_runs=max_runs,
            min_pass_rate=min_pass_rate,
            confidence=confidence,
            concurrency=concurrency,
            runs_per_minute=runs_per_minute,
            seed=seed,
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"--{str(e['loc'][0]).replace('_', '-')}: {e['msg']}" if e["loc"] else e["msg"]
            for e in exc.errors()
        )
        raise _fail(problems) from exc
    overrides = Overrides(
        agent=agent,
        mode=mode.value,
        policy=policy.value if policy else None,
        client=client,
        cassette_dir=cassette_dir,
        baseline=baseline,
    )
    try:
        report, store_failure = _flake_and_store(
            suite_path, settings, overrides, scenario, db=db, use_store=use_store
        )
    except ReplayError as exc:
        raise _fail(str(exc)) from exc

    for s in report.scenarios:
        mark = {"pass": "[bold green]PASS[/]", "fail": "[bold red]FAIL[/]"}.get(
            s.decision, "[bold yellow]ERROR[/]"
        )
        style = _FLAKE_STYLE[s.classification]
        console.print(f"{mark} {escape(s.name)} ([{style}]{escape(describe(s))}[/])")
        for name, count in s.failures.items():
            console.print(f"    [red]{escape(name)}:[/] failed {count}/{s.runs} runs")
    markdown = flake_markdown(report)
    if json_path is not None:
        _write(json_path, flake_json(report), "JSON")
    if markdown_path is not None:
        _write(markdown_path, markdown, "Markdown")
    _append_step_summary(markdown)
    total, failed = len(report.scenarios), len(report.failed)
    verdict = "[bold green]passed[/]" if report.passed else "[bold red]failed[/]"
    console.print(
        f"{total - failed}/{total} scenario(s) passed the gate ({len(report.flaky)} flaky; "
        f"{report.runs} runs, {format_cost(report.cost_usd, report.unpriced_calls)}; gate: "
        f"{escape(gate_summary(settings, report.stats_impl))}; "
        f"{interval_name(report.stats_impl)} intervals): {verdict}"
    )
    if report.suite_id is not None:
        console.print(f"[dim]stored as suite {report.suite_id[:12]}; see `replay runs`[/]")
    if store_failure is not None:
        raise _fail(store_failure)
    if report.broken:
        console.print(
            f"[bold yellow]{len(report.broken)} scenario(s) could not run; fix the suite[/]"
        )
        raise typer.Exit(EXIT_USAGE)
    raise typer.Exit(EXIT_OK if report.passed else EXIT_FAILED)


def _flake_and_store(
    suite_path: Path,
    settings: FlakeSettings,
    overrides: Overrides,
    scenarios: list[str] | None,
    *,
    db: str | None,
    use_store: bool,
) -> tuple[FlakeReport, str | None]:
    """Run the flake suite and store its report; a store failure is returned, not raised,
    because the runs are done (and paid for) and the verdict should still be printed."""
    store = Store.open(db) if use_store else None
    sink = _store_sink(store, suite_path) if store is not None else RunSink()
    try:
        report = flake_suite(
            suite_path, settings=settings, overrides=overrides, scenarios=scenarios, sink=sink
        )
        if store is None:
            return report, None
        try:
            store.save_flake(report)
        except (SQLAlchemyError, StoreError) as exc:
            return report, f"could not store the flake report: {exc}"
        return report, None
    finally:
        if store is not None:
            store.close()


@app.command("runs")
def runs_cmd(
    run_id: Annotated[
        str | None, typer.Argument(metavar="[RUN_ID]", help="Show one run (an id prefix works).")
    ] = None,
    limit: Annotated[int, typer.Option(help="How many recent runs to list.")] = 20,
    scenario: Annotated[str | None, typer.Option(help="Only runs of this scenario.")] = None,
    db: Annotated[
        str | None,
        typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db."),
    ] = None,
) -> None:
    """List recent stored runs, or show one run's steps and checks."""
    try:
        with Store.open(db) as store:
            if run_id is None:
                runs = store.list_runs(limit=limit, scenario=scenario)
                table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
                for column in ("id", "started", "scenario", "mode", "result", "steps", "cost"):
                    table.add_column(column)
                for r in runs:
                    table.add_row(
                        r.id[:12],
                        r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                        escape(r.scenario),
                        r.mode,
                        "[green]pass[/]" if r.passed else "[red]fail[/]",
                        str(r.step_count),
                        format_cost(r.cost_usd, r.unpriced_calls),
                    )
                console.print(table if runs else "No runs stored yet; run `replay ci`.")
                return
            run = store.get_run(run_id)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"[bold]{run.id}[/] {escape(run.scenario)}: "
        + ("[green]passed[/]" if run.passed else "[red]failed[/]")
    )
    console.print(f"cassette {escape(run.cassette_path)}, agent {escape(run.agent)}")
    console.print(f"mode {run.mode}, policy {run.policy}, started {run.started_at.isoformat()}")
    steps = [
        Step(
            index=s.index,
            kind=Kind(s.kind),
            name=s.name,
            status=StepStatus(s.status),
            fingerprint=s.fingerprint,
            request=s.request,
        )
        for s in run.steps
    ]
    console.print(_steps_table(Trajectory(steps=tuple(steps))))
    if run.error_type is not None:
        console.print(f"[red]run:[/] {escape(run.error_type)}: {escape(run.error_message or '')}")
    for check in run.checks:
        _print_check(
            CheckResult(
                check=check.check,
                passed=check.passed,
                message=check.message,
                details=check.details,
                advisory=check.advisory,
            )
        )


@app.command("show")
def show_cmd(
    cassette_path: Annotated[Path, typer.Argument(metavar="CASSETTE", help="Cassette to show.")],
) -> None:
    """Print a cassette's header, envelopes and outcome."""
    try:
        cassette = Cassette.load(cassette_path)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    h = cassette.header
    console.print(f"[bold]{escape(str(cassette_path))}[/]  schema v{h.schema_version}")
    console.print(f"agent: {escape(str(h.agent))}\ntask:  {escape(str(h.task))}")
    console.print(f"recorded {h.created_at.isoformat()} with {escape(str(h.recorder))}")
    steps = [
        Step(
            index=e.seq,
            kind=e.kind,
            name=e.name,
            status=StepStatus.LIVE,
            fingerprint=e.fingerprint,
            request=e.request,
            envelope_seq=e.seq,
        )
        for e in cassette.envelopes
    ]
    console.print(_steps_table(Trajectory(steps=tuple(steps))))
    f = cassette.footer
    console.print(f"status: {f.status}\noutput: {escape(str(f.output))}")


_DIFF_STYLE = {"same": "dim", "changed": "yellow", "added": "green", "removed": "red"}


def _is_cassette_ref(ref: str) -> bool:
    return ref.endswith(".jsonl") or Path(ref).is_file()


def _diff_side(ref: str, store: Store | None) -> tuple[list[DiffStep], str]:
    """Steps and a label for one side of `replay diff`: a cassette file or a stored run."""
    if store is None or _is_cassette_ref(ref):
        return steps_from_cassette(Cassette.load(ref)), ref
    try:
        run = store.get_run(ref)
    except StoreError as exc:
        raise StoreError(
            f"{ref!r} is neither a cassette file nor a stored run: {exc} "
            "Pass a .jsonl cassette path or a run id."
        ) from exc
    return steps_from_record(run), run.id[:12]


@app.command("diff")
def diff_cmd(
    base: Annotated[
        str, typer.Argument(metavar="BASE", help="Cassette (.jsonl) or stored run id (prefix).")
    ],
    head: Annotated[
        str, typer.Argument(metavar="HEAD", help="Cassette (.jsonl) or stored run id (prefix).")
    ],
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the JSON diff here.")
    ] = None,
    markdown_path: Annotated[
        Path | None, typer.Option("--markdown", help="Write the Markdown table here.")
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db."),
    ] = None,
) -> None:
    """Align the steps of BASE and HEAD and show what was added, removed or changed.

    Each side is a cassette file or a stored run (see `replay runs`). Steps are
    aligned by sequence alignment, so an inserted call shows as one added row.
    Exit 0 whenever the diff was produced (it is informational), 2 when a side
    cannot be loaded.
    """
    try:
        if _is_cassette_ref(base) and _is_cassette_ref(head):
            base_steps, base_label = _diff_side(base, None)
            head_steps, head_label = _diff_side(head, None)
        else:
            with Store.open(db) as store:
                base_steps, base_label = _diff_side(base, store)
                head_steps, head_label = _diff_side(head, store)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc

    diff = diff_steps(base_steps, head_steps, base_label=base_label, head_label=head_label)
    console.print(f"Diff [bold]{escape(base_label)}[/] → [bold]{escape(head_label)}[/]")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("op", "base", "head", "cost", "changes"):
        table.add_column(column)
    for row in diff.rows:
        style = _DIFF_STYLE[row.op]
        table.add_row(
            f"[{style}]{row.op}[/]",
            escape(row.base.label) if row.base else "—",
            escape(row.head.label) if row.head else "—",
            cost_text(row.cost),
            escape("\n".join(describe_change(c) for c in row.changes)),
        )
    console.print(table)
    if json_path is not None:
        _write(json_path, diff_json(diff), "JSON")
    if markdown_path is not None:
        _write(markdown_path, diff_markdown(diff), "Markdown")
    console.print(escape(summary(diff)))


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8000,
    db: Annotated[
        str | None,
        typer.Option(help="Run store URL; defaults to $REPLAY_DB_URL or .replay/replay.db."),
    ] = None,
    dashboard: Annotated[
        Path | None,
        typer.Option(
            help="Built dashboard to serve at / (default: $REPLAY_DASHBOARD_DIR, then the "
            "build packaged with the server)."
        ),
    ] = None,
) -> None:
    """Serve the dashboard and its read-only API over the run store.

    Logs one structured line per request (REPLAY_LOG_LEVEL defaults to INFO
    here; REPLAY_LOG_FORMAT=json for JSON lines).
    """
    try:
        import uvicorn  # noqa: PLC0415

        from agent_replay.server import create_app  # noqa: PLC0415
        from agent_replay.server.app import dashboard_dir  # noqa: PLC0415
    except ImportError as exc:
        raise _fail(
            f"replay serve needs the server extra ({exc.name} is missing): "
            "pip install 'agent-replay[server]'"
        ) from exc
    static = dashboard or dashboard_dir()
    if static is not None and not (static / "index.html").is_file():
        raise _fail(f"{static} has no index.html; build the dashboard with `make ui` first")
    os.environ.setdefault("REPLAY_LOG_LEVEL", "INFO")
    configure_logging()
    try:
        store = Store.open(db)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    where = (
        str(static)
        if static
        else "none, API only: this install has no dashboard build; the PyPI wheel bundles "
        "one, or run `make ui` in a source checkout, or set REPLAY_DASHBOARD_DIR"
    )
    console.print(
        f"serving the {store.backend} run store on http://{host}:{port} (dashboard: "
        f"{escape(where)}; API docs at /docs)"
    )
    try:
        uvicorn.run(
            create_app(store, static_dir=static),
            host=host,
            port=port,
            access_log=False,
            log_level="warning",
        )
    finally:
        store.close()


db_app = typer.Typer(help="Manage the run store.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@db_app.command("upgrade")
def db_upgrade_cmd(db: Annotated[str | None, _DB_OPTION] = None) -> None:
    """Create the run store, or migrate it to the newest schema revision.

    Every command that opens the store does this too; run it on its own to
    prepare a shared PostgreSQL store before the dashboard starts.
    """
    try:
        with Store.open(db) as store:
            revision = store.revision()
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"run store at schema revision {revision}")


@db_app.command("add-calibration")
def db_add_calibration_cmd(
    report_path: Annotated[
        Path, typer.Argument(metavar="REPORT", help="Calibration report (JSON) to store.")
    ],
    db: Annotated[str | None, _DB_OPTION] = None,
) -> None:
    """Store a calibration report so the dashboard's calibration page shows it."""
    try:
        report = CalibrationReport.load(report_path)
        with Store.open(db) as store:
            report_id = store.save_calibration(report)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"stored calibration report {report_id[:12]} for rubric {escape(report.rubric)} "
        f"({report.n_items} items, judge {escape(report.judge_id)})"
    )


commands_golden.register(app)
commands_calibrate.register(app)


def main() -> None:
    app()
