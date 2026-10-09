"""The ``replay`` command line.

Exit codes: 0 = everything passed, 1 = a check failed or the agent errored,
2 = bad input (missing cassette, unimportable agent, missing credentials).
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import JsonValue
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from agent_replay import __version__
from agent_replay._json import canonical_json, to_jsonable
from agent_replay._loading import load_object
from agent_replay._logging import configure_logging
from agent_replay.cassette import Cassette
from agent_replay.checks import CheckResult, tool_sequence
from agent_replay.checks.tool_sequence import SequenceMode
from agent_replay.errors import ReplayError
from agent_replay.record import wrap
from agent_replay.runner import RunResult, load_agent, record, replay
from agent_replay.trajectory import Step, StepStatus, Trajectory

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

app = typer.Typer(
    name="replay",
    help="Record an LLM agent's run into a cassette and replay it as a regression test.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console(highlight=False)

_STATUS_STYLE = {
    StepStatus.LIVE: "cyan",
    StepStatus.EXACT: "green",
    StepStatus.APPROXIMATE: "yellow",
    StepStatus.UNRECORDED: "bold red",
}


class Mode(StrEnum):
    full = "full"


class SeqMode(StrEnum):
    exact = "exact"
    in_order = "in_order"


def _fail(message: str) -> typer.Exit:
    console.print(f"[bold red]error:[/] {escape(message)}")
    return typer.Exit(EXIT_USAGE)


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
    if spec == "anthropic":
        import anthropic  # noqa: PLC0415 - only needed for live recording

        try:
            return anthropic.Anthropic()
        except anthropic.AnthropicError as exc:
            raise _fail(
                f"cannot create the Anthropic client ({exc}). Set ANTHROPIC_API_KEY, or record "
                "offline with --client module:factory (for example "
                "examples.toy_agent.fake_model:make_client)."
            ) from exc
    factory = load_object(spec)
    if not callable(factory):
        raise _fail(f"--client {spec!r} is not callable; it must return a model client")
    try:
        return factory()
    except Exception as exc:
        raise _fail(f"--client {spec!r} raised {type(exc).__name__}: {exc}") from exc


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
    costs = [e.cost_usd for e in cassette.envelopes if e.cost_usd is not None]
    cost = f"${sum(costs):.4f}" if costs else "no priced model calls"
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
    mark = "[bold green]PASS[/]" if check.passed else "[bold red]FAIL[/]"
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
        Mode, typer.Option(help="Cut point: full serves model and tools from the cassette.")
    ] = Mode.full,
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
    json_output: Annotated[bool, typer.Option("--json", help="Print a JSON report.")] = False,
) -> None:
    """Replay CASSETTE against the agent's current code and run the checks."""
    try:
        cassette = Cassette.load(cassette_path)
        agent_ref = agent or cassette.header.agent
        if agent_ref is None:
            raise ReplayError("the cassette does not name its agent; pass --agent module:function")
        agent_fn = load_agent(agent_ref)
        result = replay(cassette, agent_fn, mode=mode.value, task=task)
    except ReplayError as exc:
        raise _fail(str(exc)) from exc

    expected = (
        [name.strip() for name in expect.split(",") if name.strip()]
        if expect is not None
        else cassette.tool_names()
    )
    seq_mode: SequenceMode = sequence_mode.value
    checks = result.check(tool_sequence(expected, mode=seq_mode))
    approx = result.trajectory.count(StepStatus.APPROXIMATE)
    passed = result.ok and all(c.passed for c in checks) and not (strict and approx)

    if json_output:
        print(json.dumps(_report(cassette_path, agent_ref, result, checks, passed), indent=2))
        raise typer.Exit(EXIT_OK if passed else EXIT_FAILED)

    console.print(
        f"Replaying [bold]{escape(str(cassette_path))}[/] (mode {mode.value}) "
        f"against [bold]{escape(agent_ref)}[/]"
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
    console.print("[bold green]passed[/]" if passed else "[bold red]failed[/]")
    raise typer.Exit(EXIT_OK if passed else EXIT_FAILED)


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


def main() -> None:
    app()
