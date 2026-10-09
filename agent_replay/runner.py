"""Run an agent under a recording or replay session.

An *agent* here is any callable ``agent(client, task) -> output``: it receives a
model client (already wrapped, so its calls are intercepted) and the task text.
Its tools must be ``@tool``-decorated so the session sees them.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay._json import to_jsonable
from agent_replay.cassette import Cassette, Envelope, ErrorInfo
from agent_replay.checks.base import Check, CheckResult
from agent_replay.errors import AgentLoadError, ReplayError
from agent_replay.record.anthropic import wrap
from agent_replay.session import RecordingSession, ReplaySession
from agent_replay.trajectory import StepStatus, Trajectory

log = structlog.get_logger("agent_replay")

Agent = Callable[[Any, str], object]
ReplayMode = Literal["full"]


class RunResult(BaseModel):
    """Outcome of one recorded or replayed run."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["record", "full"]
    task: str
    output: JsonValue = None
    error: ErrorInfo | None = None
    trajectory: Trajectory
    unconsumed: tuple[Envelope, ...] = ()
    """Recorded envelopes the replayed run never asked for."""

    @property
    def ok(self) -> bool:
        return self.error is None

    def check(self, *checks: Check) -> list[CheckResult]:
        return [c(self) for c in checks]


def load_object(ref: str) -> object:
    """Import ``"package.module:attribute"``; the working directory is importable."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise AgentLoadError(f"{ref!r} is not a reference; use the form 'package.module:function'")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AgentLoadError(
            f"cannot import {module_name!r} for {ref!r}: {exc}. Run from the project root "
            "or install the package that contains the agent."
        ) from exc
    except Exception as exc:  # the module itself is broken: SyntaxError, error at import
        raise AgentLoadError(
            f"importing {module_name!r} for {ref!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    obj: object = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise AgentLoadError(f"{module_name!r} has no attribute {attr!r}") from exc
    return obj


def load_agent(ref: str) -> Agent:
    obj = load_object(ref)
    if not callable(obj):
        raise AgentLoadError(f"{ref!r} is not callable; an agent is a function (client, task)")
    return obj


class _OfflineClient:
    """Stands in for the model client in ``full`` mode, where nothing may go live."""

    class _Messages:
        def create(self, **_: object) -> object:
            raise ReplayError("full mode tried to reach the model; this is a bug in agent-replay")

    def __init__(self) -> None:
        self.messages = self._Messages()


def record(
    agent: Agent,
    task: str,
    *,
    client: object,
    agent_ref: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> tuple[RunResult, Cassette]:
    """Run ``agent`` for real, recording every model and tool call into a cassette.

    Raises :class:`TypeError` before anything runs if ``client`` cannot be wrapped.
    """
    model_client = wrap(client)
    session = RecordingSession(agent=agent_ref, task=task, metadata=metadata)
    output: object = None
    failure: Exception | None = None
    log.info("record.start", agent=agent_ref, task=task)
    with session:
        try:
            output = agent(model_client, task)
        except Exception as exc:
            failure = exc
            log.warning("record.agent_error", error=repr(exc))
    cassette = session.finish(output=output, error=failure)
    result = RunResult(
        mode="record",
        task=task,
        output=cassette.footer.output,
        error=cassette.footer.error,
        trajectory=session.trajectory,
    )
    log.info("record.done", envelopes=len(cassette.envelopes), status=cassette.footer.status)
    return result, cassette


def replay(
    cassette: Cassette,
    agent: Agent,
    *,
    mode: ReplayMode = "full",
    task: str | None = None,
) -> RunResult:
    """Re-run ``agent`` with every model and tool call served from ``cassette``.

    ``full`` mode never contacts a model or runs a tool, so it is offline and
    bit-stable: it tests the agent's own orchestration code against a frozen
    recording. The agent's exceptions (including :class:`UnrecordedCallError`
    when it makes a call the recording never saw) end the run and are reported
    in ``RunResult.error`` rather than raised.
    """
    if mode != "full":
        raise ReplayError(f"replay mode {mode!r} is not available yet; use mode='full'")
    task = task if task is not None else cassette.header.task
    if task is None:
        raise ReplayError(
            "the cassette has no task in its header; pass it explicitly "
            "(replay(..., task=...) or `replay run --task`)"
        )
    session = ReplaySession(cassette)
    output: object = None
    error: ErrorInfo | None = None
    log.info("replay.start", mode=mode, cassette=cassette.header.cassette_id)
    with session:
        try:
            output = agent(wrap(_OfflineClient()), task)
        except Exception as exc:
            error = ErrorInfo.from_exception(exc)
            log.warning("replay.agent_error", error=repr(exc))
    try:
        output_json = to_jsonable(output)
    except TypeError as exc:
        output_json = None
        error = error or ErrorInfo(type="TypeError", message=f"agent output is not JSON: {exc}")
    if session.fatal is not None and (error is None or error.type != type(session.fatal).__name__):
        # The agent swallowed (or masked) a replay failure; it still decides the run.
        error = ErrorInfo.from_exception(session.fatal)
    trajectory = session.trajectory
    log.info(
        "replay.done",
        steps=len(trajectory.steps),
        approximate=trajectory.count(StepStatus.APPROXIMATE),
        error=error.type if error else None,
    )
    return RunResult(
        mode=mode,
        task=task,
        output=output_json,
        error=error,
        trajectory=trajectory,
        unconsumed=tuple(session.unconsumed()),
    )
