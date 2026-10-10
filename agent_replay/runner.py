"""Run an agent under a recording or replay session.

An *agent* here is any callable ``agent(client, task) -> output``, or an
``async def`` with the same signature: it receives a model client (already
wrapped, so its calls are intercepted) and the task text. Its tools must be
``@tool``-decorated so the session sees them.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay._json import to_jsonable
from agent_replay._loading import load_object
from agent_replay.cassette import Cassette, Envelope, ErrorInfo, Kind
from agent_replay.checks.base import Check, CheckResult
from agent_replay.errors import AgentLoadError, ReplayError
from agent_replay.record import adapter_for_calls, wrap
from agent_replay.session import (
    REPLAY_MODES,
    Policy,
    RecordingSession,
    ReplayMode,
    ReplaySession,
    Session,
)
from agent_replay.trajectory import StepStatus, Trajectory

log = structlog.get_logger("agent_replay")

Agent = Callable[[Any, str], object]


class RunResult(BaseModel):
    """Outcome of one recorded or replayed run."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["record", "full", "model-live", "tools-live"]
    policy: Policy = "strict"
    task: str
    cassette_id: str | None = None
    """The recording this run replayed (or, for a recording, the one it wrote)."""
    output: JsonValue = None
    error: ErrorInfo | None = None
    trajectory: Trajectory
    unconsumed: tuple[Envelope, ...] = ()
    """Recorded envelopes the replayed run never asked for."""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def cost_usd(self) -> float:
        """Total priced model cost of the run (served steps count at their recorded cost)."""
        return round(sum(s.cost_usd or 0.0 for s in self.trajectory.steps), 8)

    @property
    def tokens(self) -> int:
        """Input plus output tokens over every model call (cache tokens included)."""
        return sum(
            u.input_tokens
            + u.output_tokens
            + u.cache_creation_input_tokens
            + u.cache_read_input_tokens
            for s in self.trajectory.steps
            if (u := s.usage) is not None
        )

    def check(self, *checks: Check) -> list[CheckResult]:
        return [c(self) for c in checks]


def make_client(ref: str, *, seed: int | None = None) -> object:
    """A model client from ``"anthropic"`` (the real API) or a ``module:factory`` reference.

    A factory with a ``seed`` parameter is called with ``seed=`` when one is
    given (``replay flake`` passes one per run), so a simulated flaky model is
    reproducible; other factories are called without arguments.
    """
    if ref == "anthropic":
        import anthropic  # noqa: PLC0415 - only needed for live calls

        try:
            return anthropic.Anthropic()
        except anthropic.AnthropicError as exc:
            raise AgentLoadError(
                f"cannot create the Anthropic client ({exc}). Set ANTHROPIC_API_KEY, or use a "
                "fake model with module:factory (for example "
                "examples.toy_agent.fake_model:make_client)."
            ) from exc
    factory = load_object(ref)
    if not callable(factory):
        raise AgentLoadError(f"client {ref!r} is not callable; it must return a model client")
    try:
        if seed is not None and _takes_seed(factory):
            return factory(seed=seed)
        return factory()
    except Exception as exc:
        raise AgentLoadError(f"client {ref!r} raised {type(exc).__name__}: {exc}") from exc


def _takes_seed(factory: Callable[..., object]) -> bool:
    """Whether ``factory`` accepts ``seed=`` by keyword."""
    try:
        parameter = inspect.signature(factory).parameters.get("seed")
    except (TypeError, ValueError):  # builtins and C callables may have no signature
        return False
    return parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY


def load_agent(ref: str) -> Agent:
    obj = load_object(ref)
    if not callable(obj):
        raise AgentLoadError(f"{ref!r} is not callable; an agent is a function (client, task)")
    return obj


def _run_agent(agent: Agent, client: object, task: str) -> object:
    """Call the agent; an ``async def`` agent runs to completion on a fresh event loop."""
    if inspect.iscoroutinefunction(agent):
        return asyncio.run(agent(client, task))
    output = agent(client, task)
    if inspect.isawaitable(output):
        return asyncio.run(_await(output))
    return output


async def _await(awaitable: Any) -> object:
    return await awaitable


def _run(
    session: Session, agent: Agent, client: object, task: str
) -> tuple[object, Exception | None]:
    with session:
        try:
            return _run_agent(agent, client, task), None
        except Exception as exc:
            log.warning("run.agent_error", mode=session.mode, error=repr(exc))
            return None, exc


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
    log.info("record.start", agent=agent_ref, task=task)
    output, failure = _run(session, agent, model_client, task)
    cassette = session.finish(output=output, error=failure)
    result = RunResult(
        mode="record",
        task=task,
        cassette_id=cassette.header.cassette_id,
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
    policy: Policy = "strict",
    task: str | None = None,
    client: object = None,
    save_to: str | Path | None = None,
) -> RunResult:
    """Re-run ``agent`` against ``cassette`` with the cut point ``mode``.

    - ``full``: every model and tool call is served from the cassette. Offline
      and bit-stable; tests the agent's own orchestration code.
    - ``model-live``: the model runs for real (``client`` is required), tools
      are served from the cassette. Tests prompt and model changes against a
      frozen world.
    - ``tools-live``: tools run for real against recorded model replies. Tests
      tool code changes without spending on the model.

    ``policy`` decides what happens to a call the cassette has no envelope for
    (``strict`` fails, ``fallback`` performs it, ``synth`` stubs a tool result;
    see :class:`agent_replay.session.Session`). In modes that serve the model,
    ``client`` is only used by ``fallback``. The agent's exceptions end the
    run and are reported in ``RunResult.error`` rather than raised.
    ``save_to`` writes the replayed run as a cassette of its own.
    """
    if mode not in REPLAY_MODES:
        raise ReplayError(f"unknown replay mode {mode!r}; use one of {', '.join(REPLAY_MODES)}")
    task = task if task is not None else cassette.header.task
    if task is None:
        raise ReplayError(
            "the cassette has no task in its header; pass it explicitly "
            "(replay(..., task=...) or `replay run --task`)"
        )
    if client is None:
        if mode == "model-live":
            raise ReplayError(
                "model-live mode calls the model for real, so it needs a client; pass "
                "client=... (or --client anthropic / --client module:factory)"
            )
        model_names = [e.name for e in cassette.envelopes if e.kind is Kind.MODEL]
        client = adapter_for_calls(model_names).offline_client(
            is_async=inspect.iscoroutinefunction(agent)
        )
    try:
        model_client = wrap(client)
    except TypeError as exc:
        raise AgentLoadError(f"{exc}; --client must return a model client") from exc
    session = ReplaySession(cassette, mode=mode, policy=policy)
    log.info("replay.start", mode=mode, policy=policy, cassette=cassette.header.cassette_id)
    output, failure = _run(session, agent, model_client, task)
    error = ErrorInfo.from_exception(failure) if failure is not None else None
    try:
        output_json = to_jsonable(output)
    except TypeError as exc:
        output_json = None
        error = error or ErrorInfo(type="TypeError", message=f"agent output is not JSON: {exc}")
    if session.fatal is not None and (error is None or error.type != type(session.fatal).__name__):
        # The agent swallowed (or masked) a replay failure; it still decides the run.
        error = ErrorInfo.from_exception(session.fatal)
    if save_to is not None:
        session.finish(output=output, error=failure).dump(save_to)
    trajectory = session.trajectory
    log.info(
        "replay.done",
        steps=len(trajectory.steps),
        approximate=trajectory.count(StepStatus.APPROXIMATE),
        error=error.type if error else None,
    )
    return RunResult(
        mode=mode,
        policy=policy,
        task=task,
        cassette_id=cassette.header.cassette_id,
        output=output_json,
        error=error,
        trajectory=trajectory,
        unconsumed=tuple(session.unconsumed()),
    )
