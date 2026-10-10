"""Recording and replay sessions.

Wrapped clients and ``@tool`` functions look up the active session and hand it
a :class:`Call`. With no session active they call straight through, so
instrumented agent code runs unchanged in production.

The session is found through a context variable, so asyncio tasks (which copy
the context) see it and nest their spans correctly. Worker threads start with
an empty context (``ThreadPoolExecutor`` does not copy it), so when the variable
is unset and exactly one session is active in the process, that session is
used: tools run from a thread pool are still recorded and served, though as
top-level calls. :class:`agent_replay.ContextThreadPoolExecutor` copies the
context into its workers, which keeps their spans nested too.
"""

from __future__ import annotations

import platform
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Literal, Self

import structlog
from pydantic import JsonValue

from agent_replay._json import canonical_json, to_jsonable
from agent_replay.cassette import (
    Cassette,
    Envelope,
    ErrorInfo,
    Footer,
    Header,
    Kind,
    Usage,
)
from agent_replay.errors import (
    ReplayError,
    UnattributedCallError,
    UnrecordedCallError,
    rebuild_exception,
)
from agent_replay.pricing import cost_usd
from agent_replay.trajectory import Step, StepStatus, Trajectory

log = structlog.get_logger("agent_replay")

_current: ContextVar[Session | None] = ContextVar("agent_replay_session", default=None)
_parent_span: ContextVar[str | None] = ContextVar("agent_replay_parent_span", default=None)
_active: list[Session] = []
_active_lock = threading.Lock()


def current_session() -> Session | None:
    """The session intercepting calls here: this context's, else the only active one.

    A call from a context with no session (a plain thread-pool worker) while
    several sessions are active cannot be attributed. Passing it through would
    run it live inside a replay, so instead every active session is failed and
    the call raises.
    """
    session = _current.get()
    if session is not None:
        return session
    with _active_lock:
        active = list(_active)
    if len(active) <= 1:
        return active[0] if active else None
    error = UnattributedCallError(
        f"a call from a thread with no session context while {len(active)} sessions are "
        "active cannot be attributed to one of them; submit work through "
        "agent_replay.ContextThreadPoolExecutor (or contextvars.copy_context().run), or "
        "run one session at a time (replay flake --concurrency 1)"
    )
    for candidate in active:
        candidate.fail(error)
    raise error


def fail_current(error: ReplayError) -> ReplayError:
    """Mark ``error`` fatal on the active session, if any, and return it for raising."""
    session = current_session()
    return session.fail(error) if session is not None else error


type Mode = Literal["record", "full", "model-live", "tools-live"]
type ReplayMode = Literal["full", "model-live", "tools-live"]
type Policy = Literal["strict", "fallback", "synth"]

REPLAY_MODES: tuple[ReplayMode, ...] = ("full", "model-live", "tools-live")
POLICIES: tuple[Policy, ...] = ("strict", "fallback", "synth")

_WORLD = frozenset({Kind.CLOCK, Kind.RANDOM})
SERVED_KINDS: dict[Mode, frozenset[Kind]] = {
    "record": frozenset(),
    "full": frozenset(Kind),
    "model-live": frozenset({Kind.TOOL}) | _WORLD,
    "tools-live": frozenset({Kind.MODEL}) | _WORLD,
}
"""The cut points: which kinds of call each mode serves from the cassette.

Everything else runs live. Clock reads and random draws are always served on
replay, so live code sees the recorded time and the recorded draws.
"""


@dataclass(frozen=True, kw_only=True)
class Call[T]:
    """One boundary call, described so any session can record, serve or perform it."""

    kind: Kind
    name: str
    request: JsonValue
    fingerprint: str
    invoke: Callable[[], T] | None = None
    """Performs the real call (synchronous interception)."""
    ainvoke: Callable[[], Awaitable[T]] | None = None
    """Performs the real call (``await session.intercept_async(call)``)."""
    encode: Callable[[T], JsonValue]
    """Turns the real result into JSON for the cassette."""
    decode: Callable[[JsonValue], T]
    """Rebuilds the result from cassette JSON on replay."""
    usage: Callable[[T], Usage | None] = field(default=lambda _: None)
    model_of: Callable[[T], str | None] = field(default=lambda _: None)
    """The model that actually served the call (priced in :mod:`agent_replay.pricing`)."""
    synth: Callable[[], JsonValue] | None = None
    """A schema-valid stand-in response for the ``synth`` policy, when one can be built."""


@dataclass(frozen=True)
class _Plan:
    """How a session decided to handle one call, made under its lock."""

    index: int
    span: str
    parent: str | None
    status: StepStatus
    envelope: Envelope | None = None
    """The recorded envelope serving the call (``exact`` / ``approximate``)."""


class Session:
    """Intercepts boundary calls; installs itself as the current session inside ``with``.

    One engine covers recording and every replay mode. ``SERVED_KINDS[mode]``
    says which kinds of call are answered from the cassette; the rest run for
    real. Every call, served or live, gets a span (in starting order), a step
    in the trajectory and an envelope in the session's own output, so any run
    (a ``model-live`` replay included) can be saved as a cassette with
    :meth:`finish`.

    Matching a served call: among the recorded envelopes of the same kind and
    name that are still unconsumed, the first (in starting order) with the same
    request fingerprint is served and the step is ``exact``; if none matches,
    the oldest one *at the same nesting level* (top-level, or made inside
    another call) is served and the step is ``approximate``. A nested call
    never takes a top-level recording as an approximation, or the reverse: a
    tool that runs live under ``fallback`` would otherwise answer its inner
    calls with data recorded for unrelated top-level ones. A call with
    nothing left to serve follows the policy: ``strict`` fails the run,
    ``fallback`` performs it for real, ``synth`` answers a tool call with a stub
    built from its return type (clock and random calls run for real; a model
    response cannot be invented, so it fails as under ``strict``).

    A recorded call nested inside another call (a tool calling a tool) is only
    asked for when its parent runs for real, so it is servable only if every
    ancestor's kind runs live in this mode; otherwise an ancestor is served
    whole and the nested call is never made.
    """

    def __init__(
        self,
        *,
        mode: Mode = "record",
        cassette: Cassette | None = None,
        policy: Policy = "strict",
        agent: str | None = None,
        task: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if mode != "record" and cassette is None:
            raise ReplayError(f"mode {mode!r} replays a cassette; pass one")
        if policy not in POLICIES:
            raise ReplayError(f"unknown policy {policy!r}; use one of {', '.join(POLICIES)}")
        self.mode = mode
        self.policy: Policy = policy
        self.cassette = cassette
        self._served = SERVED_KINDS[mode]
        self._steps: list[Step] = []
        self._token: Token[Session | None] | None = None
        self._lock = threading.RLock()
        self.fatal: ReplayError | None = None
        """The first error that invalidates the run, even if agent code swallowed it."""
        self._header = Header(
            cassette_id=uuid.uuid4().hex,
            created_at=clock(),
            agent=agent,
            task=task,
            recorder=_recorder_info(),
            metadata=metadata or {},
        )
        self._timer = timer
        self._envelopes: list[Envelope] = []
        self._spans = 0
        self._queues = _servable(cassette, self._served) if cassette is not None else {}

    def __enter__(self) -> Self:
        if _current.get() is not None:
            raise ReplayError("a recording or replay session is already active in this context")
        self._token = _current.set(self)
        with _active_lock:
            _active.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        with _active_lock:
            if self in _active:
                _active.remove(self)
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    def fail[E: ReplayError](self, error: E) -> E:
        """Remember ``error`` as fatal and return it for raising.

        Agents commonly wrap tool calls in ``except Exception`` to report tool
        failures to the model, which would swallow an unrecorded call and let the
        run carry on. The runner checks ``fatal`` afterwards, so it still fails.
        """
        with self._lock:
            if self.fatal is None:
                self.fatal = error
        return error

    @property
    def trajectory(self) -> Trajectory:
        with self._lock:
            return Trajectory(steps=tuple(self._steps))

    # --- interception ---------------------------------------------------------

    def intercept[T](self, call: Call[T]) -> T:
        """Record, serve or perform ``call`` and return its result."""
        plan = self._plan(call)
        if plan.envelope is not None:
            return self._serve(call, plan, plan.envelope)
        if plan.status is StepStatus.SYNTH:
            return self._synthesize(call, plan)
        if call.invoke is None:  # pragma: no cover - adapters always set the matching invoke
            raise ReplayError(f"{call.name!r} is asynchronous; use intercept_async")
        token = _parent_span.set(plan.span)
        started = self._timer()
        try:
            result = call.invoke()
        except BaseException as exc:
            # BaseException too: a cancelled task or Ctrl-C must still complete the
            # reserved step and write an envelope, or the cassette cannot replay.
            self._record_error(call, plan, started, exc)
            raise
        finally:
            _parent_span.reset(token)
        return self._record_result(call, plan, started, result)

    async def intercept_async[T](self, call: Call[T]) -> T:
        """:meth:`intercept` for coroutines: async clients and ``async def`` tools."""
        plan = self._plan(call)
        if plan.envelope is not None:
            return self._serve(call, plan, plan.envelope)
        if plan.status is StepStatus.SYNTH:
            return self._synthesize(call, plan)
        if call.ainvoke is None:  # pragma: no cover - adapters always set the matching invoke
            raise ReplayError(f"{call.name!r} is synchronous; use intercept")
        token = _parent_span.set(plan.span)
        started = self._timer()
        try:
            result = await call.ainvoke()
        except BaseException as exc:
            # BaseException too: a cancelled task or Ctrl-C must still complete the
            # reserved step and write an envelope, or the cassette cannot replay.
            self._record_error(call, plan, started, exc)
            raise
        finally:
            _parent_span.reset(token)
        return self._record_result(call, plan, started, result)

    def _plan[T](self, call: Call[T]) -> _Plan:
        with self._lock:
            span = f"s{self._spans}"
            self._spans += 1
            envelope: Envelope | None = None
            if call.kind not in self._served:
                status = StepStatus.LIVE
            else:
                envelope = self._take(call, nested=_parent_span.get() is not None)
                if envelope is not None:
                    exact = envelope.fingerprint == call.fingerprint
                    status = StepStatus.EXACT if exact else StepStatus.APPROXIMATE
                else:
                    status = self._unrecorded_status(call)
            plan = _Plan(len(self._steps), span, _parent_span.get(), status, envelope)
            # Reserve the step now, in starting order; live calls complete it when they end.
            self._steps.append(_step(call, plan, envelope))
        if status is StepStatus.UNRECORDED:
            raise self.fail(
                UnrecordedCallError(
                    call.kind.value, call.name, _preview(call.request), hint=self._hint(call)
                )
            )
        if envelope is not None:
            log.debug("replay.step", seq=envelope.seq, name=call.name, status=status.value)
        return plan

    def _take[T](self, call: Call[T], *, nested: bool) -> Envelope | None:
        """Pop the envelope serving ``call``: an exact match, else the oldest at its level."""
        queue = self._queues.get((call.kind, call.name))
        if not queue:
            return None

        def same_level(envelope: Envelope) -> bool:
            return (envelope.parent_span_id is not None) == nested

        def exact(envelope: Envelope) -> bool:
            return envelope.fingerprint == call.fingerprint

        for accept in (
            lambda e: exact(e) and same_level(e),
            exact,
            same_level,
        ):
            for i, envelope in enumerate(queue):
                if accept(envelope):
                    return queue.pop(i)
        return None

    def _unrecorded_status[T](self, call: Call[T]) -> StepStatus:
        if self.policy == "fallback":
            return StepStatus.FALLBACK
        if self.policy == "synth":
            if call.synth is not None:
                return StepStatus.SYNTH
            if call.kind in _WORLD:
                return StepStatus.FALLBACK
        return StepStatus.UNRECORDED

    def _hint[T](self, call: Call[T]) -> str | None:
        if self.policy == "synth" and call.kind is Kind.MODEL:
            return (
                "the synth policy cannot invent a model response; use --policy fallback with "
                "a live --client, or re-record"
            )
        return None

    def _serve[T](self, call: Call[T], plan: _Plan, envelope: Envelope) -> T:
        self._emit(
            call,
            plan,
            response=envelope.response,
            error=envelope.error,
            usage=envelope.usage,
            cost=envelope.cost_usd,
            latency_ms=envelope.latency_ms,
        )
        if envelope.error is not None:
            raise rebuild_exception(
                envelope.error.type, envelope.error.module, envelope.error.message
            )
        return self._decode(call, envelope.response)

    def _synthesize[T](self, call: Call[T], plan: _Plan) -> T:
        assert call.synth is not None
        try:
            response = call.synth()
        except ReplayError as exc:
            raise self.fail(exc) from exc
        except Exception as exc:  # a schema the stub builder never anticipated
            raise self.fail(
                ReplayError(
                    f"cannot synthesise a result for {call.kind.value} call {call.name!r} "
                    f"({type(exc).__name__}: {exc}); record the call or use --policy fallback"
                )
            ) from exc
        seq = self._emit(call, plan, response=response, error=None, latency_ms=0.0)
        self._complete(call, plan, seq, response=response, latency_ms=0.0)
        try:
            return call.decode(response)
        except Exception as exc:
            raise self.fail(
                ReplayError(
                    f"the synthesised stub for {call.kind.value} call {call.name!r} does not "
                    f"validate against its return type ({exc}); record the call or use "
                    "--policy fallback"
                )
            ) from exc

    def _decode[T](self, call: Call[T], response: JsonValue) -> T:
        try:
            return call.decode(response)
        except Exception as exc:
            # The recording is fine; the code changed what this call returns. Like
            # an unrecorded call, this decides the run even if the agent swallows it.
            raise self.fail(
                ReplayError(
                    f"the recorded response of {call.kind.value} call {call.name!r} no longer "
                    f"fits its return type ({exc}); re-record the cassette with `replay record`"
                )
            ) from exc

    def _record_error[T](
        self, call: Call[T], plan: _Plan, started: float, exc: BaseException
    ) -> None:
        latency_ms = round((self._timer() - started) * 1000, 3)
        error = ErrorInfo.from_exception(exc)
        seq = self._emit(call, plan, response=None, error=error, latency_ms=latency_ms)
        self._complete(call, plan, seq, error=error, latency_ms=latency_ms)

    def _record_result[T](self, call: Call[T], plan: _Plan, started: float, result: T) -> T:
        latency_ms = round((self._timer() - started) * 1000, 3)
        try:
            response = call.encode(result)
        except TypeError as exc:
            raise self.fail(
                ReplayError(
                    f"{call.kind.value} call {call.name!r} returned a value that cannot be "
                    f"recorded as JSON ({exc}); return JSON-compatible data or a pydantic model"
                )
            ) from exc
        usage = call.usage(result)
        cost = cost_usd(call.model_of(result), usage) if usage is not None else None
        seq = self._emit(
            call, plan, response=response, error=None, usage=usage, cost=cost, latency_ms=latency_ms
        )
        self._complete(
            call, plan, seq, response=response, usage=usage, cost=cost, latency_ms=latency_ms
        )
        return result

    def _emit[T](
        self,
        call: Call[T],
        plan: _Plan,
        *,
        response: JsonValue,
        error: ErrorInfo | None,
        latency_ms: float,
        usage: Usage | None = None,
        cost: float | None = None,
    ) -> int:
        """Append this call's envelope to the session's own output; return its seq."""
        with self._lock:
            seq = len(self._envelopes)
            self._envelopes.append(
                Envelope(
                    seq=seq,
                    kind=call.kind,
                    name=call.name,
                    fingerprint=call.fingerprint,
                    request=call.request,
                    response=response,
                    error=error,
                    usage=usage,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    span_id=plan.span,
                    parent_span_id=plan.parent,
                )
            )
        log.debug("envelope.recorded", seq=seq, kind=call.kind.value, name=call.name)
        return seq

    def _complete[T](
        self,
        call: Call[T],
        plan: _Plan,
        seq: int,
        *,
        response: JsonValue = None,
        error: ErrorInfo | None = None,
        usage: Usage | None = None,
        cost: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self._lock:
            self._steps[plan.index] = self._steps[plan.index].model_copy(
                update={
                    "response": response,
                    "error": error,
                    # In a recording the step points at the envelope it wrote; in a
                    # replay only served steps point anywhere (into the source cassette).
                    "envelope_seq": seq if self.mode == "record" else None,
                    "usage": usage,
                    "cost_usd": cost,
                    "latency_ms": latency_ms,
                }
            )

    # --- results ----------------------------------------------------------------

    def unconsumed(self) -> list[Envelope]:
        """Servable recorded envelopes the run never asked for, in recorded order."""
        with self._lock:
            left = [e for queue in self._queues.values() for e in queue]
        return sorted(left, key=lambda e: e.seq)

    def finish(self, *, output: object = None, error: BaseException | None = None) -> Cassette:
        """Close the session and return what it saw as a cassette (not written to disk here).

        For a recording that is the cassette; for a replay it is a recording of
        the replayed run, served calls included, which replays like any other.
        """
        if error is None and self.fatal is not None:
            error = self.fatal
        try:
            output_json = to_jsonable(output)
        except TypeError as exc:
            output_json = None
            error = error or ReplayError(
                f"the agent's return value is not JSON-serialisable: {exc}"
            )
        with self._lock:
            footer = Footer(
                status="error" if error is not None else "ok",
                output=output_json,
                error=ErrorInfo.from_exception(error) if error is not None else None,
                envelope_count=len(self._envelopes),
            )
            return Cassette(header=self._header, envelopes=tuple(self._envelopes), footer=footer)


def _recorder_info() -> dict[str, str]:
    try:
        pkg = version("agent-replay")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        pkg = "unknown"
    return {"agent_replay": pkg, "python": platform.python_version()}


def _servable(
    cassette: Cassette, served: frozenset[Kind]
) -> dict[tuple[Kind, str], list[Envelope]]:
    """Per (kind, name) queues of the envelopes this mode may serve, in starting order."""
    by_span = {e.span_id: e for e in cassette.envelopes}

    def reachable(envelope: Envelope) -> bool:
        # Asked for on replay only if every ancestor runs live (a served parent's
        # body never runs, so its children are never asked for).
        parent_span = envelope.parent_span_id
        while parent_span is not None:
            parent = by_span.get(parent_span)
            if parent is None or parent.kind in served:
                return False
            parent_span = parent.parent_span_id
        return True

    queues: dict[tuple[Kind, str], list[Envelope]] = {}
    for envelope in cassette.in_start_order():
        if envelope.kind in served and reachable(envelope):
            queues.setdefault((envelope.kind, envelope.name), []).append(envelope)
    return queues


class RecordingSession(Session):
    """A :class:`Session` in ``record`` mode: every call runs for real.

    As in every mode, an envelope is appended when its call finishes (``seq`` is
    the finishing order), each call gets a ``span_id`` in starting order and, when
    it ran inside another intercepted call in the same context, that call's span
    as ``parent_span_id``. The trajectory lists steps in starting order, the same
    order a replay sees them in, so parallel tools that finished in a different
    order still compare equal.
    """

    def __init__(
        self,
        *,
        agent: str | None = None,
        task: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(
            mode="record", agent=agent, task=task, metadata=metadata, clock=clock, timer=timer
        )


class ReplaySession(Session):
    """Replays ``cassette`` with the cut point ``mode`` and the unrecorded-call ``policy``.

    ``full`` serves every call (offline, bit-stable), ``model-live`` runs the
    model for real against recorded tools, ``tools-live`` runs the tools for
    real against recorded model replies. See :class:`Session` for matching.
    """

    def __init__(
        self,
        cassette: Cassette,
        *,
        mode: ReplayMode = "full",
        policy: Policy = "strict",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if mode not in REPLAY_MODES:
            raise ReplayError(f"unknown replay mode {mode!r}; use one of {', '.join(REPLAY_MODES)}")
        super().__init__(
            mode=mode,
            cassette=cassette,
            policy=policy,
            agent=cassette.header.agent,
            task=cassette.header.task,
            metadata={
                "replay_of": cassette.header.cassette_id,
                "mode": mode,
                "policy": policy,
            },
            clock=clock,
            timer=timer,
        )


def _step[T](call: Call[T], plan: _Plan, envelope: Envelope | None) -> Step:
    return Step(
        index=plan.index,
        kind=call.kind,
        name=call.name,
        status=plan.status,
        fingerprint=call.fingerprint,
        request=call.request,
        response=envelope.response if envelope else None,
        error=envelope.error if envelope else None,
        envelope_seq=envelope.seq if envelope else None,
        usage=envelope.usage if envelope else None,
        cost_usd=envelope.cost_usd if envelope else None,
        latency_ms=envelope.latency_ms if envelope else None,
        span_id=plan.span,
        parent_span_id=plan.parent,
    )


def _preview(request: JsonValue, limit: int = 200) -> str:
    text = canonical_json(request)
    return text if len(text) <= limit else text[: limit - 3] + "..."
