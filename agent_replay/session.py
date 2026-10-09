"""Recording and replay sessions.

Wrapped clients and ``@tool`` functions look up the active session and hand it
a :class:`Call`. With no session active they call straight through, so
instrumented agent code runs unchanged in production.

The session is found through a context variable. Worker threads start with an
empty context (``ThreadPoolExecutor`` does not copy it), so when the variable is
unset and exactly one session is active in the process, that session is used:
tools run from a thread pool are still recorded and still served on replay.
"""

from __future__ import annotations

import platform
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Self

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
from agent_replay.errors import ReplayError, UnrecordedCallError, rebuild_exception
from agent_replay.pricing import cost_usd
from agent_replay.trajectory import Step, StepStatus, Trajectory

log = structlog.get_logger("agent_replay")

_current: ContextVar[Session | None] = ContextVar("agent_replay_session", default=None)
_parent_span: ContextVar[str | None] = ContextVar("agent_replay_parent_span", default=None)
_active: list[Session] = []
_active_lock = threading.Lock()


def current_session() -> Session | None:
    """The session intercepting calls here: this context's, else the only active one."""
    session = _current.get()
    if session is not None:
        return session
    with _active_lock:
        return _active[0] if len(_active) == 1 else None


@dataclass(frozen=True, kw_only=True)
class Call[T]:
    """One boundary call, described so any session can record or serve it."""

    kind: Kind
    name: str
    request: JsonValue
    fingerprint: str
    invoke: Callable[[], T]
    """Performs the real call."""
    encode: Callable[[T], JsonValue]
    """Turns the real result into JSON for the cassette."""
    decode: Callable[[JsonValue], T]
    """Rebuilds the result from cassette JSON on replay."""
    usage: Callable[[T], Usage | None] = field(default=lambda _: None)
    model_of: Callable[[T], str | None] = field(default=lambda _: None)
    """The model that actually served the call (priced in :mod:`agent_replay.pricing`)."""


class Session(ABC):
    """Base class: installs itself as the current session inside ``with``."""

    mode: str

    def __init__(self) -> None:
        self._steps: list[Step] = []
        self._token: Token[Session | None] | None = None
        self._lock = threading.RLock()
        self.fatal: ReplayError | None = None
        """The first error that invalidates the run, even if agent code swallowed it."""

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

    @abstractmethod
    def intercept[T](self, call: Call[T]) -> T:
        """Record or serve ``call`` and return its result."""


def _recorder_info() -> dict[str, str]:
    try:
        pkg = version("agent-replay")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        pkg = "unknown"
    return {"agent_replay": pkg, "python": platform.python_version()}


class RecordingSession(Session):
    """Performs every call for real and appends an envelope when each call finishes.

    ``seq`` is the finishing order. Each call also gets a ``span_id`` (in starting
    order) and, when it ran inside another intercepted call in the same context
    (a tool that calls a tool or the model), that call's span as ``parent_span_id``.
    The trajectory lists steps in starting order, the same order a replay sees
    them in, so parallel tools that finished in a different order still compare
    equal.
    """

    mode = "record"

    def __init__(
        self,
        *,
        agent: str | None = None,
        task: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__()
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

    def intercept[T](self, call: Call[T]) -> T:
        with self._lock:
            span = f"s{self._spans}"
            self._spans += 1
            # Reserve the step now, in starting order; it is completed when the call ends.
            index = len(self._steps)
            self._steps.append(_step(index, call, StepStatus.LIVE))
        parent = _parent_span.get()
        token = _parent_span.set(span)
        started = self._timer()
        try:
            result = call.invoke()
        except Exception as exc:
            latency_ms = (self._timer() - started) * 1000
            self._append(
                call,
                index,
                span,
                parent,
                latency_ms,
                response=None,
                error=ErrorInfo.from_exception(exc),
            )
            raise
        finally:
            _parent_span.reset(token)
        latency_ms = (self._timer() - started) * 1000
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
        self._append(
            call,
            index,
            span,
            parent,
            latency_ms,
            response=response,
            error=None,
            usage=usage,
            cost=cost_usd(call.model_of(result), usage) if usage is not None else None,
        )
        return result

    def _append[T](
        self,
        call: Call[T],
        index: int,
        span: str,
        parent: str | None,
        latency_ms: float,
        *,
        response: JsonValue,
        error: ErrorInfo | None,
        usage: Usage | None = None,
        cost: float | None = None,
    ) -> None:
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
                    latency_ms=round(latency_ms, 3),
                    span_id=span,
                    parent_span_id=parent,
                )
            )
            self._steps[index] = _step(
                index, call, StepStatus.LIVE, response=response, error=error, envelope_seq=seq
            )
        log.debug("envelope.recorded", seq=seq, kind=call.kind.value, name=call.name)

    def finish(self, *, output: object = None, error: BaseException | None = None) -> Cassette:
        """Close the recording and return the cassette (it is not written to disk here)."""
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


class ReplaySession(Session):
    """Serves every call from a cassette without performing it (``full`` mode).

    Matching: each (kind, name) pair has its own cursor over the recorded
    top-level envelopes in the order they started, so the n-th call to
    ``lookup_order`` gets the n-th recorded ``lookup_order`` result. The request
    fingerprint is then compared
    with the recording: equal means ``exact``, different means ``approximate``
    (served anyway, flagged in the trajectory). A call with nothing left to serve
    raises :class:`UnrecordedCallError`.

    Nested envelopes (calls made inside a recorded call) are never served: their
    parent is served whole, so its body, and the nested calls, do not run.
    """

    mode = "full"

    def __init__(self, cassette: Cassette) -> None:
        super().__init__()
        self.cassette = cassette
        self._queues: dict[tuple[Kind, str], deque[Envelope]] = {}
        for envelope in cassette.top_level(order="started"):
            self._queues.setdefault((envelope.kind, envelope.name), deque()).append(envelope)

    def intercept[T](self, call: Call[T]) -> T:
        with self._lock:
            queue = self._queues.get((call.kind, call.name))
            envelope = queue.popleft() if queue else None
            if envelope is None:
                status = StepStatus.UNRECORDED
            elif envelope.fingerprint == call.fingerprint:
                status = StepStatus.EXACT
            else:
                status = StepStatus.APPROXIMATE
            self._steps.append(
                _step(
                    len(self._steps),
                    call,
                    status,
                    response=envelope.response if envelope else None,
                    error=envelope.error if envelope else None,
                    envelope_seq=envelope.seq if envelope else None,
                )
            )
        if envelope is None:
            raise self.fail(UnrecordedCallError(call.kind.value, call.name, _preview(call.request)))
        log.debug("replay.step", seq=envelope.seq, name=call.name, status=status.value)
        if envelope.error is not None:
            raise rebuild_exception(
                envelope.error.type, envelope.error.module, envelope.error.message
            )
        return call.decode(envelope.response)

    def unconsumed(self) -> list[Envelope]:
        """Recorded top-level envelopes the replayed run never asked for, in recorded order."""
        with self._lock:
            left = [e for queue in self._queues.values() for e in queue]
        return sorted(left, key=lambda e: e.seq)


def _step[T](
    index: int,
    call: Call[T],
    status: StepStatus,
    *,
    response: JsonValue = None,
    error: ErrorInfo | None = None,
    envelope_seq: int | None = None,
) -> Step:
    return Step(
        index=index,
        kind=call.kind,
        name=call.name,
        status=status,
        fingerprint=call.fingerprint,
        request=call.request,
        response=response,
        error=error,
        envelope_seq=envelope_seq,
    )


def _preview(request: JsonValue, limit: int = 200) -> str:
    text = canonical_json(request)
    return text if len(text) <= limit else text[: limit - 3] + "..."
