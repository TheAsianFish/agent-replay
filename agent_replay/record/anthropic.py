"""Interception for the official ``anthropic`` client, sync and async.

We wrap the client object rather than its HTTP transport: ``anthropic`` 1.x runs
on ``httpx2``, so transport-level mocks written for ``httpx`` never see its
traffic, and the SDK layer hands us typed messages and usage for free.

``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` share one recording
path: the request is normalised and fingerprinted the same way, and only the
call that performs it differs (``session.intercept`` or
``await session.intercept_async``).

Streaming (``messages.create(stream=True)`` and ``messages.stream(...)``) is
recorded as the final accumulated ``Message`` under the same call name, with
``"stream": true`` in the request but not in the fingerprint (it is delivery,
not content), so a recording made with streaming replays ``exact`` against
code that no longer streams, and vice versa. While recording, the real stream
is drained before the agent sees any of it: the agent receives the stream only
once the model has finished, rebuilt from the final message by
:func:`agent_replay.record._streams.events_from_message`. Replay delivers the
identical rebuilt stream, so the agent observes the same events in both. The
trade-off: a recording run shows no incremental output, and a stream that
fails part-way fails at ``create`` / ``__enter__`` rather than mid-iteration.

``messages.parse(..., output_format=T)`` (and ``messages.stream(...,
output_format=T)``) is recorded the same way: the request holds what goes on
the wire, ``output_config.format`` with ``T``'s JSON schema, and the response
the plain ``Message``. The ``ParsedMessage`` is rebuilt from it with the SDK's
own parser on both recording and replay, so a reply that fails to validate
still has its message (and usage) recorded and fails identically on replay.

``client.beta.messages.create`` (plain and ``stream=True``) and
``client.beta.messages.stream`` are recorded the same way under their own call
name, ``anthropic.beta.messages.create``. Frameworks such as pydantic-ai call
only this surface. ``betas`` stays in the fingerprint (a beta changes what the
model does), and a replay returns the beta types (``BetaMessage`` and
``BetaRawMessageStartEvent`` and so on), since such code checks them with
``isinstance``. ``count_tokens``, batches and the beta ``parse`` are not
recorded and say so instead of going live.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, NoReturn, cast

import anthropic
from anthropic.lib._parse._response import parse_response
from anthropic.lib._parse._transform import transform_schema
from anthropic.lib.streaming import (
    AsyncMessageStream,
    BetaAsyncMessageStream,
    BetaMessageStream,
    MessageStream,
)
from anthropic.resources.messages import AsyncMessages
from anthropic.types import Message
from anthropic.types.beta import BetaMessage
from pydantic import JsonValue, PydanticSchemaGenerationError, TypeAdapter

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ReplayError
from agent_replay.record._base import WrappedClient
from agent_replay.record._streams import (
    BETA_MESSAGE_EVENTS,
    MESSAGE_EVENTS,
    AsyncSyntheticStream,
    EventTypes,
    SyntheticStream,
    events_from_message,
)
from agent_replay.session import Call, Session, current_session, fail_current

MODEL_CALL = "anthropic.messages.create"
BETA_MODEL_CALL = "anthropic.beta.messages.create"

# Transport options: never recorded (headers can carry credentials) and never
# part of the request identity.
_TRANSPORT_KWARGS = frozenset({"extra_headers", "extra_query", "timeout"})
# Recorded, but excluded from the fingerprint: ``metadata`` varies between
# otherwise identical requests, and ``stream`` changes how the response is
# delivered, not what it is.
_UNIDENTIFYING_FIELDS = frozenset({"metadata", "stream"})
# Client surfaces that would make calls this proxy cannot record yet. Inside a
# session, calling them raises instead of silently going live; merely looking
# them up (``hasattr``) is harmless.
_UNRECORDED_MESSAGES = frozenset(
    {"batches", "count_tokens", "with_raw_response", "with_streaming_response"}
)
# On ``client.beta.messages``: ``parse`` too, and ``tool_runner``, which loops over
# ``create`` and ``parse`` in ways the proxy does not see.
_UNRECORDED_BETA_MESSAGES = _UNRECORDED_MESSAGES | {"parse", "tool_runner"}
_UNRECORDED_CLIENT = frozenset({"with_raw_response", "with_streaming_response"})
_RECORDED_HINT = (
    "use client.messages.create(...), .stream(...) or .parse(...), or "
    "client.beta.messages.create(...) or .stream(...)"
)


def _not_recorded(what: str) -> NoReturn:
    session = current_session()
    assert session is not None
    hint = _RECORDED_HINT
    if what.endswith("count_tokens"):
        # pydantic-ai calls it for UsageLimits(count_tokens_before_request=True).
        hint = (
            "token counting is not supported: its result would have to be recorded and "
            "matched like a model call; leave count_tokens_before_request off, or " + hint
        )
    raise session.fail(
        ReplayError(
            f"{what} is not recorded yet, so it cannot run inside a recording or replay "
            f"session; {hint}"
        )
    )


class _Unrecorded:
    """Stands in for a client surface the proxy cannot record.

    Looking it up is harmless; calling it, or reaching through it (for example
    ``client.beta.messages.create``), fails the session via :func:`_not_recorded`.
    """

    def __init__(self, what: str) -> None:
        self._what = what

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        _not_recorded(self._what)

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("__"):
            raise AttributeError(name)
        _not_recorded(self._what)

    def __repr__(self) -> str:
        return f"<unrecorded {self._what}>"


type AnyMessage = Message | BetaMessage


@dataclass(frozen=True)
class _Surface:
    """One recorded API surface: ``client.messages`` or ``client.beta.messages``."""

    call_name: str
    what: str
    """How errors name it, e.g. ``beta.messages``."""
    message: type[Message] | type[BetaMessage]
    events: EventTypes
    sync_stream: type[MessageStream[Any]] | type[BetaMessageStream[Any]]
    async_stream: type[AsyncMessageStream[Any]] | type[BetaAsyncMessageStream[Any]]

    def encode(self, message: object) -> JsonValue:
        if not isinstance(message, self.message):
            raise TypeError(
                f"{self.what}.create returned {type(message).__name__}, not a "
                f"{self.message.__name__}; wrap the client itself, not a raw-response view of it"
            )
        return to_jsonable(message)

    def decode(self, data: JsonValue) -> AnyMessage:
        return self.message.model_validate(data)

    def rebuild(self, data: object) -> AnyMessage:
        """``data`` (an SDK message, perhaps a parsed subclass) as this surface's plain type."""
        return self.message.model_validate(to_jsonable(data))

    def events_of(self, message: AnyMessage) -> list[Any]:
        return events_from_message(message, self.events)

    def betas_of(self, kwargs: Mapping[str, Any]) -> list[str] | None:
        """The ``betas`` a rebuilt stream reports (see ``RebuiltResponse``); ``None`` off beta."""
        if self.message is not BetaMessage:
            return None
        betas = kwargs.get("betas")
        return [str(b) for b in betas] if isinstance(betas, list | tuple) else []


MESSAGES = _Surface(
    MODEL_CALL, "messages", Message, MESSAGE_EVENTS, MessageStream, AsyncMessageStream
)
BETA_MESSAGES = _Surface(
    BETA_MODEL_CALL,
    "beta.messages",
    BetaMessage,
    BETA_MESSAGE_EVENTS,
    BetaMessageStream,
    BetaAsyncMessageStream,
)


def _usage(message: AnyMessage) -> Usage:
    u = message.usage
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
        cache_read_input_tokens=u.cache_read_input_tokens or 0,
    )


def _model_of(message: AnyMessage) -> str:
    return message.model


def wire_output_format(kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Move ``output_format=SomeType`` into ``output_config.format``, as the SDK sends it.

    ``messages.parse`` and ``messages.stream`` accept a Python type and send its
    JSON schema; the cassette records what goes on the wire, so an agent that
    switches between ``output_format=Answer`` and the equivalent
    ``output_config`` still matches ``exact``. Returns the rewritten kwargs and
    the type (``NOT_GIVEN`` when there is none). Mirrors the SDK's conversion,
    including its ``TypeError`` for a schema dict.
    """
    kwargs = dict(kwargs)
    output_format = kwargs.pop("output_format", anthropic.NOT_GIVEN)
    if isinstance(output_format, anthropic.NotGiven | anthropic.Omit) or output_format is None:
        return kwargs, anthropic.NOT_GIVEN
    if isinstance(output_format, Mapping):
        raise TypeError(
            "`output_format` must be a type; pass a schema dict as "
            "`output_config={'format': ...}` instead"
        )
    try:
        schema = TypeAdapter(output_format).json_schema()
    except PydanticSchemaGenerationError as exc:
        raise TypeError(
            "Could not generate JSON schema for the given `output_format` type. "
            "Use a type that works with `pydantic.TypeAdapter`"
        ) from exc
    wire_format = {"type": "json_schema", "schema": transform_schema(schema)}
    config = kwargs.get("output_config")
    kwargs["output_config"] = (
        {**config, "format": wire_format}
        if isinstance(config, Mapping)
        else {"format": wire_format}
    )
    return kwargs, output_format


@dataclass(frozen=True)
class _Prepared:
    """One model request, normalised for the cassette."""

    kwargs: dict[str, Any]
    """What the real ``messages.create`` is called with (transport options included)."""
    request: JsonValue
    """What the cassette records: the request body without transport options."""
    fingerprint: str
    output_format: Any
    """The type ``stream``/``parse`` validate the reply's text into, or ``NOT_GIVEN``."""
    surface: _Surface


def _prepare(
    session: Session, kwargs: dict[str, Any], what: str, surface: _Surface = MESSAGES
) -> _Prepared:
    output_format: Any = anthropic.NOT_GIVEN
    if not what.endswith(".create"):  # only the helpers take output_format
        kwargs, output_format = wire_output_format(kwargs)
    # NOT_GIVEN / omit are SDK sentinels for "leave the field out"; so do we.
    given = {
        k: v
        for k, v in kwargs.items()
        if k not in _TRANSPORT_KWARGS and not isinstance(v, anthropic.NotGiven | anthropic.Omit)
    }
    try:
        request = to_jsonable(given)
    except TypeError as exc:
        raise session.fail(
            ReplayError(f"{what} arguments cannot be recorded as JSON: {exc}")
        ) from exc
    return _Prepared(
        kwargs=kwargs,
        request=request,
        fingerprint=fingerprint(
            Kind.MODEL, surface.call_name, request, ignore=_UNIDENTIFYING_FIELDS
        ),
        output_format=output_format,
        surface=surface,
    )


def _call(
    prepared: _Prepared,
    *,
    invoke: Callable[[], Any] | None = None,
    ainvoke: Callable[[], Awaitable[Any]] | None = None,
) -> Call[Any]:
    surface = prepared.surface
    return Call(
        kind=Kind.MODEL,
        name=surface.call_name,
        request=prepared.request,
        fingerprint=prepared.fingerprint,
        invoke=invoke,
        ainvoke=ainvoke,
        encode=surface.encode,
        decode=surface.decode,
        usage=_usage,
        model_of=_model_of,
    )


def _drain(raw: Any, surface: _Surface = MESSAGES) -> AnyMessage:
    """The final message of a real stream, accumulated by the SDK's own ``MessageStream``."""
    with surface.sync_stream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = stream.get_final_message()
    return surface.rebuild(final)


async def _adrain(pending: Awaitable[Any], surface: _Surface = MESSAGES) -> AnyMessage:
    """:func:`_drain` for an async client's ``create(stream=True)``."""
    raw = await pending
    async with surface.async_stream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = await stream.get_final_message()
    return surface.rebuild(final)


class _StreamManager:
    """What ``messages.stream(...)`` returns in a session; the call happens on ``__enter__``.

    As with the SDK's ``MessageStreamManager``, nothing is requested until the
    ``with`` block is entered; the block then gets the SDK's own
    ``MessageStream`` over the rebuilt events, so ``text_stream``,
    ``get_final_message()`` and ``until_done()`` behave exactly as they do live.
    """

    def __init__(
        self, request: Callable[[], AnyMessage], prepared: _Prepared, surface: _Surface
    ) -> None:
        self._request = request
        self._output_format = prepared.output_format
        self._betas = surface.betas_of(prepared.kwargs)
        self._surface = surface
        self._stream: Any = None

    def __enter__(self) -> Any:
        raw = SyntheticStream(self._surface.events_of(self._request()), self._betas)
        self._stream = self._surface.sync_stream(cast(Any, raw), output_format=self._output_format)
        return self._stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.close()


class _AsyncStreamManager:
    """:class:`_StreamManager` for ``async with client.messages.stream(...)``."""

    def __init__(
        self,
        request: Callable[[], Awaitable[AnyMessage]],
        prepared: _Prepared,
        surface: _Surface,
    ) -> None:
        self._request = request
        self._output_format = prepared.output_format
        self._betas = surface.betas_of(prepared.kwargs)
        self._surface = surface
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        raw = AsyncSyntheticStream(self._surface.events_of(await self._request()), self._betas)
        self._stream = self._surface.async_stream(cast(Any, raw), output_format=self._output_format)
        return self._stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            await self._stream.close()


class _MessagesBase:
    _surface = MESSAGES
    _unrecorded = _UNRECORDED_MESSAGES

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in self._unrecorded and current_session() is not None:
            return _Unrecorded(f"{self._surface.what}.{name}")
        return getattr(self._inner, name)

    def _prepare(self, session: Session, kwargs: dict[str, Any], method: str) -> _Prepared:
        return _prepare(session, kwargs, f"{self._surface.what}.{method}", self._surface)


class _MessagesProxy(_MessagesBase):
    def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.create(**kwargs)
        p = self._prepare(session, kwargs, "create")
        if not kwargs.get("stream"):
            return session.intercept(_call(p, invoke=lambda: self._inner.create(**p.kwargs)))
        surface = self._surface
        message = session.intercept(
            _call(p, invoke=lambda: _drain(self._inner.create(**p.kwargs), surface))
        )
        return SyntheticStream(surface.events_of(message), surface.betas_of(kwargs))

    def stream(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.stream(**kwargs)
        p = self._prepare(session, {**kwargs, "stream": True}, "stream")
        surface = self._surface
        call = _call(p, invoke=lambda: _drain(self._inner.create(**p.kwargs), surface))
        return _StreamManager(lambda: session.intercept(call), p, surface)

    def parse(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.parse(**kwargs)
        p = self._prepare(session, kwargs, "parse")
        message = session.intercept(_call(p, invoke=lambda: self._inner.create(**p.kwargs)))
        return parse_response(output_format=p.output_format, response=message)


class _AsyncMessagesProxy(_MessagesBase):
    async def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.create(**kwargs)
        p = self._prepare(session, kwargs, "create")
        if not kwargs.get("stream"):
            return await session.intercept_async(
                _call(p, ainvoke=lambda: self._inner.create(**p.kwargs))
            )
        surface = self._surface
        message = await session.intercept_async(
            _call(p, ainvoke=lambda: _adrain(self._inner.create(**p.kwargs), surface))
        )
        return AsyncSyntheticStream(surface.events_of(message), surface.betas_of(kwargs))

    def stream(self, **kwargs: Any) -> Any:
        # Not a coroutine function, like the SDK's: ``async with client.messages.stream(...)``.
        session = current_session()
        if session is None:
            return self._inner.stream(**kwargs)
        p = self._prepare(session, {**kwargs, "stream": True}, "stream")
        surface = self._surface
        call = _call(p, ainvoke=lambda: _adrain(self._inner.create(**p.kwargs), surface))
        return _AsyncStreamManager(lambda: session.intercept_async(call), p, surface)

    async def parse(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.parse(**kwargs)
        p = self._prepare(session, kwargs, "parse")
        message = await session.intercept_async(
            _call(p, ainvoke=lambda: self._inner.create(**p.kwargs))
        )
        return parse_response(output_format=p.output_format, response=message)


class _BetaMessagesProxy(_MessagesProxy):
    _surface = BETA_MESSAGES
    _unrecorded = _UNRECORDED_BETA_MESSAGES


class _AsyncBetaMessagesProxy(_AsyncMessagesProxy):
    _surface = BETA_MESSAGES
    _unrecorded = _UNRECORDED_BETA_MESSAGES


class _BetaProxy:
    """``client.beta``: ``messages`` is recorded; every other beta resource is refused."""

    def __init__(self, inner: Any, messages: _MessagesBase) -> None:
        self._inner = inner
        self.messages = messages

    def __getattr__(self, name: str) -> Any:
        if current_session() is not None and not name.startswith("__"):
            return _Unrecorded(f"client.beta.{name}")
        return getattr(self._inner, name)


class _ClientProxy(WrappedClient):
    _messages: type[_MessagesBase] = _MessagesProxy
    _beta_messages: type[_MessagesBase] = _BetaMessagesProxy

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.messages: _MessagesBase = self._messages(inner.messages)
        # Lazily: some clients (the fakes, the offline stand-ins) have no ``beta``.
        self._beta: _BetaProxy | None = None

    @property
    def beta(self) -> Any:
        if self._beta is None:
            inner_beta = self._inner.beta
            self._beta = _BetaProxy(inner_beta, self._beta_messages(inner_beta.messages))
        return self._beta

    def with_options(self, **kwargs: Any) -> Any:
        """A copy of the client with other options, still recorded.

        pydantic-ai sends a request that sets ``max_retries`` through such a copy;
        without this the copy would be the bare client and the call would go live.
        A client with no options to copy (the fakes, the offline stand-ins) is its own copy.
        """
        with_options = getattr(self._inner, "with_options", None)
        return self if with_options is None else type(self)(with_options(**kwargs))

    copy = with_options

    def with_middleware(self, *middleware: Any) -> Any:
        """:meth:`with_options` for ``client.with_middleware(...)``: the copy stays recorded."""
        return type(self)(self._inner.with_middleware(*middleware))

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED_CLIENT and current_session() is not None:
            return _Unrecorded(f"client.{name}")
        return getattr(self._inner, name)

    def __enter__(self) -> Any:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> Any:
        return self._inner.__exit__(*exc_info)


class _AsyncClientProxy(_ClientProxy):
    _messages = _AsyncMessagesProxy
    _beta_messages = _AsyncBetaMessagesProxy

    async def __aenter__(self) -> Any:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> Any:
        return await self._inner.__aexit__(*exc_info)


def _offline_error() -> ReplayError:
    # Fatal, like an unrecorded call: an agent that retries or swallows model
    # errors must not turn "no client to fall back to" into a passing run.
    return fail_current(
        ReplayError(
            "this replay has no live model client, so an unrecorded model call cannot run; "
            "pass one (--client anthropic, or replay(..., client=...)) to use the fallback "
            "policy for model calls, or re-record"
        )
    )


class _OfflineOptions:
    """The client options frameworks read without making a call.

    pydantic-ai reads ``base_url`` (into every response's ``provider_url``),
    ``timeout`` and, for ``max_retries``, ``_client``; the SDK's defaults stand in,
    since no request is ever sent.
    """

    base_url = "https://api.anthropic.com"
    timeout = anthropic.DEFAULT_TIMEOUT
    max_retries = anthropic.DEFAULT_MAX_RETRIES
    _client = None
    """pydantic-ai passes it to ``with_options(http_client=...)`` when ``max_retries`` is set."""


class _Offline(_OfflineOptions):
    """Stands in for the Anthropic client when replay serves the model and none was given.

    Streams and ``parse`` are performed through ``messages.create`` too, so
    this method and its twin on ``beta.messages`` cover every recorded surface.
    """

    class _Messages:
        def create(self, **_: object) -> NoReturn:
            raise _offline_error()

    class _Beta:
        def __init__(self, messages: object) -> None:
            self.messages = messages

    def __init__(self) -> None:
        self.messages = self._Messages()
        self.beta = self._Beta(self._Messages())


class _AsyncOffline(_OfflineOptions):
    """:class:`_Offline` for async agents: ``await messages.create(...)`` raises the same."""

    class _Messages:
        async def create(self, **_: object) -> NoReturn:
            raise _offline_error()

    def __init__(self) -> None:
        self.messages = self._Messages()
        self.beta = _Offline._Beta(self._Messages())


def _is_async(client: object) -> bool:
    # AsyncAnthropic's ``messages.create`` is wrapped by a decorator that hides
    # the coroutine flag, so test the SDK types before the function itself.
    messages = getattr(client, "messages", None)
    return (
        isinstance(client, anthropic.AsyncAnthropic)
        or isinstance(messages, AsyncMessages)
        or inspect.iscoroutinefunction(getattr(messages, "create", None))
    )


class AnthropicAdapter:
    """The :class:`~agent_replay.record.ProviderAdapter` for the ``anthropic`` SDK."""

    name = "anthropic"

    def accepts(self, client: object) -> bool:
        return callable(getattr(getattr(client, "messages", None), "create", None))

    def wrap(self, client: object) -> WrappedClient:
        return _AsyncClientProxy(client) if _is_async(client) else _ClientProxy(client)

    def owns(self, call_name: str) -> bool:
        return call_name.startswith("anthropic.")  # messages.create and beta.messages.create

    def offline_client(self, *, is_async: bool) -> object:
        return _AsyncOffline() if is_async else _Offline()


ANTHROPIC = AnthropicAdapter()
