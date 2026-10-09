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
from anthropic.lib.streaming import AsyncMessageStream, MessageStream
from anthropic.resources.messages import AsyncMessages
from anthropic.types import Message
from pydantic import JsonValue, PydanticSchemaGenerationError, TypeAdapter

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ReplayError
from agent_replay.record._base import WrappedClient
from agent_replay.record._streams import AsyncSyntheticStream, SyntheticStream, events_from_message
from agent_replay.session import Call, Session, current_session

MODEL_CALL = "anthropic.messages.create"

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
_UNRECORDED_MESSAGES = frozenset({"with_raw_response", "with_streaming_response"})
_UNRECORDED_CLIENT = frozenset({"beta", "with_raw_response", "with_streaming_response"})


def _not_recorded(what: str) -> NoReturn:
    session = current_session()
    assert session is not None
    raise session.fail(
        ReplayError(
            f"{what} is not recorded yet, so it cannot run inside a recording or replay "
            "session; use client.messages.create(...), .stream(...) or .parse(...)"
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


def _encode(message: object) -> JsonValue:
    if not isinstance(message, Message):
        raise TypeError(
            f"messages.create returned {type(message).__name__}, not a Message; "
            "wrap the client itself, not a raw-response view of it"
        )
    return to_jsonable(message)


def _decode(data: JsonValue) -> Message:
    return Message.model_validate(data)


def _usage(message: Message) -> Usage:
    u = message.usage
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
        cache_read_input_tokens=u.cache_read_input_tokens or 0,
    )


def _model_of(message: Message) -> str:
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


def _prepare(session: Session, kwargs: dict[str, Any], what: str) -> _Prepared:
    output_format: Any = anthropic.NOT_GIVEN
    if what != "messages.create":  # only the helpers take output_format
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
        fingerprint=fingerprint(Kind.MODEL, MODEL_CALL, request, ignore=_UNIDENTIFYING_FIELDS),
        output_format=output_format,
    )


def _call(
    prepared: _Prepared,
    *,
    invoke: Callable[[], Message] | None = None,
    ainvoke: Callable[[], Awaitable[Message]] | None = None,
) -> Call[Message]:
    return Call(
        kind=Kind.MODEL,
        name=MODEL_CALL,
        request=prepared.request,
        fingerprint=prepared.fingerprint,
        invoke=invoke,
        ainvoke=ainvoke,
        encode=_encode,
        decode=_decode,
        usage=_usage,
        model_of=_model_of,
    )


def _drain(raw: Any) -> Message:
    """The final message of a real stream, accumulated by the SDK's own ``MessageStream``."""
    with MessageStream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = stream.get_final_message()
    return Message.model_validate(to_jsonable(final))


async def _adrain(pending: Awaitable[Any]) -> Message:
    """:func:`_drain` for an async client's ``create(stream=True)``."""
    raw = await pending
    async with AsyncMessageStream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = await stream.get_final_message()
    return Message.model_validate(to_jsonable(final))


class _StreamManager:
    """What ``messages.stream(...)`` returns in a session; the call happens on ``__enter__``.

    As with the SDK's ``MessageStreamManager``, nothing is requested until the
    ``with`` block is entered; the block then gets the SDK's own
    ``MessageStream`` over the rebuilt events, so ``text_stream``,
    ``get_final_message()`` and ``until_done()`` behave exactly as they do live.
    """

    def __init__(self, request: Callable[[], Message], output_format: Any) -> None:
        self._request = request
        self._output_format = output_format
        self._stream: MessageStream[Any] | None = None

    def __enter__(self) -> MessageStream[Any]:
        raw = SyntheticStream(events_from_message(self._request()))
        self._stream = MessageStream(cast(Any, raw), output_format=self._output_format)
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

    def __init__(self, request: Callable[[], Awaitable[Message]], output_format: Any) -> None:
        self._request = request
        self._output_format = output_format
        self._stream: AsyncMessageStream[Any] | None = None

    async def __aenter__(self) -> AsyncMessageStream[Any]:
        raw = AsyncSyntheticStream(events_from_message(await self._request()))
        self._stream = AsyncMessageStream(cast(Any, raw), output_format=self._output_format)
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
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED_MESSAGES and current_session() is not None:
            return _Unrecorded(f"messages.{name}")
        return getattr(self._inner, name)


class _MessagesProxy(_MessagesBase):
    def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.create(**kwargs)
        p = _prepare(session, kwargs, "messages.create")
        if not kwargs.get("stream"):
            return session.intercept(_call(p, invoke=lambda: self._inner.create(**p.kwargs)))
        message = session.intercept(_call(p, invoke=lambda: _drain(self._inner.create(**p.kwargs))))
        return SyntheticStream(events_from_message(message))

    def stream(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.stream(**kwargs)
        p = _prepare(session, {**kwargs, "stream": True}, "messages.stream")
        call = _call(p, invoke=lambda: _drain(self._inner.create(**p.kwargs)))
        return _StreamManager(lambda: session.intercept(call), p.output_format)

    def parse(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.parse(**kwargs)
        p = _prepare(session, kwargs, "messages.parse")
        message = session.intercept(_call(p, invoke=lambda: self._inner.create(**p.kwargs)))
        return parse_response(output_format=p.output_format, response=message)


class _AsyncMessagesProxy(_MessagesBase):
    async def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.create(**kwargs)
        p = _prepare(session, kwargs, "messages.create")
        if not kwargs.get("stream"):
            return await session.intercept_async(
                _call(p, ainvoke=lambda: self._inner.create(**p.kwargs))
            )
        message = await session.intercept_async(
            _call(p, ainvoke=lambda: _adrain(self._inner.create(**p.kwargs)))
        )
        return AsyncSyntheticStream(events_from_message(message))

    def stream(self, **kwargs: Any) -> Any:
        # Not a coroutine function, like the SDK's: ``async with client.messages.stream(...)``.
        session = current_session()
        if session is None:
            return self._inner.stream(**kwargs)
        p = _prepare(session, {**kwargs, "stream": True}, "messages.stream")
        call = _call(p, ainvoke=lambda: _adrain(self._inner.create(**p.kwargs)))
        return _AsyncStreamManager(lambda: session.intercept_async(call), p.output_format)

    async def parse(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.parse(**kwargs)
        p = _prepare(session, kwargs, "messages.parse")
        message = await session.intercept_async(
            _call(p, ainvoke=lambda: self._inner.create(**p.kwargs))
        )
        return parse_response(output_format=p.output_format, response=message)


class _ClientProxy(WrappedClient):
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.messages: _MessagesBase = _MessagesProxy(inner.messages)

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
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.messages = _AsyncMessagesProxy(inner.messages)

    async def __aenter__(self) -> Any:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> Any:
        return await self._inner.__aexit__(*exc_info)


def _offline_error() -> ReplayError:
    return ReplayError(
        "this replay has no live model client, so an unrecorded model call cannot run; "
        "pass one (--client anthropic, or replay(..., client=...)) to use the fallback "
        "policy for model calls, or re-record"
    )


class _Offline:
    """Stands in for the Anthropic client when replay serves the model and none was given.

    Streams and ``parse`` are performed through ``messages.create`` too, so
    this one method covers every recorded surface.
    """

    class _Messages:
        def create(self, **_: object) -> NoReturn:
            raise _offline_error()

    def __init__(self) -> None:
        self.messages = self._Messages()


class _AsyncOffline:
    """:class:`_Offline` for async agents: ``await messages.create(...)`` raises the same."""

    class _Messages:
        async def create(self, **_: object) -> NoReturn:
            raise _offline_error()

    def __init__(self) -> None:
        self.messages = self._Messages()


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
        return call_name.startswith("anthropic.")

    def offline_client(self, *, is_async: bool) -> object:
        return _AsyncOffline() if is_async else _Offline()


ANTHROPIC = AnthropicAdapter()
