"""Interception for the official ``anthropic`` client, sync and async.

We wrap the client object rather than its HTTP transport: ``anthropic`` 1.x runs
on ``httpx2``, so transport-level mocks written for ``httpx`` never see its
traffic, and the SDK layer hands us typed messages and usage for free.

``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` share one recording
path: the request is normalised and fingerprinted the same way, and only the
call that performs it differs (``session.intercept`` or
``await session.intercept_async``).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import anthropic
from anthropic.resources.messages import AsyncMessages
from anthropic.types import Message
from pydantic import JsonValue

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ReplayError
from agent_replay.record._base import WrappedClient
from agent_replay.session import Call, Session, current_session

MODEL_CALL = "anthropic.messages.create"

# Transport options: never recorded (headers can carry credentials) and never
# part of the request identity.
_TRANSPORT_KWARGS = frozenset({"extra_headers", "extra_query", "timeout"})
# Recorded, but excluded from the fingerprint because they vary between
# otherwise identical requests.
_VOLATILE_FIELDS = frozenset({"metadata"})
# Client surfaces that would make calls this proxy cannot record yet. Inside a
# session, calling them raises instead of silently going live; merely looking
# them up (``hasattr``) is harmless.
_UNRECORDED_MESSAGES = frozenset(
    {"parse", "stream", "with_raw_response", "with_streaming_response"}
)
_UNRECORDED_CLIENT = frozenset({"beta", "with_raw_response", "with_streaming_response"})


def _not_recorded(what: str) -> NoReturn:
    session = current_session()
    assert session is not None
    raise session.fail(
        ReplayError(
            f"{what} is not recorded yet, so it cannot run inside a recording or replay "
            "session; use client.messages.create(...) without stream=True"
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
            "only non-streaming calls can be recorded"
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


@dataclass(frozen=True)
class _Prepared:
    """One model request, normalised for the cassette."""

    kwargs: dict[str, Any]
    """What the real client is called with (transport options included)."""
    request: JsonValue
    """What the cassette records: the request body without transport options."""
    fingerprint: str


def _prepare(session: Session, kwargs: dict[str, Any], what: str) -> _Prepared:
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
        fingerprint=fingerprint(Kind.MODEL, MODEL_CALL, request, ignore=_VOLATILE_FIELDS),
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
        if kwargs.get("stream"):
            _not_recorded("messages.create(stream=True)")
        p = _prepare(session, kwargs, "messages.create")
        return session.intercept(_call(p, invoke=lambda: self._inner.create(**p.kwargs)))


class _AsyncMessagesProxy(_MessagesBase):
    async def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.create(**kwargs)
        if kwargs.get("stream"):
            _not_recorded("messages.create(stream=True)")
        p = _prepare(session, kwargs, "messages.create")
        return await session.intercept_async(
            _call(p, ainvoke=lambda: self._inner.create(**p.kwargs))
        )


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
    """Stands in for the Anthropic client when replay serves the model and none was given."""

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
