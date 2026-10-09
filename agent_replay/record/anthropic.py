"""Interception for the official ``anthropic`` client.

We wrap the client object rather than its HTTP transport: ``anthropic`` 1.x runs
on ``httpx2``, so transport-level mocks written for ``httpx`` never see its
traffic, and the SDK layer hands us typed messages and usage for free.
"""

from __future__ import annotations

import inspect
from typing import Any, NoReturn

import anthropic
from anthropic.types import Message
from pydantic import JsonValue

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ReplayError
from agent_replay.record._base import WrappedClient
from agent_replay.session import Call, current_session

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
            "only synchronous, non-streaming calls can be recorded"
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


class _MessagesProxy:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.create(**kwargs)
        if kwargs.get("stream"):
            _not_recorded("messages.create(stream=True)")
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
                ReplayError(f"messages.create arguments cannot be recorded as JSON: {exc}")
            ) from exc
        call: Call[Message] = Call(
            kind=Kind.MODEL,
            name=MODEL_CALL,
            request=request,
            fingerprint=fingerprint(Kind.MODEL, MODEL_CALL, request, ignore=_VOLATILE_FIELDS),
            invoke=lambda: self._inner.create(**kwargs),
            encode=_encode,
            decode=_decode,
            usage=_usage,
            model_of=lambda message: message.model,
        )
        return session.intercept(call)

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED_MESSAGES and current_session() is not None:
            return _Unrecorded(f"messages.{name}")
        return getattr(self._inner, name)


class _ClientProxy(WrappedClient):
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.messages = _MessagesProxy(inner.messages)

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED_CLIENT and current_session() is not None:
            return _Unrecorded(f"client.{name}")
        return getattr(self._inner, name)

    def __enter__(self) -> Any:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> Any:
        return self._inner.__exit__(*exc_info)


class _Offline:
    """Stands in for the Anthropic client when replay serves the model and none was given."""

    class _Messages:
        def create(self, **_: object) -> object:
            raise ReplayError(
                "this replay has no live model client, so an unrecorded model call cannot run; "
                "pass one (--client anthropic, or replay(..., client=...)) to use the fallback "
                "policy for model calls, or re-record"
            )

    def __init__(self) -> None:
        self.messages = self._Messages()


class AnthropicAdapter:
    """The :class:`~agent_replay.record.ProviderAdapter` for the ``anthropic`` SDK."""

    name = "anthropic"

    def accepts(self, client: object) -> bool:
        return callable(getattr(getattr(client, "messages", None), "create", None))

    def wrap(self, client: object) -> WrappedClient:
        create = getattr(getattr(client, "messages", None), "create", None)
        if isinstance(client, anthropic.AsyncAnthropic) or inspect.iscoroutinefunction(create):
            raise TypeError("async clients are not supported yet; wrap anthropic.Anthropic")
        return _ClientProxy(client)

    def owns(self, call_name: str) -> bool:
        return call_name.startswith("anthropic.")

    def offline_client(self, *, is_async: bool) -> object:
        return _Offline()


ANTHROPIC = AnthropicAdapter()
