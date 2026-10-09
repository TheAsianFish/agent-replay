"""Interception for the ``ollama`` client (``ollama.Client`` and ``ollama.AsyncClient``).

Only ``chat`` is recorded, as model calls named ``ollama.chat``. ``ollama`` is an
optional dependency (``agent-replay[ollama]``): nothing here imports it until a
client is wrapped or a recorded response is rebuilt, and :meth:`OllamaAdapter.accepts`
is False when it is not installed.

The request is recorded in the form the client sends: ``model`` (positional or
keyword) and ``messages`` as keywords, ``ollama.Message`` objects as plain JSON, and
Python functions in ``tools`` converted to tool schemas by ollama's own helper.
Two fields are recorded but left out of the request fingerprint:

- ``keep_alive`` says how long the server keeps the model loaded; it never
  changes the answer.
- ``stream`` changes how the answer is delivered, not what it is.

``options`` stays in the fingerprint: temperature, seed, ``num_ctx`` and the like
change what the model produces, so a replay with different options should show
up as ``approximate``.

Streaming (``chat(..., stream=True)``) is recorded as the list of chunks. While
recording, the proxy drains the real stream before handing the agent an iterator
over the collected chunks, so the agent sees the first chunk only once the model
has finished: the content is identical, the timing is not. Replay hands back an
iterator over the recorded chunks (an async iterator for ``AsyncClient``).
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import JsonValue

from agent_replay._json import fingerprint, to_jsonable
from agent_replay.cassette import Kind, Usage
from agent_replay.errors import ReplayError
from agent_replay.record._base import WrappedClient
from agent_replay.session import Call, current_session, fail_current

if TYPE_CHECKING:
    from ollama import ChatResponse

MODEL_CALL = "ollama.chat"

# Recorded, but excluded from the fingerprint (see the module docstring).
_VOLATILE_FIELDS = frozenset({"keep_alive", "stream"})
# Client methods that send requests this proxy does not record. Inside a session,
# calling them fails the run instead of silently going live.
_UNRECORDED = frozenset(
    {
        "generate",
        "embed",
        "embeddings",
        "systemone",
        "web_search",
        "web_fetch",
        "pull",
        "push",
        "create",
        "delete",
        "copy",
        "list",
        "show",
        "ps",
    }
)


def _not_recorded(what: str) -> NoReturn:
    session = current_session()
    assert session is not None
    raise session.fail(
        ReplayError(
            f"ollama {what} is not recorded, so it cannot run inside a recording or replay "
            "session; only client.chat(...) is recorded"
        )
    )


class _Unrecorded:
    """Stands in for a client method the proxy cannot record; calling it fails the session."""

    def __init__(self, what: str) -> None:
        self._what = what

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        _not_recorded(self._what)

    def __repr__(self) -> str:
        return f"<unrecorded ollama {self._what}>"


# --- request ------------------------------------------------------------------


def _request_json(
    model: str, messages: Sequence[Any] | None, kwargs: Mapping[str, Any]
) -> JsonValue:
    """The chat request as JSON, converted the way ``ollama.Client.chat`` converts it."""
    from ollama._client import _copy_messages, _copy_tools  # noqa: PLC0415 - optional dependency

    request: dict[str, object] = {
        "model": model,
        "messages": [
            m.model_dump(mode="json", exclude_none=True) for m in _copy_messages(messages)
        ],
    }
    for key, value in kwargs.items():
        if value is None:
            continue  # the client drops None fields before sending
        if key == "tools":
            request[key] = [
                t.model_dump(mode="json", exclude_none=True, by_alias=True)
                for t in _copy_tools(value)
            ]
        elif hasattr(value, "model_dump"):  # ollama.Options
            request[key] = value.model_dump(mode="json", exclude_none=True)
        else:
            request[key] = value
    return to_jsonable(request)


# --- response -----------------------------------------------------------------


def _encode(response: ChatResponse) -> JsonValue:
    from ollama import ChatResponse  # noqa: PLC0415 - optional dependency

    if not isinstance(response, ChatResponse):
        raise TypeError(f"chat returned {type(response).__name__}, not a ChatResponse")
    return to_jsonable(response.model_dump(mode="json", exclude_none=True))


def _decode(data: JsonValue) -> ChatResponse:
    from ollama import ChatResponse  # noqa: PLC0415 - optional dependency

    return ChatResponse.model_validate(data)


def _encode_chunks(chunks: list[ChatResponse]) -> JsonValue:
    return [_encode(c) for c in chunks]


def _decode_chunks(data: JsonValue) -> list[ChatResponse]:
    if not isinstance(data, list):
        raise TypeError("a streamed chat is recorded as a list of chunks")
    return [_decode(c) for c in data]


def _usage(response: ChatResponse) -> Usage | None:
    if response.prompt_eval_count is None and response.eval_count is None:
        return None
    return Usage(
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
    )


def _chunks_usage(chunks: list[ChatResponse]) -> Usage | None:
    # Ollama reports the counts on the final (done) chunk.
    for chunk in reversed(chunks):
        if (usage := _usage(chunk)) is not None:
            return usage
    return None


def _chunks_model(chunks: list[ChatResponse]) -> str | None:
    return chunks[-1].model if chunks else None


# --- proxies ------------------------------------------------------------------


def _request_or_fail(
    model: str, messages: Sequence[Any] | None, kwargs: Mapping[str, Any]
) -> tuple[JsonValue, str]:
    session = current_session()
    assert session is not None
    try:
        request = _request_json(model, messages, kwargs)
    except (TypeError, ValueError) as exc:
        raise session.fail(
            ReplayError(f"ollama chat arguments cannot be recorded as JSON: {exc}")
        ) from exc
    return request, fingerprint(Kind.MODEL, MODEL_CALL, request, ignore=_VOLATILE_FIELDS)


def _chat_call(
    request: JsonValue,
    fp: str,
    *,
    invoke: Callable[[], ChatResponse] | None = None,
    ainvoke: Callable[[], Awaitable[ChatResponse]] | None = None,
) -> Call[ChatResponse]:
    return Call(
        kind=Kind.MODEL,
        name=MODEL_CALL,
        request=request,
        fingerprint=fp,
        invoke=invoke,
        ainvoke=ainvoke,
        encode=_encode,
        decode=_decode,
        usage=_usage,
        model_of=lambda r: r.model,
    )


def _stream_call(
    request: JsonValue,
    fp: str,
    *,
    invoke: Callable[[], list[ChatResponse]] | None = None,
    ainvoke: Callable[[], Awaitable[list[ChatResponse]]] | None = None,
) -> Call[list[ChatResponse]]:
    return Call(
        kind=Kind.MODEL,
        name=MODEL_CALL,
        request=request,
        fingerprint=fp,
        invoke=invoke,
        ainvoke=ainvoke,
        encode=_encode_chunks,
        decode=_decode_chunks,
        usage=_chunks_usage,
        model_of=_chunks_model,
    )


class _Proxy(WrappedClient):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED and current_session() is not None:
            return _Unrecorded(f"client.{name}")
        return getattr(self._inner, name)


class _SyncProxy(_Proxy):
    def chat(self, model: str = "", messages: Sequence[Any] | None = None, **kwargs: Any) -> Any:
        session = current_session()
        if session is None:
            return self._inner.chat(model, messages, **kwargs)
        request, fp = _request_or_fail(model, messages, kwargs)
        if kwargs.get("stream"):
            chunks = session.intercept(
                _stream_call(
                    request, fp, invoke=lambda: list(self._inner.chat(model, messages, **kwargs))
                )
            )
            return iter(chunks)
        return session.intercept(
            _chat_call(request, fp, invoke=lambda: self._inner.chat(model, messages, **kwargs))
        )

    def __enter__(self) -> Any:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> Any:
        return self._inner.__exit__(*exc_info)


async def _aiter(chunks: list[ChatResponse]) -> AsyncIterator[ChatResponse]:
    for chunk in chunks:
        yield chunk


class _AsyncProxy(_Proxy):
    async def chat(
        self, model: str = "", messages: Sequence[Any] | None = None, **kwargs: Any
    ) -> Any:
        session = current_session()
        if session is None:
            return await self._inner.chat(model, messages, **kwargs)
        request, fp = _request_or_fail(model, messages, kwargs)
        if kwargs.get("stream"):

            async def drain() -> list[ChatResponse]:
                stream = await self._inner.chat(model, messages, **kwargs)
                return [chunk async for chunk in stream]

            chunks = await session.intercept_async(_stream_call(request, fp, ainvoke=drain))
            return _aiter(chunks)

        async def once() -> ChatResponse:
            response: ChatResponse = await self._inner.chat(model, messages, **kwargs)
            return response

        return await session.intercept_async(_chat_call(request, fp, ainvoke=once))

    async def __aenter__(self) -> Any:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> Any:
        return await self._inner.__aexit__(*exc_info)


# --- offline stand-in ---------------------------------------------------------

_OFFLINE_MESSAGE = (
    "this replay has no live Ollama client, so an unrecorded model call cannot run; "
    "pass one (--client module:factory returning ollama.Client(), or "
    "replay(..., client=...)) to use the fallback policy for model calls, or re-record"
)


class _Offline:
    """Stands in for ``ollama.Client`` when replay serves the model and none was given."""

    def chat(self, *args: object, **kwargs: object) -> NoReturn:
        raise fail_current(ReplayError(_OFFLINE_MESSAGE))


class _AsyncOffline:
    """Stands in for ``ollama.AsyncClient`` when replay serves the model and none was given."""

    async def chat(self, *args: object, **kwargs: object) -> NoReturn:
        raise fail_current(ReplayError(_OFFLINE_MESSAGE))


# --- adapter ------------------------------------------------------------------


def _ollama_clients() -> tuple[type, ...]:
    try:
        import ollama  # noqa: PLC0415 - optional dependency
    except ImportError:
        return ()
    return (ollama.Client, ollama.AsyncClient)


class OllamaAdapter:
    """The :class:`~agent_replay.record.ProviderAdapter` for the ``ollama`` SDK."""

    name = "ollama"

    def accepts(self, client: object) -> bool:
        return isinstance(client, (_Offline, _AsyncOffline, *_ollama_clients()))

    def wrap(self, client: object) -> WrappedClient:
        if not self.accepts(client):
            raise TypeError(f"cannot wrap {type(client).__name__}: it is not an ollama client")
        if inspect.iscoroutinefunction(getattr(client, "chat", None)):
            return _AsyncProxy(client)
        return _SyncProxy(client)

    def owns(self, call_name: str) -> bool:
        return call_name.startswith("ollama.")

    def offline_client(self, *, is_async: bool) -> object:
        return _AsyncOffline() if is_async else _Offline()


OLLAMA = OllamaAdapter()

__all__ = ["MODEL_CALL", "OLLAMA", "OllamaAdapter"]
