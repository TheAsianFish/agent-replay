"""A deterministic stand-in for ``ollama.Client`` / ``ollama.AsyncClient`` for offline tests.

``FakeOllama(policy)`` implements ``chat`` by asking ``policy`` what to say given
the request, and returns real ``ollama.ChatResponse`` objects (a list of chunks
with ``stream=True``), so agent code and the recorder cannot tell it apart from
a local server. Token counts are synthetic (about four characters per token).

The fakes subclass ``ollama.Client`` / ``ollama.AsyncClient`` without calling
their ``__init__`` (which would build an HTTP client): agents annotated with the
real client type accept them without a cast, and ``OllamaAdapter.accepts`` stays
a plain ``isinstance`` check instead of trusting a duck-typed ``chat``. Every
other client method fails loudly rather than reaching for a server.

Policies are shared with :class:`agent_replay.fake.FakeAnthropic`: they return a
:class:`~agent_replay.fake.Reply` and receive a :class:`FakeOllamaRequest`, a
:class:`~agent_replay.fake.FakeRequest` whose helpers read Ollama's message
shapes, so a policy written for the Anthropic fake works here unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, overload

import ollama
from ollama import ChatResponse, Message, Options, Tool
from ollama._client import _copy_messages, _copy_tools
from pydantic.json_schema import JsonSchemaValue

from agent_replay._json import canonical_json
from agent_replay.fake import FakeRequest, Reply, ToolResult

ERROR_PREFIX = "error:"
"""Ollama tool messages have no error flag; content starting with this counts as one."""

type Messages = Sequence[Mapping[str, Any] | Message] | None
type Tools = Sequence[Mapping[str, Any] | Tool | Callable[..., Any]] | None
type Format = Literal["", "json"] | JsonSchemaValue | None


@dataclass(frozen=True)
class FakeOllamaRequest(FakeRequest):
    """An Ollama ``chat`` request as plain JSON, with the :class:`FakeRequest` helpers.

    ``system`` is the content of the first ``system`` message, if any.
    """

    def first_user_text(self) -> str:
        for message in self.messages:
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def tool_results(self) -> list[ToolResult]:
        """Every ``role: tool`` message so far, oldest first.

        Ollama has no tool-call ids, so ``tool_use_id`` is the message's position.
        """
        return [
            ToolResult(
                tool_use_id=f"msg_{i}",
                name=str(m.get("tool_name") or "?"),
                content=str(m.get("content") or ""),
                is_error=str(m.get("content") or "").startswith(ERROR_PREFIX),
            )
            for i, m in enumerate(self.messages)
            if m.get("role") == "tool"
        ]


type OllamaPolicy = Callable[[FakeOllamaRequest], Reply]


class _Brain:
    """The shared part of the sync and async fakes: turns a request into chunks."""

    def __init__(self, policy: OllamaPolicy) -> None:
        self.policy = policy
        self.turn = 0

    def respond(self, model: str, messages: Messages, tools: Tools) -> list[ChatResponse]:
        sent_messages = [
            m.model_dump(mode="json", exclude_none=True) for m in _copy_messages(messages)
        ]
        sent_tools = [
            t.model_dump(mode="json", exclude_none=True, by_alias=True) for t in _copy_tools(tools)
        ]
        system = next((m.get("content") for m in sent_messages if m["role"] == "system"), None)
        request = FakeOllamaRequest(
            model=model, system=system, messages=sent_messages, tools=sent_tools
        )
        reply = self.policy(request)
        self.turn += 1
        tool_calls = [
            Message.ToolCall(
                function=Message.ToolCall.Function(name=c.name, arguments=dict(c.input))
            )
            for c in reply.tool_calls
        ]
        text = reply.text or ""
        prompt_tokens = max(1, len(canonical_json([sent_messages, sent_tools])) // 4)
        output_tokens = max(1, (len(text) + len(canonical_json(reply.tool_calls))) // 4)
        # Content arrives word by word; the last chunk carries the tool calls and counts.
        words = [w + " " for w in text.split(" ")]
        if words:
            words[-1] = words[-1][:-1]
        chunks = [
            ChatResponse(
                model=model,
                created_at=_created_at(self.turn),
                done=False,
                message=Message(role="assistant", content=w),
            )
            for w in words
            if w
        ]
        chunks.append(
            ChatResponse(
                model=model,
                created_at=_created_at(self.turn),
                done=True,
                done_reason="stop",
                prompt_eval_count=prompt_tokens,
                eval_count=output_tokens,
                message=Message(role="assistant", content="", tool_calls=tool_calls or None),
            )
        )
        return chunks

    def whole(self, chunks: list[ChatResponse]) -> ChatResponse:
        """The non-streamed response: the chunks merged into one."""
        last = chunks[-1]
        content = "".join(c.message.content or "" for c in chunks)
        return last.model_copy(
            update={"message": last.message.model_copy(update={"content": content})}
        )


def _created_at(turn: int) -> str:
    return f"2026-01-01T00:00:{turn:02d}Z"


def _no_server(what: str) -> NoReturn:
    raise NotImplementedError(f"the fake Ollama client only implements chat, not {what}")


class FakeOllama(ollama.Client):
    """Deterministic fake of ``ollama.Client`` (``chat`` only, streaming included)."""

    def __init__(self, policy: OllamaPolicy) -> None:
        self._brain = _Brain(policy)

    @overload
    def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: Literal[False] = False,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse: ...

    @overload
    def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: Literal[True] = True,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> Iterator[ChatResponse]: ...

    def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: bool = False,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse | Iterator[ChatResponse]:
        chunks = self._brain.respond(model, messages, tools)
        return iter(chunks) if stream else self._brain.whole(chunks)

    def _request_raw(self, *args: Any, **kwargs: Any) -> NoReturn:
        _no_server("raw requests")

    def close(self) -> None:
        pass


class FakeAsyncOllama(ollama.AsyncClient):
    """Deterministic fake of ``ollama.AsyncClient`` (``chat`` only, streaming included)."""

    def __init__(self, policy: OllamaPolicy) -> None:
        self._brain = _Brain(policy)

    @overload
    async def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: Literal[False] = False,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse: ...

    @overload
    async def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: Literal[True] = True,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> AsyncIterator[ChatResponse]: ...

    async def chat(
        self,
        model: str = "",
        messages: Messages = None,
        *,
        tools: Tools = None,
        stream: bool = False,
        think: bool | str | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: Format = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
    ) -> ChatResponse | AsyncIterator[ChatResponse]:
        chunks = self._brain.respond(model, messages, tools)
        if stream:
            return _aiter(chunks)
        return self._brain.whole(chunks)

    async def _request_raw(self, *args: Any, **kwargs: Any) -> NoReturn:
        _no_server("raw requests")

    async def close(self) -> None:
        pass


async def _aiter(chunks: list[ChatResponse]) -> AsyncIterator[ChatResponse]:
    for chunk in chunks:
        yield chunk
