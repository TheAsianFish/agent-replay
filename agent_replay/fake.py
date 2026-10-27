"""A deterministic stand-in for ``anthropic.Anthropic`` for offline tests and demos.

``FakeAnthropic(policy)`` implements ``messages.create`` by asking ``policy`` what
to say given the request, and returns real ``anthropic.types.Message`` objects,
so agent code and the recorder cannot tell it apart from the API. Token counts
are synthetic (about four characters per token) and only exist so token and
budget code paths have something to work with; responses report the model as
``fake:<requested model>``, which has no price, so fake runs never show a cost.
:class:`FakeAsyncAnthropic` is the same model behind ``await messages.create``.
Both stream (``create(stream=True)`` and ``messages.stream(...)``), delivering
the same message as raw events, so streaming agents record offline too, and
both answer on ``beta.messages.create`` with ``BetaMessage`` and beta stream
events, the surface frameworks such as pydantic-ai call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib._parse._response import parse_response
from anthropic.lib.streaming import AsyncMessageStreamManager, MessageStreamManager
from anthropic.types import Message, ParsedMessage
from anthropic.types.beta import BetaMessage
from pydantic import JsonValue

from agent_replay._json import canonical_json, to_jsonable
from agent_replay.record._streams import (
    BETA_MESSAGE_EVENTS,
    AsyncSyntheticStream,
    SyntheticStream,
    events_from_message,
)
from agent_replay.record.anthropic import wire_output_format


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Reply:
    """What the fake model says on one turn: optional text, then tool calls."""

    text: str | None = None
    tool_calls: Sequence[ToolCall] = ()


@dataclass(frozen=True)
class ToolResult:
    tool_use_id: str
    name: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class FakeRequest:
    """A ``messages.create`` request as plain JSON, with helpers for policies."""

    model: str
    system: JsonValue
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]

    def first_user_text(self) -> str:
        return _text_of(self.messages[0]["content"]) if self.messages else ""

    def tool_results(self) -> list[ToolResult]:
        """Every tool result in the conversation so far, oldest first."""
        names: dict[str, str] = {}
        results: list[ToolResult] = []
        for message in self.messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
                elif block.get("type") == "tool_result":
                    results.append(
                        ToolResult(
                            tool_use_id=block["tool_use_id"],
                            name=names.get(block["tool_use_id"], "?"),
                            content=_text_of(block.get("content", "")),
                            is_error=bool(block.get("is_error", False)),
                        )
                    )
        return results

    def latest_result(self, name: str) -> ToolResult | None:
        for result in reversed(self.tool_results()):
            if result.name == name:
                return result
        return None


Policy = Callable[[FakeRequest], Reply]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def scripted(replies: Sequence[Reply]) -> Policy:
    """A policy that returns ``replies`` in order, regardless of the request."""
    it: Iterator[Reply] = iter(replies)

    def policy(_: FakeRequest) -> Reply:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(
                f"scripted fake model ran out of replies after {len(replies)} turn(s)"
            ) from None

    return policy


class _FakeModel:
    """The policy and turn counter behind a fake client's ``messages``."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._turn = 0

    def reply(self, kwargs: dict[str, Any]) -> Message:
        # ``stream`` only changes the delivery, so it does not count towards usage.
        payload = to_jsonable(
            {
                k: v
                for k, v in kwargs.items()
                if k != "stream" and not isinstance(v, anthropic.NotGiven | anthropic.Omit)
            }
        )
        assert isinstance(payload, dict)
        request = FakeRequest(
            model=str(payload["model"]),
            system=payload.get("system"),
            messages=cast(list[dict[str, Any]], payload["messages"]),
            tools=cast(list[dict[str, Any]], payload.get("tools", [])),
        )
        reply = self._policy(request)
        turn = self._turn
        self._turn += 1
        content: list[dict[str, JsonValue]] = []
        if reply.text is not None:
            content.append({"type": "text", "text": reply.text})
        for i, call in enumerate(reply.tool_calls):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_fake_{turn:03d}_{i}",
                    "name": call.name,
                    "input": dict(call.input),
                }
            )
        return Message.model_validate(
            {
                "id": f"msg_fake_{turn:03d}",
                "type": "message",
                "role": "assistant",
                # Not a real model id, so the synthetic usage is never priced.
                "model": f"fake:{request.model}",
                "content": content,
                "stop_reason": "tool_use" if reply.tool_calls else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": max(1, len(canonical_json(payload)) // 4),
                    "output_tokens": max(1, len(canonical_json(content)) // 4),
                },
            }
        )


class _FakeMessages:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    def create(self, **kwargs: Any) -> Any:
        message = self._model.reply(kwargs)
        if kwargs.get("stream"):
            return SyntheticStream(events_from_message(message))
        return message

    def stream(self, **kwargs: Any) -> MessageStreamManager[Any]:
        wire, output_format = wire_output_format(kwargs)
        return MessageStreamManager(
            lambda: self.create(**wire, stream=True), output_format=output_format
        )

    def parse(self, **kwargs: Any) -> ParsedMessage[Any]:
        wire, output_format = wire_output_format(kwargs)
        return parse_response(output_format=output_format, response=self._model.reply(wire))


class _FakeAsyncMessages:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    async def create(self, **kwargs: Any) -> Any:
        message = self._model.reply(kwargs)
        if kwargs.get("stream"):
            return AsyncSyntheticStream(events_from_message(message))
        return message

    def stream(self, **kwargs: Any) -> AsyncMessageStreamManager[Any]:
        wire, output_format = wire_output_format(kwargs)
        return AsyncMessageStreamManager(
            self.create(**wire, stream=True), output_format=output_format
        )

    async def parse(self, **kwargs: Any) -> ParsedMessage[Any]:
        wire, output_format = wire_output_format(kwargs)
        return parse_response(output_format=output_format, response=self._model.reply(wire))


def _betas(kwargs: dict[str, Any]) -> list[str]:
    betas = kwargs.get("betas")
    return [str(b) for b in betas] if isinstance(betas, list | tuple) else []


class _FakeBetaMessages:
    """``beta.messages.create``: the same replies as :class:`BetaMessage` and beta events."""

    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    def create(self, **kwargs: Any) -> Any:
        # ``betas`` is a header on the wire, not part of what the model is asked.
        betas = _betas(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k != "betas"}
        message = BetaMessage.model_validate(to_jsonable(self._model.reply(kwargs)))
        if kwargs.get("stream"):
            return SyntheticStream(events_from_message(message, BETA_MESSAGE_EVENTS), betas)
        return message


class _FakeAsyncBetaMessages:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    async def create(self, **kwargs: Any) -> Any:
        betas = _betas(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k != "betas"}
        message = BetaMessage.model_validate(to_jsonable(self._model.reply(kwargs)))
        if kwargs.get("stream"):
            return AsyncSyntheticStream(events_from_message(message, BETA_MESSAGE_EVENTS), betas)
        return message


class _FakeBeta:
    def __init__(self, messages: object) -> None:
        self.messages = messages


class FakeAnthropic:
    """Deterministic fake of the ``anthropic.Anthropic`` client.

    Supports ``messages.create`` (with or without ``stream=True``),
    ``messages.stream(...)`` and ``messages.parse(...)``; a streamed reply is the
    same message delivered as raw stream events, with the same ids and usage.
    For ``parse``, the policy's reply text must be the JSON the output format expects.
    """

    base_url = "https://fake.invalid"
    """Read by frameworks (pydantic-ai's ``provider_url``); never contacted."""
    _client = None
    """Read by pydantic-ai when ``max_retries`` is set (the wrapper's copy is the fake)."""

    def __init__(self, policy: Policy) -> None:
        model = _FakeModel(policy)
        self.messages = _FakeMessages(model)
        self.beta = _FakeBeta(_FakeBetaMessages(model))

    def as_anthropic(self) -> Anthropic:
        """This object, typed as the client it imitates, for agents annotated with it."""
        return cast(Anthropic, self)


class FakeAsyncAnthropic:
    """:class:`FakeAnthropic` for async agents: ``await client.messages.create(...)``.

    Same policy contract, ids and token accounting as the sync fake.
    """

    base_url = "https://fake.invalid"
    _client = None

    def __init__(self, policy: Policy) -> None:
        model = _FakeModel(policy)
        self.messages = _FakeAsyncMessages(model)
        self.beta = _FakeBeta(_FakeAsyncBetaMessages(model))

    def as_anthropic(self) -> AsyncAnthropic:
        """This object, typed as the client it imitates, for agents annotated with it."""
        return cast(AsyncAnthropic, self)
