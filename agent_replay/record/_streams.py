"""Synthetic message streams rebuilt from a complete ``Message``.

A cassette stores the final message of a streamed call, not its server-sent
events: the events carry no information the message lacks (only the chunking
of the text, which the model chooses afresh every time), and one stored shape
means a recording made with streaming replays against code that does not
stream, and vice versa. :func:`events_from_message` turns the message back into
the event sequence the API would send for it, in the coarsest chunking: one
delta per content block. Fed through the SDK's own ``MessageStream`` it
accumulates to a message equal to the original, so ``text_stream``,
``get_final_message()`` and event iteration behave as they do against the API.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, NoReturn, Self

import httpx2
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
)
from anthropic.types.beta import (
    BetaMessage,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
    BetaRawMessageStreamEvent,
)
from pydantic import BaseModel

from agent_replay._json import to_jsonable
from agent_replay.errors import ReplayError

StreamEvent = RawMessageStreamEvent | BetaRawMessageStreamEvent
"""A raw stream event of either surface."""

# Block types whose input arrives as ``input_json_delta`` (the SDK's TRACKS_TOOL_INPUT).
_TOOL_INPUT_BLOCKS = frozenset({"tool_use", "server_tool_use"})
# Cumulative usage totals a ``message_delta`` carries (and the SDK copies over).
_DELTA_USAGE = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "server_tool_use",
    "output_tokens_details",
)


@dataclass(frozen=True)
class EventTypes:
    """The six raw stream event classes of one API surface (``messages`` or ``beta.messages``).

    The two surfaces send the same events under different classes, and code
    such as pydantic-ai checks ``isinstance(event, BetaRawMessageStartEvent)``,
    so a rebuilt stream must use the classes of the surface that was called.
    """

    message_start: type[BaseModel]
    block_start: type[BaseModel]
    block_delta: type[BaseModel]
    block_stop: type[BaseModel]
    message_delta: type[BaseModel]
    message_stop: type[BaseModel]


MESSAGE_EVENTS = EventTypes(
    RawMessageStartEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
)
BETA_MESSAGE_EVENTS = EventTypes(
    BetaRawMessageStartEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStopEvent,
)


def events_from_message(
    message: Message | BetaMessage, types: EventTypes = MESSAGE_EVENTS
) -> list[StreamEvent]:
    """The raw stream events that accumulate to ``message``, as ``types``' classes.

    ``message_start`` carries the message with no content, no stop reason and
    zero output tokens; each content block opens empty and receives its whole
    text, tool input (as JSON), thinking or signature in a single delta; other
    block types (redacted thinking, server tool results) arrive complete in
    ``content_block_start``, as the API sends them; ``message_delta`` carries
    the stop reason and the final usage.
    """
    data = to_jsonable(message)
    assert isinstance(data, dict)
    blocks = data.pop("content")
    assert isinstance(blocks, list)
    usage = data["usage"]
    assert isinstance(usage, dict)
    start: dict[str, Any] = {
        **{k: v for k, v in data.items() if k not in ("stop_reason", "stop_sequence")},
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "stop_details": None,
        "usage": {**usage, "output_tokens": 0},
    }
    events: list[Any] = [
        types.message_start.model_validate({"type": "message_start", "message": start})
    ]
    for index, block in enumerate(blocks):
        assert isinstance(block, dict)
        opened, deltas = _split_block(block)
        events.append(
            types.block_start.model_validate(
                {"type": "content_block_start", "index": index, "content_block": opened}
            )
        )
        events.extend(
            types.block_delta.model_validate(
                {"type": "content_block_delta", "index": index, "delta": delta}
            )
            for delta in deltas
        )
        events.append(
            types.block_stop.model_validate({"type": "content_block_stop", "index": index})
        )
    delta: dict[str, Any] = {
        "stop_reason": data.get("stop_reason"),
        "stop_sequence": data.get("stop_sequence"),
    }
    for key in ("stop_details", "container"):
        if key in data:
            delta[key] = data[key]
    message_delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": delta,
        "usage": {k: usage[k] for k in _DELTA_USAGE if k in usage},
    }
    if data.get("context_management") is not None:  # beta only; the stream copies it over
        message_delta["context_management"] = data["context_management"]
    events.append(types.message_delta.model_validate(message_delta))
    events.append(types.message_stop.model_validate({"type": "message_stop"}))
    return events


def _split_block(block: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A content block as it opens, and the deltas that complete it."""
    kind = block.get("type")
    deltas: list[dict[str, Any]] = []
    if kind == "text":
        opened = {**block, "text": ""}
        if "citations" in block:
            opened["citations"] = []
            deltas += [{"type": "citations_delta", "citation": c} for c in block["citations"]]
        if block["text"]:
            deltas.append({"type": "text_delta", "text": block["text"]})
        return opened, deltas
    if kind in _TOOL_INPUT_BLOCKS:
        deltas.append({"type": "input_json_delta", "partial_json": json.dumps(block["input"])})
        return {**block, "input": {}}, deltas
    if kind == "thinking":
        if block["thinking"]:
            deltas.append({"type": "thinking_delta", "thinking": block["thinking"]})
        if block["signature"]:
            deltas.append({"type": "signature_delta", "signature": block["signature"]})
        return {**block, "thinking": "", "signature": ""}, deltas
    return block, deltas


class NoHTTPResponseError(ReplayError, AttributeError):
    """A rebuilt stream was asked for its HTTP response; there is none.

    Also an :class:`AttributeError`, so ``hasattr(stream, "response")`` is false.
    """


def _no_response() -> NoReturn:
    raise NoHTTPResponseError(
        "this stream was rebuilt from a recorded Message (agent-replay delivers recorded "
        "and replayed streams this way, and the fake model streams so too), so it has no "
        "HTTP response, headers or request id; read the message's id from the stream's "
        "events or get_final_message() instead"
    )


class _RebuiltRequest:
    def __init__(self, headers: httpx2.Headers) -> None:
        self.headers = headers


class RebuiltResponse:
    """The stand-in HTTP response of a rebuilt ``beta.messages`` stream.

    The SDK's ``BetaMessageStream`` reads ``response.request.headers`` on every
    event, to see whether the ``anthropic-beta`` header asked for fine-grained
    tool streaming, so a beta stream cannot work without one. It carries the
    recorded ``betas`` as that header and nothing else: no status, no response
    headers (so ``request_id`` is ``None``).
    """

    def __init__(self, betas: Sequence[str]) -> None:
        request_headers = {"anthropic-beta": ",".join(betas)} if betas else {}
        self.request = _RebuiltRequest(httpx2.Headers(request_headers))
        self.headers = httpx2.Headers()


class SyntheticStream:
    """A finished list of events with the interface of the SDK's ``Stream``.

    ``betas`` (``None`` for the plain ``messages`` surface) gives a beta stream
    its :class:`RebuiltResponse`; a plain one has no response at all.
    """

    def __init__(self, events: Iterable[StreamEvent], betas: Sequence[str] | None = None) -> None:
        self._iterator: Iterator[StreamEvent] = iter(list(events))
        self._response = RebuiltResponse(betas) if betas is not None else None

    @property
    def response(self) -> RebuiltResponse:
        if self._response is None:
            _no_response()
        return self._response

    def __iter__(self) -> Iterator[StreamEvent]:
        return self

    def __next__(self) -> StreamEvent:
        return next(self._iterator)

    def close(self) -> None:
        self._iterator = iter(())

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncSyntheticStream:
    """A finished list of events with the interface of the SDK's ``AsyncStream``."""

    def __init__(self, events: Iterable[StreamEvent], betas: Sequence[str] | None = None) -> None:
        self._iterator: Iterator[StreamEvent] = iter(list(events))
        self._response = RebuiltResponse(betas) if betas is not None else None

    @property
    def response(self) -> RebuiltResponse:
        if self._response is None:
            _no_response()
        return self._response

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self._iterator = iter(())

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
