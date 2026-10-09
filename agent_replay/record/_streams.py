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
from collections.abc import AsyncIterator, Iterable, Iterator
from types import TracebackType
from typing import Any, NoReturn, Self

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

from agent_replay._json import to_jsonable
from agent_replay.errors import ReplayError

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


def events_from_message(message: Message) -> list[RawMessageStreamEvent]:
    """The raw stream events that accumulate to ``message``.

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
    events: list[RawMessageStreamEvent] = [
        RawMessageStartEvent.model_validate({"type": "message_start", "message": start})
    ]
    for index, block in enumerate(blocks):
        assert isinstance(block, dict)
        opened, deltas = _split_block(block)
        events.append(
            RawContentBlockStartEvent.model_validate(
                {"type": "content_block_start", "index": index, "content_block": opened}
            )
        )
        events.extend(
            RawContentBlockDeltaEvent.model_validate(
                {"type": "content_block_delta", "index": index, "delta": delta}
            )
            for delta in deltas
        )
        events.append(
            RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": index})
        )
    delta: dict[str, Any] = {
        "stop_reason": data.get("stop_reason"),
        "stop_sequence": data.get("stop_sequence"),
    }
    for key in ("stop_details", "container"):
        if key in data:
            delta[key] = data[key]
    events.append(
        RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": delta,
                "usage": {k: usage[k] for k in _DELTA_USAGE if k in usage},
            }
        )
    )
    events.append(RawMessageStopEvent.model_validate({"type": "message_stop"}))
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


class SyntheticStream:
    """A finished list of events with the interface of the SDK's ``Stream``."""

    def __init__(self, events: Iterable[RawMessageStreamEvent]) -> None:
        self._iterator: Iterator[RawMessageStreamEvent] = iter(list(events))

    @property
    def response(self) -> NoReturn:
        _no_response()

    def __iter__(self) -> Iterator[RawMessageStreamEvent]:
        return self

    def __next__(self) -> RawMessageStreamEvent:
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

    def __init__(self, events: Iterable[RawMessageStreamEvent]) -> None:
        self._iterator: Iterator[RawMessageStreamEvent] = iter(list(events))

    @property
    def response(self) -> NoReturn:
        _no_response()

    def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        return self

    async def __anext__(self) -> RawMessageStreamEvent:
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
