"""``client.beta.messages``: record and replay ``create``, ``create(stream=True)``, ``stream``.

Frameworks such as pydantic-ai call only this surface, so it is recorded under its
own call name (``anthropic.beta.messages.create``) and replays the beta types.
"""

from __future__ import annotations

import asyncio
from typing import Any

import anthropic
import pydantic_ai
import pytest
from anthropic.lib.streaming import BetaAsyncMessageStream, BetaMessageStream
from anthropic.lib.streaming._beta_messages import accumulate_event
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaRawMessageStartEvent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from agent_replay import RecordingSession, ReplaySession, tool, wrap
from agent_replay.cassette import Cassette, Kind
from agent_replay.errors import ReplayError, UnrecordedCallError
from agent_replay.fake import FakeAnthropic, FakeAsyncAnthropic, Reply, ToolCall, scripted
from agent_replay.record._base import WrappedClient
from agent_replay.record._streams import (
    BETA_MESSAGE_EVENTS,
    NoHTTPResponseError,
    SyntheticStream,
    events_from_message,
)
from agent_replay.record.anthropic import (
    ANTHROPIC,
    BETA_MESSAGES,
    BETA_MODEL_CALL,
    MODEL_CALL,
    _prepare,
)
from agent_replay.runner import RunResult, record, replay
from agent_replay.trajectory import StepStatus

MODEL = "claude-opus-5-5"
TASK = "What is the weather in Oslo?"
BETAS = ["context-management-2025-06-27"]
KW: dict[str, Any] = {"model": MODEL, "max_tokens": 64}


def _ask(task: str) -> list[BetaMessageParam]:
    return [{"role": "user", "content": task}]


def policy() -> Any:
    return scripted([Reply(text="Let me look.", tool_calls=[ToolCall("lookup", {"q": "Oslo"})])])


def _text(message: BetaMessage) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def _model_envelopes(cassette: Cassette) -> list[Any]:
    return [e for e in cassette.envelopes if e.kind is Kind.MODEL]


def _exact(result: RunResult) -> None:
    assert result.ok, result.error
    assert result.trajectory.count(StepStatus.APPROXIMATE) == 0
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}


# --- plain create -----------------------------------------------------------------


def sync_create_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    reply = client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS)
    assert isinstance(reply, BetaMessage)
    return reply.model_dump(mode="json")


async def async_create_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    reply = await client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS)
    assert isinstance(reply, BetaMessage)
    return reply.model_dump(mode="json")


@pytest.mark.parametrize(
    ("agent", "fake"),
    [(sync_create_agent, FakeAnthropic), (async_create_agent, FakeAsyncAnthropic)],
)
def test_beta_create_records_under_its_own_name_and_replays_a_beta_message(
    agent: Any, fake: Any
) -> None:
    result, cassette = record(agent, TASK, client=fake(policy()))
    assert result.ok, result.error
    [envelope] = _model_envelopes(cassette)
    assert envelope.name == BETA_MODEL_CALL == "anthropic.beta.messages.create"
    assert isinstance(envelope.request, dict)
    assert envelope.request["betas"] == BETAS
    assert envelope.usage is not None
    assert envelope.usage.output_tokens > 0

    replayed = replay(Cassette.loads(cassette.dumps()), agent)
    _exact(replayed)
    assert replayed.output == result.output
    assert BetaMessage.model_validate(replayed.output) == BetaMessage.model_validate(
        envelope.response
    )


def test_replayed_beta_create_returns_an_equal_beta_message() -> None:
    _, cassette = record(sync_create_agent, TASK, client=FakeAnthropic(policy()))
    client = wrap(FakeAnthropic(scripted([])))  # never asked: full replay serves the model
    with ReplaySession(cassette) as session:
        reply = client.beta.messages.create(**KW, messages=_ask(TASK), betas=BETAS)
    assert type(reply) is BetaMessage
    assert reply == BetaMessage.model_validate(_model_envelopes(cassette)[0].response)
    assert [s.status for s in session.trajectory.steps] == [StepStatus.EXACT]


# --- fingerprint: stream is delivery, betas is content ------------------------------


def stream_create_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    stream = client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS, stream=True)
    with stream:
        events = list(stream)
    assert isinstance(events[0], BetaRawMessageStartEvent)
    snapshot: Any = None
    bufs: dict[int, bytes] = {}
    for event in events:
        snapshot = accumulate_event(
            event=event, current_snapshot=snapshot, json_bufs=bufs, request_headers=None
        )
    return BetaMessage.model_validate(snapshot.to_dict()).model_dump(mode="json")


async def async_stream_create_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    stream = await client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS, stream=True)
    events = [event async for event in stream]
    await stream.close()
    assert isinstance(events[0], BetaRawMessageStartEvent)
    snapshot: Any = None
    bufs: dict[int, bytes] = {}
    for event in events:
        snapshot = accumulate_event(
            event=event, current_snapshot=snapshot, json_bufs=bufs, request_headers=None
        )
    return BetaMessage.model_validate(snapshot.to_dict()).model_dump(mode="json")


@pytest.mark.parametrize(
    ("plain", "streaming", "fake"),
    [
        (sync_create_agent, stream_create_agent, FakeAnthropic),
        (async_create_agent, async_stream_create_agent, FakeAsyncAnthropic),
    ],
)
def test_replayed_beta_stream_yields_beta_events_that_accumulate_to_the_message(
    plain: Any, streaming: Any, fake: Any
) -> None:
    # Recorded without streaming, replayed with stream=True: stream is not in the
    # fingerprint, so this is exact, and the rebuilt events are the beta classes.
    result, cassette = record(plain, TASK, client=fake(policy()))
    replayed = replay(cassette, streaming)
    _exact(replayed)
    assert replayed.output == result.output


def test_stream_is_not_in_the_fingerprint_but_is_recorded() -> None:
    streamed_request, streamed = _prepared_fingerprints(stream=True)
    plain_request, plain = _prepared_fingerprints(stream=False)
    assert streamed == plain
    assert isinstance(streamed_request, dict)
    assert isinstance(plain_request, dict)
    assert streamed_request["stream"] is True
    assert "stream" not in plain_request
    assert streamed_request["betas"] == plain_request["betas"] == BETAS
    _, cassette = record(sync_create_agent, TASK, client=FakeAnthropic(policy()))
    assert _model_envelopes(cassette)[0].fingerprint == plain


def _prepared_fingerprints(*, stream: bool) -> tuple[Any, str]:
    kwargs: dict[str, Any] = {**KW, "messages": _ask(TASK), "betas": BETAS}
    if stream:
        kwargs["stream"] = True
    with RecordingSession() as session:
        p = _prepare(session, kwargs, "beta.messages.create", BETA_MESSAGES)
    return p.request, p.fingerprint


@pytest.mark.parametrize(
    ("betas", "status"),
    [(BETAS, StepStatus.EXACT), (["interleaved-thinking-2025-05-14"], StepStatus.APPROXIMATE)],
)
def test_betas_are_in_the_fingerprint(betas: list[str], status: StepStatus) -> None:
    _, cassette = record(sync_create_agent, TASK, client=FakeAnthropic(policy()))

    def agent(client: anthropic.Anthropic, task: str) -> str:
        return _text(client.beta.messages.create(**KW, messages=_ask(task), betas=betas))

    replayed = replay(cassette, agent)
    # Different betas are a different request: the session serves the only
    # recording of this call approximately (strict still serves it; ``replay ci
    # --strict`` / fail_on_approximate is what rejects it).
    assert replayed.ok, replayed.error
    assert [s.status for s in replayed.trajectory.steps] == [status]


def test_plain_and_beta_recordings_do_not_cross_match() -> None:
    def plain(client: anthropic.Anthropic, task: str) -> str:
        reply = client.messages.create(**KW, messages=[{"role": "user", "content": task}])
        return "".join(b.text for b in reply.content if b.type == "text")

    def beta(client: anthropic.Anthropic, task: str) -> str:
        return _text(client.beta.messages.create(**KW, messages=_ask(task)))

    _, plain_cassette = record(plain, TASK, client=FakeAnthropic(policy()))
    assert _model_envelopes(plain_cassette)[0].name == MODEL_CALL
    _, beta_cassette = record(beta, TASK, client=FakeAnthropic(policy()))
    assert _model_envelopes(beta_cassette)[0].name == BETA_MODEL_CALL

    for cassette, agent in ((plain_cassette, beta), (beta_cassette, plain)):
        result = replay(cassette, agent)
        assert result.error is not None
        assert result.error.type == UnrecordedCallError.__name__


# --- SDK stream accumulation (BetaMessageStream) ------------------------------------


def sync_stream_manager_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    with client.beta.messages.stream(**KW, messages=_ask(task), betas=BETAS) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert isinstance(final, BetaMessage)
    return {"text": text, "final": final.model_dump(mode="json")}


async def async_stream_manager_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    async with client.beta.messages.stream(**KW, messages=_ask(task), betas=BETAS) as stream:
        events = [event async for event in stream]
        final = await stream.get_final_message()
    assert isinstance(final, BetaMessage)
    return {"events": len(events), "final": final.model_dump(mode="json")}


def sdk_accumulating_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    raw = client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS, stream=True)
    with BetaMessageStream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = stream.get_final_message()
    return BetaMessage.model_validate(final.to_dict()).model_dump(mode="json")


async def async_sdk_accumulating_agent(
    client: anthropic.AsyncAnthropic, task: str
) -> dict[str, Any]:
    raw = await client.beta.messages.create(**KW, messages=_ask(task), betas=BETAS, stream=True)
    async with BetaAsyncMessageStream(raw, output_format=anthropic.NOT_GIVEN) as stream:
        final = await stream.get_final_message()
    return BetaMessage.model_validate(final.to_dict()).model_dump(mode="json")


def _raise_recorded_failure(result: RunResult) -> None:
    assert result.ok, result.error


@pytest.mark.parametrize(
    ("agent", "fake"),
    [
        (stream_create_agent, FakeAnthropic),
        (async_stream_create_agent, FakeAsyncAnthropic),
        (sync_stream_manager_agent, FakeAnthropic),
        (async_stream_manager_agent, FakeAsyncAnthropic),
    ],
)
def test_beta_streams_record_and_replay_exactly(agent: Any, fake: Any) -> None:
    result, cassette = record(agent, TASK, client=fake(policy()))
    _raise_recorded_failure(result)
    [envelope] = _model_envelopes(cassette)
    assert envelope.name == BETA_MODEL_CALL
    assert isinstance(envelope.request, dict)
    assert envelope.request["stream"] is True
    replayed = replay(cassette, agent)
    _raise_recorded_failure(replayed)
    _exact(replayed)
    assert replayed.output == result.output


@pytest.mark.parametrize(
    ("plain", "streaming", "fake"),
    [
        (sync_create_agent, sync_stream_manager_agent, FakeAnthropic),
        (async_create_agent, async_stream_manager_agent, FakeAsyncAnthropic),
        (sync_create_agent, sdk_accumulating_agent, FakeAnthropic),
        (async_create_agent, async_sdk_accumulating_agent, FakeAsyncAnthropic),
    ],
)
def test_replayed_beta_streams_accumulate_with_the_sdk(
    plain: Any, streaming: Any, fake: Any
) -> None:
    result, cassette = record(plain, TASK, client=fake(policy()))
    replayed = replay(cassette, streaming)
    _raise_recorded_failure(replayed)
    _exact(replayed)
    output = replayed.output
    assert isinstance(output, dict)
    final = output.get("final", output)
    assert BetaMessage.model_validate(final) == BetaMessage.model_validate(result.output)


# --- events_from_message ----------------------------------------------------------


def test_beta_events_round_trip_text_tool_use_thinking_and_context_management() -> None:
    message = BetaMessage.model_validate(
        {
            "id": "msg_beta_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [
                {"type": "thinking", "thinking": "Need the weather.", "signature": "sig=="},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "Oslo"}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "context_management": {
                "applied_edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "cleared_input_tokens": 120,
                        "cleared_tool_uses": 2,
                    }
                ]
            },
            "usage": {"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 7},
        }
    )
    events = events_from_message(message, BETA_MESSAGE_EVENTS)
    assert isinstance(events[0], BetaRawMessageStartEvent)
    assert all(type(e).__module__.startswith("anthropic.types.beta") for e in events)
    snapshot: Any = None
    bufs: dict[int, bytes] = {}
    for event in events:
        snapshot = accumulate_event(
            event=event,  # type: ignore[arg-type]
            current_snapshot=snapshot,
            json_bufs=bufs,
            request_headers=None,
        )
    rebuilt = BetaMessage.model_validate(snapshot.to_dict())
    assert rebuilt.context_management is not None
    assert rebuilt.model_dump(exclude_none=True) == message.model_dump(exclude_none=True)


# --- with_options -------------------------------------------------------------------


class _OptionsFake(FakeAnthropic):
    """The fake, plus ``with_options`` (returning a copy that shares the model)."""

    def __init__(self, policy: Any) -> None:
        super().__init__(policy)
        self.options: dict[str, Any] = {}

    def with_options(self, **kwargs: Any) -> _OptionsFake:
        copy = object.__new__(_OptionsFake)
        copy.__dict__.update(self.__dict__)
        copy.options = kwargs
        return copy


def test_with_options_returns_a_wrapped_client_whose_calls_are_recorded() -> None:
    real = wrap(anthropic.Anthropic(api_key="test-key"))
    for copy in (real.with_options(max_retries=0), real.copy(timeout=5.0)):
        assert isinstance(copy, WrappedClient)
        assert type(copy) is type(real)
    assert real.with_options(max_retries=0)._inner.max_retries == 0  # type: ignore[attr-defined]
    real_async = wrap(anthropic.AsyncAnthropic(api_key="test-key"))
    assert type(real_async.with_options(max_retries=0)) is type(real_async)
    assert type(real.with_middleware()) is type(real)
    assert type(real_async.with_middleware()) is type(real_async)

    client = wrap(_OptionsFake(policy()))
    with RecordingSession() as session:
        copy = client.with_options(max_retries=0)
        assert copy._inner.options == {"max_retries": 0}  # type: ignore[attr-defined]
        copy.beta.messages.create(**KW, messages=_ask(TASK))
    [envelope] = session.finish().envelopes
    assert envelope.name == BETA_MODEL_CALL


# --- offline stand-in -----------------------------------------------------------------


async def two_call_agent(client: anthropic.AsyncAnthropic, task: str) -> str:
    assert client.base_url  # read by frameworks; the stand-in must have one
    first = await client.beta.messages.create(**KW, messages=_ask(task))
    if task != TASK:
        return _text(first)
    second = await client.beta.messages.create(**KW, messages=_ask("unrecorded"))
    return _text(second)


def test_full_replay_of_an_async_beta_agent_uses_the_offline_stand_in() -> None:
    once = scripted([Reply(text="Sunny.")])
    result, cassette = record(two_call_agent, "only once", client=FakeAsyncAnthropic(once))
    assert result.ok, result.error

    replayed = replay(cassette, two_call_agent)  # no client: the offline stand-in
    _exact(replayed)
    assert replayed.output == "Sunny."

    unrecorded = replay(cassette, two_call_agent, task=TASK, policy="fallback")
    assert unrecorded.error is not None
    assert "no live model client" in unrecorded.error.message


def test_offline_stand_ins_expose_client_options_and_refuse_beta_calls() -> None:
    for is_async in (False, True):
        offline = ANTHROPIC.offline_client(is_async=is_async)
        assert offline.base_url == "https://api.anthropic.com"  # type: ignore[attr-defined]
        assert offline.timeout is not None  # type: ignore[attr-defined]
    _, cassette = record(sync_create_agent, TASK, client=FakeAnthropic(policy()))
    client = wrap(ANTHROPIC.offline_client(is_async=False))
    with ReplaySession(cassette, policy="fallback"):
        client.beta.messages.create(**KW, messages=_ask(TASK), betas=BETAS)
        with pytest.raises(ReplayError, match="no live model client"):
            client.beta.messages.create(**KW, messages=_ask("unrecorded"), betas=BETAS)


# --- pydantic-ai ----------------------------------------------------------------------


@tool
def weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 12C and sunny"


def _pai_agent(client: Any) -> Any:
    model = AnthropicModel("fake-model", provider=AnthropicProvider(anthropic_client=client))
    return pydantic_ai.Agent(model, tools=[weather])


async def pai_run(client: anthropic.AsyncAnthropic, task: str) -> str:
    result = await _pai_agent(client).run(task)
    return str(result.output)


async def pai_run_stream(client: anthropic.AsyncAnthropic, task: str) -> str:
    async with _pai_agent(client).run_stream(task) as result:
        return str(await result.get_output())


def pai_policy() -> Any:
    return scripted(
        [
            Reply(tool_calls=[ToolCall("weather", {"city": "Oslo"})]),
            Reply(text="It is 12C and sunny in Oslo."),
        ]
    )


def _check_pai_cassette(cassette: Cassette) -> None:
    assert [e.name for e in cassette.envelopes] == [BETA_MODEL_CALL, "weather", BETA_MODEL_CALL]


def test_pydantic_ai_agent_run_records_and_replays_offline() -> None:
    result, cassette = record(pai_run, TASK, client=FakeAsyncAnthropic(pai_policy()))
    assert result.ok, result.error
    assert result.output == "It is 12C and sunny in Oslo."
    _check_pai_cassette(cassette)

    replayed = replay(Cassette.loads(cassette.dumps()), pai_run)  # no client
    _exact(replayed)
    assert replayed.output == result.output


def test_pydantic_ai_run_stream_replays_a_recording_made_without_streaming() -> None:
    result, cassette = record(pai_run, TASK, client=FakeAsyncAnthropic(pai_policy()))
    replayed = replay(cassette, pai_run_stream)  # stream is not in the fingerprint
    _exact(replayed)
    assert replayed.output == result.output


def test_pydantic_ai_agent_run_stream_records_and_replays_offline() -> None:
    result, cassette = record(pai_run_stream, TASK, client=FakeAsyncAnthropic(pai_policy()))
    _raise_recorded_failure(result)
    assert result.output == "It is 12C and sunny in Oslo."
    _check_pai_cassette(cassette)
    replayed = replay(cassette, pai_run_stream)
    _exact(replayed)
    assert replayed.output == result.output


def test_async_beta_count_tokens_names_the_fix() -> None:
    client = wrap(FakeAsyncAnthropic(pai_policy()))

    async def go() -> None:
        with (
            RecordingSession(),
            pytest.raises(ReplayError, match="token counting is not supported"),
        ):
            await client.beta.messages.count_tokens(model=MODEL, messages=_ask(TASK))

    asyncio.run(go())


def test_a_rebuilt_beta_stream_reports_its_betas_and_a_plain_one_has_no_response() -> None:
    # The SDK's BetaMessageStream reads response.request.headers on every event (for the
    # fine-grained tool streaming beta); a plain stream keeps refusing .response.
    message = BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    beta = SyntheticStream(events_from_message(message, BETA_MESSAGE_EVENTS), BETAS)
    assert beta.response.request.headers["anthropic-beta"] == ",".join(BETAS)
    assert beta.response.headers.get("request-id") is None
    assert "anthropic-beta" not in SyntheticStream([], []).response.request.headers
    with pytest.raises(NoHTTPResponseError):
        _ = SyntheticStream([]).response
