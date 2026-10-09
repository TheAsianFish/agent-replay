"""Async clients, streaming, ``messages.parse`` and async tools: record, then replay."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import anthropic
import pytest
from anthropic.lib._parse._transform import transform_schema
from anthropic.lib.streaming._messages import accumulate_event
from anthropic.types import (
    Message,
    MessageParam,
    ParsedMessage,
    ToolResultBlockParam,
    ToolUseBlock,
)
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from agent_replay._json import canonical_json
from agent_replay.cassette import Cassette, Kind
from agent_replay.errors import ReplayError, UnrecordedCallError
from agent_replay.fake import FakeAnthropic, FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.record import wrap
from agent_replay.record._streams import events_from_message
from agent_replay.record.anthropic import ANTHROPIC
from agent_replay.record.tools import tool
from agent_replay.runner import record, replay
from agent_replay.session import RecordingSession, ReplaySession
from agent_replay.trajectory import StepStatus

MODEL = "claude-opus-5-5"
TASK = "What is the weather in Oslo and Lima?"


@tool
def weather(city: str) -> dict[str, Any]:
    """Current weather for a city."""
    return {"city": city, "temp_c": len(city) * 3}


def weather_policy(request: FakeRequest) -> Reply:
    results = request.tool_results()
    if not results:
        return Reply(
            text="Checking both cities.",
            tool_calls=[
                ToolCall("weather", {"city": "Oslo"}),
                ToolCall("weather", {"city": "Lima"}),
            ],
        )
    temps = ", ".join(
        f"{json.loads(r.content)['city']} {json.loads(r.content)['temp_c']}C" for r in results
    )
    return Reply(text=f"Now: {temps}.")


class Weather(BaseModel):
    city: str
    temp_c: int


@tool(name="weather")
async def aweather(city: str) -> Weather:
    """Current weather for a city (async)."""
    for _ in range(3 if city == "Oslo" else 1):  # yield more often: finish out of start order
        await asyncio.sleep(0)
    return Weather(city=city, temp_c=len(city) * 3)


async def run_tool(block: ToolUseBlock) -> ToolResultBlockParam:
    result = await aweather(**block.input)
    assert isinstance(result, Weather)  # rebuilt from JSON on replay
    content = json.dumps(result.model_dump())
    return {"type": "tool_result", "tool_use_id": block.id, "content": content}


async def async_agent(client: anthropic.AsyncAnthropic, task: str) -> str:
    messages: list[MessageParam] = [{"role": "user", "content": task}]
    for _ in range(4):
        response = await client.messages.create(
            model=MODEL, max_tokens=256, messages=messages, tools=[aweather.to_anthropic()]
        )
        messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if not uses:
            return "".join(b.text for b in response.content if b.type == "text")
        results = await asyncio.gather(*(run_tool(b) for b in uses))
        messages.append({"role": "user", "content": list(results)})
    raise AssertionError("agent did not finish")


def fake_async() -> anthropic.AsyncAnthropic:
    return FakeAsyncAnthropic(weather_policy).as_anthropic()


def statuses(result: Any) -> set[StepStatus]:
    return {s.status for s in result.trajectory.steps}


# --- async clients ------------------------------------------------------------------


def test_async_client_is_wrapped_and_passes_through_outside_a_session() -> None:
    client = wrap(fake_async())
    assert asyncio.iscoroutinefunction(client.messages.create)
    reply = asyncio.run(
        client.messages.create(
            model=MODEL, max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )
    )
    assert isinstance(reply, Message)


def test_real_async_client_is_accepted_and_context_managers_pass_through() -> None:
    client = wrap(anthropic.AsyncAnthropic(api_key="test-key"))
    assert wrap(client) is client

    async def use() -> None:
        async with client as entered:
            assert entered is client

    asyncio.run(use())


def test_async_agent_records_and_replays_exactly() -> None:
    result, cassette = record(async_agent, TASK, client=fake_async())
    assert result.ok, result.error
    assert result.output == "Now: Oslo 12C, Lima 12C."
    assert [e.kind for e in cassette.envelopes].count(Kind.MODEL) == 2
    model = next(e for e in cassette.envelopes if e.kind is Kind.MODEL)
    assert model.usage is not None
    assert model.usage.input_tokens > 0

    replayed = replay(Cassette.loads(cassette.dumps()), async_agent)
    assert replayed.ok, replayed.error
    assert replayed.output == result.output
    assert statuses(replayed) == {StepStatus.EXACT}


def test_async_and_sync_fakes_account_usage_identically() -> None:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Lima?"}],
    }
    sync = FakeAnthropic(weather_policy).messages.create(**kwargs)
    async_ = asyncio.run(FakeAsyncAnthropic(weather_policy).messages.create(**kwargs))
    assert sync.model_dump() == async_.model_dump()


def test_model_live_replay_with_an_async_client() -> None:
    _, cassette = record(async_agent, TASK, client=fake_async())
    replayed = replay(cassette, async_agent, mode="model-live", client=fake_async())
    assert replayed.ok, replayed.error
    by_kind = {s.kind: s.status for s in replayed.trajectory.steps}
    assert by_kind == {Kind.MODEL: StepStatus.LIVE, Kind.TOOL: StepStatus.EXACT}


def _one_call_cassette() -> Cassette:
    async def once(client: anthropic.AsyncAnthropic, task: str) -> str:
        reply = await client.messages.create(
            model=MODEL, max_tokens=8, messages=[{"role": "user", "content": task}]
        )
        return reply.id

    return record(once, "hi", client=FakeAsyncAnthropic(lambda _: Reply(text="hey")))[1]


async def _create(client: Any, text: str) -> Message:
    reply: Message = await client.messages.create(
        model=MODEL, max_tokens=8, messages=[{"role": "user", "content": text}]
    )
    return reply


def test_unrecorded_async_call_fails_under_strict() -> None:
    client = wrap(FakeAsyncAnthropic(lambda _: Reply(text="live")))
    with ReplaySession(_one_call_cassette()) as session:
        asyncio.run(_create(client, "hi"))
        with pytest.raises(UnrecordedCallError):
            asyncio.run(_create(client, "again"))
    assert session.fatal is not None


def test_unrecorded_async_call_runs_under_fallback() -> None:
    client = wrap(FakeAsyncAnthropic(lambda _: Reply(text="live")))
    with ReplaySession(_one_call_cassette(), policy="fallback") as session:
        asyncio.run(_create(client, "hi"))
        reply = asyncio.run(_create(client, "again"))
    assert reply.content[0].type == "text"
    assert reply.content[0].text == "live"
    assert [s.status for s in session.trajectory.steps] == [StepStatus.EXACT, StepStatus.FALLBACK]


def test_async_offline_client_names_the_fix() -> None:
    offline = ANTHROPIC.offline_client(is_async=True)
    client = wrap(offline)
    with ReplaySession(_one_call_cassette(), policy="fallback"):
        asyncio.run(_create(client, "hi"))
        with pytest.raises(ReplayError, match="no live model client"):
            asyncio.run(_create(client, "unrecorded"))


def test_async_agent_full_replay_uses_the_async_offline_client() -> None:
    cassette = _one_call_cassette()

    async def twice(client: anthropic.AsyncAnthropic, task: str) -> str:
        await _create(client, task)
        reply = await _create(client, "more")
        return reply.id

    result = replay(cassette, twice, policy="fallback")
    assert result.error is not None
    assert "no live model client" in result.error.message


# --- async tools --------------------------------------------------------------------


@tool
async def report(cities: list[str]) -> list[Weather]:
    """Weather for several cities, fetched concurrently."""
    return list(await asyncio.gather(*(aweather(c) for c in cities)))


async def report_agent(client: anthropic.AsyncAnthropic, task: str) -> list[Any]:
    reports = await asyncio.gather(report(["Oslo", "Lima"]), report(["Rome"]))
    return [[w.model_dump() for w in r] for r in reports]


def test_async_tool_is_a_coroutine_function_and_passes_through() -> None:
    assert inspect.iscoroutinefunction(aweather)
    assert aweather.is_async
    assert not weather.is_async
    pending = aweather("Rome")
    assert inspect.iscoroutine(pending)
    assert asyncio.run(pending) == Weather(city="Rome", temp_c=12)


def test_async_tools_record_with_types_and_replay() -> None:
    result, cassette = record(async_agent, TASK, client=fake_async())
    tools = [e for e in cassette.envelopes if e.kind is Kind.TOOL]
    assert [e.response for e in tools] == [
        {"city": "Lima", "temp_c": 12},  # Oslo sleeps, so Lima finishes first
        {"city": "Oslo", "temp_c": 12},
    ]
    replayed = replay(cassette, async_agent)
    assert replayed.output == result.output
    assert replayed.trajectory.tool_names() == ["weather", "weather"]


def test_nested_async_tools_keep_their_parent_across_gather() -> None:
    result, cassette = record(report_agent, "report", client=fake_async())
    assert result.ok, result.error
    spans = {canonical_json(e.request): e.span_id for e in cassette.envelopes if e.name == "report"}
    parents = {
        e.request["city"]: e.parent_span_id
        for e in cassette.envelopes
        if e.name == "weather" and isinstance(e.request, dict)
    }
    oslo_lima = spans[canonical_json({"cities": ["Oslo", "Lima"]})]
    assert parents == {
        "Oslo": oslo_lima,
        "Lima": oslo_lima,
        "Rome": spans[canonical_json({"cities": ["Rome"]})],
    }
    assert all(e.parent_span_id is None for e in cassette.envelopes if e.name == "report")

    full = replay(cassette, report_agent)
    assert full.ok, full.error
    assert full.output == result.output
    assert full.trajectory.tool_names() == ["report", "report"]  # served whole
    assert statuses(full) == {StepStatus.EXACT}


def test_unrecorded_async_tool_follows_the_policy() -> None:
    _, cassette = record(report_agent, "report", client=fake_async())

    async def more(client: anthropic.AsyncAnthropic, task: str) -> list[Any]:
        return [*(await report_agent(client, task)), (await aweather("Kyiv")).model_dump()]

    strict = replay(cassette, more)
    assert strict.error is not None
    assert strict.error.type == "UnrecordedCallError"
    synth = replay(cassette, more, policy="synth")
    assert synth.ok, synth.error
    assert isinstance(synth.output, list)
    assert synth.output[-1] == {"city": "", "temp_c": 0}
    fallback = replay(cassette, more, policy="fallback")
    assert isinstance(fallback.output, list)
    assert fallback.output[-1] == {"city": "Kyiv", "temp_c": 12}


# --- streaming ----------------------------------------------------------------------

STREAM_KW: dict[str, Any] = {"model": MODEL, "max_tokens": 64}


def stream_policy(request: FakeRequest) -> Reply:
    if request.tool_results():
        return Reply(text="Sunny.")
    return Reply(text="Let me look.", tool_calls=[ToolCall("weather", {"city": "Oslo"})])


def _events(events: list[Any]) -> list[Any]:
    return [e.model_dump(mode="json", exclude_none=True) for e in events]


def _ask(task: str) -> list[MessageParam]:
    return [{"role": "user", "content": task}]


def sync_create_stream_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    stream = client.messages.create(**STREAM_KW, messages=_ask(task), stream=True)
    with stream:
        events = list(stream)
    return {"events": _events(events), "weather": weather("Oslo")}


def sync_stream_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    with client.messages.stream(**STREAM_KW, messages=_ask(task)) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    return {"text": text, "final": final.model_dump(mode="json", exclude_none=True)}


async def async_create_stream_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    stream = await client.messages.create(**STREAM_KW, messages=_ask(task), stream=True)
    events = [event async for event in stream]
    await stream.close()
    return {"events": _events(events)}


async def async_stream_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    async with client.messages.stream(**STREAM_KW, messages=_ask(task)) as stream:
        events = [event async for event in stream]
        text = await stream.get_final_text()
    return {"events": _events(events), "text": text}


def sync_plain_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    reply = client.messages.create(**STREAM_KW, messages=_ask(task))
    return {"text": "".join(b.text for b in reply.content if b.type == "text")}


@pytest.mark.parametrize(
    ("agent", "fake"),
    [
        (sync_create_stream_agent, FakeAnthropic),
        (sync_stream_agent, FakeAnthropic),
        (async_create_stream_agent, FakeAsyncAnthropic),
        (async_stream_agent, FakeAsyncAnthropic),
    ],
)
def test_streams_record_and_replay_exactly(agent: Any, fake: Any) -> None:
    result, cassette = record(agent, TASK, client=fake(stream_policy))
    assert result.ok, result.error
    model = next(e for e in cassette.envelopes if e.kind is Kind.MODEL)
    assert isinstance(model.request, dict)
    assert model.request["stream"] is True
    assert isinstance(model.response, dict)
    assert model.response["content"][0] == {"type": "text", "text": "Let me look."}
    assert model.usage is not None
    assert model.usage.output_tokens > 0

    replayed = replay(Cassette.loads(cassette.dumps()), agent)
    assert replayed.ok, replayed.error
    # The replayed stream yields exactly the events the recorded run saw.
    assert replayed.output == result.output
    assert statuses(replayed) == {StepStatus.EXACT}


def test_stream_text_and_final_message_match_the_reply() -> None:
    result, _ = record(sync_stream_agent, TASK, client=FakeAnthropic(stream_policy))
    assert isinstance(result.output, dict)
    assert result.output["text"] == "Let me look."
    final = result.output["final"]
    assert isinstance(final, dict)
    assert final["content"][1]["input"] == {"city": "Oslo"}
    assert final["stop_reason"] == "tool_use"


def test_streamed_recording_replays_exact_without_streaming() -> None:
    _, cassette = record(sync_stream_agent, TASK, client=FakeAnthropic(stream_policy))
    replayed = replay(cassette, sync_plain_agent)
    assert replayed.ok, replayed.error
    assert replayed.output == {"text": "Let me look."}
    assert statuses(replayed) == {StepStatus.EXACT}


def test_stream_is_not_requested_until_entered() -> None:
    client = wrap(FakeAnthropic(stream_policy))
    with RecordingSession() as session:
        client.messages.stream(**STREAM_KW, messages=_ask(TASK))
    assert session.finish().envelopes == ()


def test_replayed_stream_has_no_http_response() -> None:
    _, cassette = record(sync_stream_agent, TASK, client=FakeAnthropic(stream_policy))
    client = wrap(FakeAnthropic(stream_policy))
    with ReplaySession(cassette), client.messages.stream(**STREAM_KW, messages=_ask(TASK)) as s:
        assert not hasattr(s, "response")
        with pytest.raises(ReplayError, match="no HTTP response"):
            _ = s.request_id


_TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=40)
_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**53), 2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | _TEXT,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(_TEXT, inner, max_size=4),
    max_leaves=12,
)
_BLOCK = st.one_of(
    st.builds(lambda t: {"type": "text", "text": t}, _TEXT),
    st.builds(
        lambda i, n, inp: {"type": "tool_use", "id": f"toolu_{i}", "name": n, "input": inp},
        st.integers(0, 99),
        st.text("abcdefgh_", min_size=1, max_size=8),
        st.dictionaries(_TEXT, _JSON, max_size=4),
    ),
    st.builds(lambda t, s: {"type": "thinking", "thinking": t, "signature": s}, _TEXT, _TEXT),
    st.builds(lambda d: {"type": "redacted_thinking", "data": d}, _TEXT),
)


@given(
    content=st.lists(_BLOCK, max_size=5),
    stop=st.sampled_from(["end_turn", "tool_use", "max_tokens"]),
    usage=st.tuples(st.integers(0, 10**6), st.integers(0, 10**6), st.integers(0, 10**4)),
)
def test_events_from_message_accumulate_back_to_the_message(
    content: list[dict[str, Any]], stop: str, usage: tuple[int, int, int]
) -> None:
    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "cache_read_input_tokens": usage[2],
            },
        }
    )
    snapshot = None
    json_bufs: dict[int, bytes] = {}
    for event in events_from_message(message):
        snapshot = accumulate_event(event=event, current_snapshot=snapshot, json_bufs=json_bufs)
    assert snapshot is not None
    assert snapshot.model_dump(exclude_none=True) == message.model_dump(exclude_none=True)


def test_streams_pass_through_outside_a_session() -> None:
    client = wrap(FakeAnthropic(stream_policy))
    with client.messages.stream(**STREAM_KW, messages=_ask(TASK)) as stream:
        assert stream.get_final_text() == "Let me look."


# --- messages.parse -----------------------------------------------------------------


class Forecast(BaseModel):
    city: str
    temp_c: int


def forecast_policy(_: FakeRequest) -> Reply:
    return Reply(text='{"city": "Oslo", "temp_c": 4}')


def sync_parse_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    reply = client.messages.parse(**STREAM_KW, messages=_ask(task), output_format=Forecast)
    assert isinstance(reply, ParsedMessage)
    assert isinstance(reply.parsed_output, Forecast)
    return reply.parsed_output.model_dump()


async def async_parse_agent(client: anthropic.AsyncAnthropic, task: str) -> dict[str, Any]:
    reply = await client.messages.parse(**STREAM_KW, messages=_ask(task), output_format=Forecast)
    assert isinstance(reply.parsed_output, Forecast)
    return reply.parsed_output.model_dump()


def stream_parse_agent(client: anthropic.Anthropic, task: str) -> dict[str, Any]:
    with client.messages.stream(**STREAM_KW, messages=_ask(task), output_format=Forecast) as s:
        parsed = s.get_final_message().parsed_output
    assert isinstance(parsed, Forecast)
    return parsed.model_dump()


def config_agent(client: anthropic.Anthropic, task: str) -> str:
    """The same request as ``sync_parse_agent``, spelled as a raw ``output_config``."""
    schema = transform_schema(Forecast.model_json_schema())
    reply = client.messages.create(
        **STREAM_KW,
        messages=_ask(task),
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return "".join(b.text for b in reply.content if b.type == "text")


@pytest.mark.parametrize(
    ("agent", "fake"),
    [
        (sync_parse_agent, FakeAnthropic),
        (async_parse_agent, FakeAsyncAnthropic),
        (stream_parse_agent, FakeAnthropic),
    ],
)
def test_parse_records_the_wire_request_and_replays_a_parsed_message(agent: Any, fake: Any) -> None:
    result, cassette = record(agent, TASK, client=fake(forecast_policy))
    assert result.ok, result.error
    assert result.output == {"city": "Oslo", "temp_c": 4}
    [model] = cassette.envelopes
    assert model.name == "anthropic.messages.create"
    assert isinstance(model.request, dict)
    assert "output_format" not in model.request
    assert model.request["output_config"] == {
        "format": {"type": "json_schema", "schema": transform_schema(Forecast.model_json_schema())}
    }

    replayed = replay(Cassette.loads(cassette.dumps()), agent)
    assert replayed.ok, replayed.error
    assert replayed.output == result.output
    assert statuses(replayed) == {StepStatus.EXACT}


def test_parse_recording_matches_the_equivalent_output_config_exactly() -> None:
    _, cassette = record(sync_parse_agent, TASK, client=FakeAnthropic(forecast_policy))
    replayed = replay(cassette, config_agent)
    assert replayed.ok, replayed.error
    assert statuses(replayed) == {StepStatus.EXACT}


def test_parse_failure_is_recorded_with_its_message_and_replays_the_same() -> None:
    def bad(_: FakeRequest) -> Reply:
        return Reply(text="not json")

    result, cassette = record(sync_parse_agent, TASK, client=FakeAnthropic(bad))
    assert result.error is not None
    [model] = cassette.envelopes
    assert model.error is None  # the model answered; parsing it failed in the agent
    replayed = replay(cassette, sync_parse_agent)
    assert replayed.error is not None
    assert replayed.error.type == result.error.type
