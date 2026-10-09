"""Async clients, streaming, ``messages.parse`` and async tools: record, then replay."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, MessageParam, ToolResultBlockParam, ToolUseBlock

from agent_replay.cassette import Cassette, Kind
from agent_replay.errors import ReplayError, UnrecordedCallError
from agent_replay.fake import FakeAnthropic, FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.record import wrap
from agent_replay.record.anthropic import ANTHROPIC
from agent_replay.record.tools import tool
from agent_replay.runner import record, replay
from agent_replay.session import ReplaySession
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


async def run_tool(block: ToolUseBlock) -> ToolResultBlockParam:
    result = weather(**block.input)
    return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}


async def async_agent(client: anthropic.AsyncAnthropic, task: str) -> str:
    messages: list[MessageParam] = [{"role": "user", "content": task}]
    for _ in range(4):
        response = await client.messages.create(
            model=MODEL, max_tokens=256, messages=messages, tools=[weather.to_anthropic()]
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
