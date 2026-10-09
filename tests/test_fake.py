"""Tests for agent_replay.fake."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types import Message

from agent_replay.cassette import Kind
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall, scripted
from agent_replay.record import wrap
from agent_replay.session import RecordingSession

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi there"}]


def create(client: FakeAnthropic, model: str = "claude-opus-5-5", **kw: Any) -> Message:
    return client.messages.create(model=model, max_tokens=10, messages=MESSAGES, **kw)


def test_text_only_reply_is_end_turn_message() -> None:
    msg = create(FakeAnthropic(scripted([Reply(text="hello")])))
    assert isinstance(msg, Message)
    assert msg.stop_reason == "end_turn"
    assert msg.id == "msg_fake_000"
    assert msg.model == "fake:claude-opus-5-5"
    assert [b.type for b in msg.content] == ["text"]


def test_tool_calls_give_tool_use_with_deterministic_ids() -> None:
    replies = [
        Reply(text="a", tool_calls=[ToolCall("x", {"k": 1}), ToolCall("y")]),
        Reply(tool_calls=[ToolCall("z")]),
    ]
    client = FakeAnthropic(scripted(replies))
    first = create(client)
    second = create(client)
    assert first.stop_reason == second.stop_reason == "tool_use"
    uses = [b for b in first.content if b.type == "tool_use"]
    assert [u.id for u in uses] == ["toolu_fake_000_0", "toolu_fake_000_1"]
    assert uses[0].input == {"k": 1}
    assert uses[1].input == {}
    assert second.id == "msg_fake_001"
    assert [b.type for b in second.content] == ["tool_use"]
    assert second.content[0].id == "toolu_fake_001_0"  # type: ignore[union-attr]


def test_usage_is_positive_and_deterministic() -> None:
    a = create(FakeAnthropic(scripted([Reply(text="same")])))
    b = create(FakeAnthropic(scripted([Reply(text="same")])))
    assert a.usage.input_tokens > 0
    assert a.usage.output_tokens > 0
    assert isinstance(a.usage.input_tokens, int)
    assert a.usage == b.usage


def test_scripted_returns_in_order_then_raises() -> None:
    policy = scripted([Reply(text="one"), Reply(text="two")])
    req = FakeRequest(model="m", system=None, messages=MESSAGES, tools=[])
    assert policy(req).text == "one"
    assert policy(req).text == "two"
    with pytest.raises(AssertionError, match="ran out of replies after 2 turn"):
        policy(req)


def test_stream_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        create(FakeAnthropic(scripted([Reply(text="x")])), stream=True)


def _request(messages: list[dict[str, Any]]) -> FakeRequest:
    return FakeRequest(model="m", system=None, messages=messages, tools=[])


def _conversation() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "find "}, {"type": "text", "text": "A1"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "policy", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "order-1"},
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": [{"type": "text", "text": "30 "}, {"type": "text", "text": "days"}],
                    "is_error": True,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t3", "name": "lookup", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t3", "content": "order-2"}],
        },
    ]


def test_tool_results_pairs_names_and_handles_content_shapes() -> None:
    results = _request(_conversation()).tool_results()
    assert [(r.tool_use_id, r.name, r.content, r.is_error) for r in results] == [
        ("t1", "lookup", "order-1", False),
        ("t2", "policy", "30 days", True),
        ("t3", "lookup", "order-2", False),
    ]


def test_tool_result_with_unknown_id_gets_placeholder_name() -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "zz", "content": "c"}]},
    ]
    assert _request(msgs).tool_results()[0].name == "?"


def test_latest_result_and_first_user_text() -> None:
    req = _request(_conversation())
    latest = req.latest_result("lookup")
    assert latest is not None
    assert latest.content == "order-2"
    policy = req.latest_result("policy")
    assert policy is not None
    assert policy.is_error
    assert req.latest_result("nope") is None
    assert req.first_user_text() == "find A1"
    assert _request([{"role": "user", "content": "plain"}]).first_user_text() == "plain"
    assert _request([]).first_user_text() == ""


def test_policy_sees_request_through_client() -> None:
    seen: list[FakeRequest] = []

    def policy(req: FakeRequest) -> Reply:
        seen.append(req)
        return Reply(text="ok")

    create(FakeAnthropic(policy), system="sys")
    assert seen[0].model == "claude-opus-5-5"
    assert seen[0].system == "sys"
    assert seen[0].first_user_text() == "hi there"
    assert seen[0].tools == []


def test_recording_session_records_usage_but_no_cost() -> None:
    client = FakeAnthropic(scripted([Reply(text="done")]))
    session = RecordingSession(task="t")
    with session:
        wrap(client).messages.create(model="claude-opus-5-5", max_tokens=10, messages=MESSAGES)
    cassette = session.finish(output="done", error=None)
    (env,) = cassette.envelopes
    assert env.kind is Kind.MODEL
    assert env.cost_usd is None
    assert env.usage is not None
    assert env.usage.input_tokens > 0
    assert env.usage.output_tokens > 0
