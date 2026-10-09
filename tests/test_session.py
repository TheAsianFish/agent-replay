"""Tests for agent_replay.session, record.anthropic and record.tools."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any

import pytest
from anthropic.types import Message
from pydantic import BaseModel

from agent_replay.cassette import Cassette, Kind, Usage
from agent_replay.errors import RecordedException, ReplayError, UnrecordedCallError
from agent_replay.pricing import cost_usd
from agent_replay.record.anthropic import wrap
from agent_replay.record.tools import Tool, tool
from agent_replay.session import RecordingSession, ReplaySession, current_session
from agent_replay.trajectory import StepStatus

CLOCK = datetime(2026, 10, 9, tzinfo=UTC)


class Order(BaseModel):
    id: str
    total: int


def make_message(text: str = "hello", model: str = "claude-opus-5-5") -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }
    )


class StubMessages:
    def __init__(self, reply: Message) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return self.reply


class StubClient:
    def __init__(self, reply: Message | None = None) -> None:
        self.messages = StubMessages(reply or make_message())
        self.api_key = "secret"


def recorder() -> RecordingSession:
    counter = itertools.count()
    return RecordingSession(clock=lambda: CLOCK, timer=lambda: next(counter) * 0.5)


@tool
def get_order(order_id: str) -> Order:
    """Fetch an order."""
    return Order(id=order_id, total=len(order_id))


@tool
def counts(word: str) -> dict[str, int]:
    return {c: word.count(c) for c in sorted(set(word))}


@tool
def ping(n: int = 0) -> int:
    return n + 1


@tool
def explode(why: str) -> str:
    raise KeyError(why)


@tool
def never_called(x: int) -> int:
    raise AssertionError("the real function must not run during replay")


def record_basic() -> Cassette:
    session = recorder()
    client = wrap(StubClient())
    with session:
        client.messages.create(model="claude-opus-5-5", messages=[{"role": "user", "content": "x"}])
        get_order("abc")
        ping(1)
        ping(2)
    return session.finish()


# ---------------------------------------------------------------- pass-through


def test_no_session_passes_through() -> None:
    assert current_session() is None
    assert get_order("abcd") == Order(id="abcd", total=4)
    stub = StubClient()
    reply = wrap(stub).messages.create(model="m", timeout=5)
    assert reply is stub.messages.reply
    assert stub.messages.calls == [{"model": "m", "timeout": 5}]


def test_wrap_proxies_other_attributes() -> None:
    assert wrap(StubClient()).api_key == "secret"


def test_wrap_idempotent() -> None:
    proxy = wrap(StubClient())
    assert wrap(proxy) is proxy


# -------------------------------------------------------------------- recording


def test_records_model_and_tool_envelopes() -> None:
    cassette = record_basic()
    model, order, p1, p2 = cassette.envelopes
    assert [e.seq for e in cassette.envelopes] == [0, 1, 2, 3]
    assert [e.span_id for e in cassette.envelopes] == ["s0", "s1", "s2", "s3"]
    assert model.kind is Kind.MODEL
    assert model.name == "anthropic.messages.create"
    assert model.request == {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": "x"}],
    }
    assert isinstance(model.response, dict)
    assert model.response["content"] == [{"type": "text", "text": "hello"}]
    assert order.kind is Kind.TOOL
    assert order.name == "get_order"
    assert order.request == {"order_id": "abc"}
    assert order.response == {"id": "abc", "total": 3}
    assert order.usage is None
    assert order.cost_usd is None
    assert p1.request == {"n": 1}
    assert p1.response == 2
    assert p2.response == 3
    assert model.latency_ms == 500.0


def test_usage_and_cost_known_model() -> None:
    model = record_basic().envelopes[0]
    assert model.usage is not None
    assert (model.usage.input_tokens, model.usage.output_tokens) == (1000, 500)
    assert model.cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)


def test_cost_prices_cache_reads_and_writes_at_their_own_rates() -> None:
    usage = Usage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=20_000,
        cache_creation_input_tokens=4_000,
    )
    expected = (1000 * 4 + 500 * 20 + 20_000 * 0.20 + 4_000 * 5.00) / 1e6
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(expected)
    assert cost_usd("claude-opus-5-5", Usage()) == 0.0
    assert cost_usd(None, usage) is None


def test_cost_none_for_unknown_model() -> None:
    session = recorder()
    client = wrap(StubClient(make_message(model="mystery-model")))
    with session:
        client.messages.create(model="mystery-model", messages=[])
    envelope = session.finish().envelopes[0]
    assert envelope.usage is not None
    assert envelope.cost_usd is None


def test_cost_uses_the_model_that_answered_not_the_one_requested() -> None:
    session = recorder()
    client = wrap(StubClient(make_message(model="claude-opus-5-5")))
    with session:
        client.messages.create(model="some-alias", messages=[])
    assert session.finish().envelopes[0].cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)


def test_trajectory_steps_are_live() -> None:
    session = recorder()
    with session:
        ping()
        counts("aab")
    steps = session.trajectory.steps
    assert [s.status for s in steps] == [StepStatus.LIVE, StepStatus.LIVE]
    assert session.trajectory.tool_names() == ["ping", "counts"]
    assert steps[0].request == {"n": 0}  # defaults are applied
    assert [s.envelope_seq for s in steps] == [0, 1]


def test_transport_kwargs_not_recorded() -> None:
    session = recorder()
    stub = StubClient()
    client = wrap(stub)
    with session:
        client.messages.create(
            model="claude-opus-5-5",
            messages=[],
            extra_headers={"x-api-key": "secret"},
            timeout=3,
        )
    envelope = session.finish().envelopes[0]
    assert envelope.request == {"model": "claude-opus-5-5", "messages": []}
    assert "secret" not in envelope.model_dump_json()
    # The real call still received them.
    assert stub.messages.calls[0]["timeout"] == 3
    assert stub.messages.calls[0]["extra_headers"] == {"x-api-key": "secret"}


def test_metadata_recorded_but_not_fingerprinted() -> None:
    session = recorder()
    client = wrap(StubClient())
    with session:
        client.messages.create(model="m", messages=[], metadata={"user_id": "1"})
        client.messages.create(model="m", messages=[], metadata={"user_id": "2"})
        client.messages.create(model="m", messages=[], max_tokens=5)
    a, b, c = session.finish().envelopes
    assert a.request == {"model": "m", "messages": [], "metadata": {"user_id": "1"}}
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_tool_exception_recorded_and_propagates() -> None:
    session = recorder()
    with session, pytest.raises(KeyError):
        explode("bad")
    envelope = session.finish().envelopes[0]
    assert envelope.error is not None
    assert envelope.error.type == "KeyError"
    assert envelope.error.message == "'bad'"
    assert envelope.response is None
    assert envelope.usage is None
    assert session.trajectory.steps[0].error == envelope.error


def test_replay_raises_recorded_exception() -> None:
    session = recorder()
    with session, pytest.raises(KeyError):
        explode("bad")
    cassette = session.finish()
    with ReplaySession(cassette), pytest.raises(RecordedException) as info:
        explode("bad")
    assert info.value.type_name == "KeyError"
    assert info.value.message == "'bad'"


def test_stream_rejected_in_session() -> None:
    client = wrap(StubClient())
    with recorder(), pytest.raises(ReplayError, match="stream"):
        client.messages.create(model="m", messages=[], stream=True)
    with ReplaySession(record_basic()), pytest.raises(ReplayError, match="stream"):
        client.messages.create(model="m", messages=[], stream=True)


# ----------------------------------------------------------------------- replay


def test_replay_serves_without_calling_real_things() -> None:
    session = recorder()
    with session:
        never_called_recorded = Tool(lambda x: x + 1, name="never_called")
        assert never_called_recorded(1) == 2
    cassette = session.finish()

    stub = StubClient()
    with ReplaySession(cassette) as replay:
        assert never_called(1) == 2
    assert stub.messages.calls == []
    assert [s.status for s in replay.trajectory.steps] == [StepStatus.EXACT]


def test_replay_model_call_does_not_hit_inner_client() -> None:
    cassette = record_basic()
    stub = StubClient(make_message("different"))
    client = wrap(stub)
    with ReplaySession(cassette) as replay:
        reply = client.messages.create(
            model="claude-opus-5-5", messages=[{"role": "user", "content": "x"}]
        )
    assert stub.messages.calls == []
    assert isinstance(reply, Message)
    assert reply.content[0].model_dump()["text"] == "hello"
    assert replay.trajectory.steps[0].status is StepStatus.EXACT


def test_replay_exact_vs_approximate() -> None:
    cassette = record_basic()
    with ReplaySession(cassette) as replay:
        assert get_order("abc") == Order(id="abc", total=3)
        assert ping(1) == 2  # same args
        assert ping(99) == 3  # different args: still served the 2nd recorded result
    statuses = [s.status for s in replay.trajectory.steps]
    assert statuses == [StepStatus.EXACT, StepStatus.EXACT, StepStatus.APPROXIMATE]
    last = replay.trajectory.steps[-1]
    assert last.request == {"n": 99}
    assert last.envelope_seq == 3


def test_replay_per_name_cursors_when_interleaved() -> None:
    session = recorder()
    with session:
        ping(10)
        counts("a")
        ping(20)
        counts("bb")
        ping(30)
    cassette = session.finish()
    with ReplaySession(cassette):
        # A different interleaving than recorded.
        assert counts("a") == {"a": 1}
        assert ping(10) == 11
        assert ping(20) == 21
        assert counts("bb") == {"b": 2}
        assert ping(30) == 31


def test_unrecorded_extra_call_and_unknown_tool() -> None:
    cassette = record_basic()
    with ReplaySession(cassette) as replay:
        get_order("abc")
        with pytest.raises(UnrecordedCallError) as extra:
            get_order("abc")
        with pytest.raises(UnrecordedCallError) as unknown:
            counts("zzz")
    assert extra.value.name == "get_order"
    assert extra.value.kind == "tool"
    assert "replay record" in str(extra.value)
    assert unknown.value.name == "counts"
    steps = replay.trajectory.steps
    assert [s.status for s in steps] == [
        StepStatus.EXACT,
        StepStatus.UNRECORDED,
        StepStatus.UNRECORDED,
    ]
    assert steps[1].envelope_seq is None
    assert steps[2].request == {"word": "zzz"}


def test_unconsumed_in_seq_order() -> None:
    cassette = record_basic()
    with ReplaySession(cassette) as replay:
        assert [e.seq for e in replay.unconsumed()] == [0, 1, 2, 3]
        ping(1)
        assert [e.seq for e in replay.unconsumed()] == [0, 1, 3]
        get_order("abc")
        assert [e.name for e in replay.unconsumed()] == ["anthropic.messages.create", "ping"]
        ping(2)
        wrap(StubClient()).messages.create(model="claude-opus-5-5", messages=[])
        assert replay.unconsumed() == []


def test_replay_rebuilds_return_types() -> None:
    session = recorder()
    with session:
        live_order = get_order("xy")
        live_counts = counts("hello")
    cassette = session.finish()
    with ReplaySession(cassette):
        replayed_order = get_order("xy")
        replayed_counts = counts("hello")
    assert isinstance(live_order, Order)
    assert isinstance(replayed_order, Order)
    assert replayed_order == live_order
    assert isinstance(replayed_counts, dict)
    assert replayed_counts == live_counts


def test_replay_without_return_annotation_returns_json() -> None:
    @tool
    def bare(x):  # type: ignore[no-untyped-def]
        return [x, (1, 2)]

    session = recorder()
    with session:
        bare(5)
    with ReplaySession(session.finish()):
        assert bare(5) == [5, [1, 2]]


# -------------------------------------------------------------- session context


def test_nested_sessions_rejected() -> None:
    with recorder() as outer:
        assert current_session() is outer
        with pytest.raises(ReplayError, match="already active"), recorder():
            pass
        assert current_session() is outer
    assert current_session() is None


def test_context_reset_after_exit_and_on_exception() -> None:
    with recorder():
        assert current_session() is not None
    assert current_session() is None
    with pytest.raises(RuntimeError), recorder():  # noqa: PT012
        assert current_session() is not None
        raise RuntimeError("boom")
    assert current_session() is None
    with pytest.raises(RuntimeError), ReplaySession(record_basic()):
        raise RuntimeError("boom")
    assert current_session() is None
    with recorder():  # usable again
        pass


# ------------------------------------------------------------------------ Tool


def test_to_anthropic() -> None:
    @tool
    def search(query: str, limit: int = 5, tags: list[str] | None = None) -> list[str]:
        """Search the catalogue.

        Longer details.
        """
        return []

    param = search.to_anthropic()
    assert param["name"] == "search"
    assert param["description"].startswith("Search the catalogue.")
    schema = param["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"query", "limit", "tags"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"]["default"] == 5

    def has_title(node: object) -> bool:
        if isinstance(node, dict):
            return "title" in node or any(has_title(v) for v in node.values())
        if isinstance(node, list):
            return any(has_title(v) for v in node)
        return False

    assert "title" not in schema
    assert all("title" not in p for p in schema["properties"].values())
    assert not has_title(schema["properties"]["query"])


def test_tool_name_and_description_override() -> None:
    @tool(name="x", description="custom")
    def whatever(a: int) -> int:
        """Ignored docstring."""
        return a

    assert whatever.name == "x"
    assert whatever.to_anthropic()["name"] == "x"
    assert whatever.to_anthropic()["description"] == "custom"
    session = recorder()
    with session:
        assert whatever(3) == 3
    assert session.finish().envelopes[0].name == "x"


def test_tool_without_docstring_has_empty_description() -> None:
    assert ping.to_anthropic()["description"] == ""


def test_varargs_rejected() -> None:
    def star(*args: int) -> int:
        return 0

    def kw(**kwargs: int) -> int:
        return 0

    with pytest.raises(TypeError, match="not supported"):
        tool(star)
    with pytest.raises(TypeError, match="not supported"):
        tool(kw)
