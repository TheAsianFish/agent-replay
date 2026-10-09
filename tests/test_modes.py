"""Integration tests for the replay engine: cut-point modes, matching, policies, capture."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_replay import capture
from agent_replay.cassette import Cassette, Kind
from agent_replay.checks import tool_sequence
from agent_replay.errors import RecordedException, ReplayError, UnrecordedCallError
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.record import tool, wrap
from agent_replay.runner import replay
from agent_replay.session import SERVED_KINDS, RecordingSession, ReplaySession
from agent_replay.threads import ContextThreadPoolExecutor
from agent_replay.trajectory import StepStatus
from examples.toy_agent import agent as toy
from examples.toy_agent import tools as toy_tools
from examples.toy_agent.fake_model import make_client
from examples.toy_agent.fake_model import policy as toy_policy

CASSETTES = Path(__file__).parents[1] / "examples" / "toy_agent" / "cassettes"
ELIGIBLE = "Order A1001 arrived damaged. Can I get a refund?"
CLOCK = datetime(2026, 10, 9, tzinfo=UTC)


def eligible() -> Cassette:
    return Cassette.load(CASSETTES / "refund_eligible.jsonl")


def statuses(result: Any, kind: Kind) -> set[StepStatus]:
    return {s.status for s in result.trajectory.steps if s.kind is kind}


def recorder() -> RecordingSession:
    counter = itertools.count()
    return RecordingSession(task="t", clock=lambda: CLOCK, timer=lambda: next(counter) * 0.25)


@pytest.fixture
def no_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the toy order table: a tool that really runs now fails."""
    monkeypatch.setattr(toy_tools, "ORDERS", {})


def test_served_kinds_per_mode() -> None:
    assert SERVED_KINDS["record"] == frozenset()
    assert SERVED_KINDS["full"] == frozenset(Kind)
    assert Kind.MODEL not in SERVED_KINDS["model-live"]
    assert Kind.TOOL in SERVED_KINDS["model-live"]
    assert Kind.TOOL not in SERVED_KINDS["tools-live"]
    assert Kind.MODEL in SERVED_KINDS["tools-live"]
    for mode in ("full", "model-live", "tools-live"):
        assert {Kind.CLOCK, Kind.RANDOM} <= SERVED_KINDS[mode]


@pytest.mark.parametrize(("mode", "policy"), [("partial", "strict"), ("full", "lenient")])
def test_unknown_mode_or_policy_is_rejected(mode: str, policy: str) -> None:
    with pytest.raises(ReplayError, match="unknown"):
        replay(eligible(), toy.run, mode=mode, policy=policy)  # type: ignore[arg-type]


# --- model-live ---------------------------------------------------------------


def test_model_live_runs_the_model_and_serves_tools(no_orders: None) -> None:
    result = replay(eligible(), toy.run, mode="model-live", client=make_client())

    assert result.ok, result.error
    assert result.output == eligible().footer.output
    assert statuses(result, Kind.MODEL) == {StepStatus.LIVE}
    assert statuses(result, Kind.TOOL) == {StepStatus.EXACT}  # served: ORDERS is empty
    assert result.unconsumed == ()
    assert result.check(tool_sequence(eligible().tool_names()))[0].passed


def test_model_live_needs_a_client() -> None:
    with pytest.raises(ReplayError, match=r"model-live mode .* needs a client"):
        replay(eligible(), toy.run, mode="model-live")


def forgetful_policy(request: FakeRequest) -> Reply:
    """A model regression: it no longer asks for the refund policy before refunding."""
    reply = toy_policy(request)
    if request.latest_result("lookup_order") is None:
        return Reply(text=reply.text, tool_calls=reply.tool_calls[:1])
    if request.latest_result("issue_refund") is None:
        return Reply(tool_calls=[ToolCall("issue_refund", {"order_id": "A1001", "amount": 42.5})])
    return reply


def test_model_live_catches_a_model_regression() -> None:
    client = FakeAnthropic(forgetful_policy)

    result = replay(eligible(), toy.run, mode="model-live", client=client)

    [check] = result.check(tool_sequence(eligible().tool_names()))
    assert not check.passed
    assert "missing get_refund_policy" in check.message
    assert [e.name for e in result.unconsumed] == ["get_refund_policy"]


def test_different_arguments_are_served_approximately() -> None:
    def other_order(request: FakeRequest) -> Reply:
        if request.latest_result("lookup_order") is None:
            return Reply(tool_calls=[ToolCall("lookup_order", {"order_id": "A1003"})])
        return Reply(text="ok")

    result = replay(eligible(), toy.run, mode="model-live", client=FakeAnthropic(other_order))

    # Nothing recorded A1003, so the oldest lookup_order envelope (A1001) answers, flagged.
    [lookup] = result.trajectory.tool_steps()
    assert lookup.status is StepStatus.APPROXIMATE
    assert lookup.request == {"order_id": "A1003"}
    assert result.ok


def test_second_unrecorded_lookup_fails_under_strict() -> None:
    def twice(request: FakeRequest) -> Reply:
        done = len(request.tool_results())
        if done < 2:
            order = ["A1001", "A1003"][done]
            return Reply(tool_calls=[ToolCall("lookup_order", {"order_id": order})])
        return Reply(text="done")

    result = replay(eligible(), toy.run, mode="model-live", client=FakeAnthropic(twice))

    assert not result.ok
    assert result.error is not None
    assert result.error.type == "UnrecordedCallError"
    assert "--policy fallback" in result.error.message
    steps = result.trajectory.tool_steps()
    assert [s.status for s in steps] == [StepStatus.EXACT, StepStatus.UNRECORDED]


def twice_policy(request: FakeRequest) -> Reply:
    done = len(request.tool_results())
    if done < 2:
        order = ["A1001", "A1003"][done]
        return Reply(tool_calls=[ToolCall("lookup_order", {"order_id": order})])
    return Reply(text=" | ".join(r.content for r in request.tool_results()))


def test_fallback_policy_performs_the_unrecorded_call_for_real() -> None:
    result = replay(
        eligible(),
        toy.run,
        mode="model-live",
        policy="fallback",
        client=FakeAnthropic(twice_policy),
    )

    assert result.ok, result.error
    steps = result.trajectory.tool_steps()
    assert [s.status for s in steps] == [StepStatus.EXACT, StepStatus.FALLBACK]
    assert steps[1].response == toy_tools.ORDERS["A1003"].model_dump(mode="json")
    assert '"status": "shipped"' in str(result.output)


def test_synth_policy_stubs_the_unrecorded_tool_from_its_return_type() -> None:
    result = replay(
        eligible(), toy.run, mode="model-live", policy="synth", client=FakeAnthropic(twice_policy)
    )

    assert result.ok, result.error
    steps = result.trajectory.tool_steps()
    assert [s.status for s in steps] == [StepStatus.EXACT, StepStatus.SYNTH]
    # The smallest valid Order: required fields only, first Literal member, zeros.
    assert steps[1].response == {"order_id": "", "status": "processing", "total": 0.0}


def test_saved_model_live_run_replays_in_full_mode(tmp_path: Path) -> None:
    saved = tmp_path / "model_live.jsonl"
    first = replay(eligible(), toy.run, mode="model-live", client=make_client(), save_to=saved)

    again = replay(Cassette.load(saved), toy.run)

    assert again.ok
    assert again.output == first.output
    assert {s.status for s in again.trajectory.steps} == {StepStatus.EXACT}
    header = Cassette.load(saved).header
    assert header.metadata["replay_of"] == eligible().header.cassette_id
    assert header.metadata["mode"] == "model-live"


# --- tools-live ---------------------------------------------------------------


def test_tools_live_runs_tools_against_recorded_model_replies() -> None:
    result = replay(eligible(), toy.run, mode="tools-live")

    assert result.ok, result.error
    assert result.output == eligible().footer.output
    assert statuses(result, Kind.MODEL) == {StepStatus.EXACT}
    assert statuses(result, Kind.TOOL) == {StepStatus.LIVE}
    assert all(s.latency_ms is not None for s in result.trajectory.tool_steps())


def test_tools_live_catches_a_tool_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    cheaper = toy_tools.ORDERS["A1001"].model_copy(update={"total": 10.0})
    monkeypatch.setitem(toy_tools.ORDERS, "A1001", cheaper)

    result = replay(eligible(), toy.run, mode="tools-live")

    # The recorded model still refunds 42.50; the real tool now refuses, and the
    # next recorded model reply no longer matches the conversation.
    refund = result.trajectory.tool_steps()[-1]
    assert refund.name == "issue_refund"
    assert refund.error is not None
    assert "exceeds order total" in refund.error.message
    assert StepStatus.APPROXIMATE in statuses(result, Kind.MODEL)


def extra_turn(client: Any, task: str) -> str:
    """The toy agent, plus one model call the recording never saw."""
    answer = toy.run(client, task)
    client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "?"}])
    return answer


def test_fallback_for_a_model_call_without_a_client_names_the_fix() -> None:
    result = replay(eligible(), extra_turn, mode="tools-live", policy="fallback")

    assert not result.ok
    assert result.error is not None
    assert "no live model client" in result.error.message
    assert result.trajectory.steps[-1].status is StepStatus.FALLBACK


def test_fallback_for_a_model_call_uses_the_given_client() -> None:
    result = replay(eligible(), extra_turn, policy="fallback", client=make_client())

    assert result.ok, result.error
    last = result.trajectory.steps[-1]
    assert (last.kind, last.status) == (Kind.MODEL, StepStatus.FALLBACK)


def test_synth_cannot_invent_a_model_reply() -> None:
    result = replay(eligible(), extra_turn, policy="synth")

    assert result.error is not None
    assert result.error.type == "UnrecordedCallError"
    assert "cannot invent a model response" in result.error.message


def test_tools_live_fallback_for_a_model_call_uses_the_given_client(no_orders: None) -> None:
    result = replay(
        eligible(), extra_turn, mode="tools-live", policy="fallback", client=make_client()
    )

    assert result.ok, result.error
    assert result.output == eligible().footer.output
    last = result.trajectory.steps[-1]
    assert (last.kind, last.status) == (Kind.MODEL, StepStatus.FALLBACK)
    assert statuses(result, Kind.TOOL) == {StepStatus.LIVE}  # tools ran: ORDERS is empty
    assert any(s.error is not None for s in result.trajectory.tool_steps())


def test_tools_live_synth_cannot_invent_a_model_reply() -> None:
    result = replay(eligible(), extra_turn, mode="tools-live", policy="synth")

    assert result.error is not None
    assert result.error.type == "UnrecordedCallError"
    assert "synth" in result.error.message
    assert "cannot invent a model response" in result.error.message
    assert statuses(result, Kind.TOOL) == {StepStatus.LIVE}


# --- matching -----------------------------------------------------------------


@tool
def ping(n: int) -> int:
    return n + 1


@tool
def untyped(n: int):  # type: ignore[no-untyped-def]
    return {"n": n}


def record_pings(*ns: int) -> Cassette:
    session = recorder()
    with session:
        for n in ns:
            ping(n)
    return session.finish()


def test_fingerprint_search_serves_reordered_calls_exactly() -> None:
    cassette = record_pings(1, 2)
    session = ReplaySession(cassette)
    with session:
        assert ping(2) == 3
        assert ping(1) == 2
    assert [s.status for s in session.trajectory.steps] == [StepStatus.EXACT] * 2
    assert [s.envelope_seq for s in session.trajectory.steps] == [1, 0]


def test_no_fingerprint_match_falls_back_to_the_oldest_unconsumed() -> None:
    cassette = record_pings(1, 2)
    session = ReplaySession(cassette)
    with session:
        assert ping(7) == 2  # served the n=1 recording: approximate, flagged
        assert ping(2) == 3  # still exact: its own envelope is untouched
    assert [s.status for s in session.trajectory.steps] == [
        StepStatus.APPROXIMATE,
        StepStatus.EXACT,
    ]
    assert session.unconsumed() == []


def test_exact_match_is_preferred_over_an_older_envelope() -> None:
    cassette = record_pings(1, 2, 1)
    session = ReplaySession(cassette)
    with session:
        ping(1)
        ping(1)
        ping(2)
    assert [s.envelope_seq for s in session.trajectory.steps] == [0, 2, 1]
    assert {s.status for s in session.trajectory.steps} == {StepStatus.EXACT}


def test_synth_without_a_return_annotation_returns_none() -> None:
    session = ReplaySession(record_pings(), policy="synth")
    with session:
        assert untyped(3) is None
    [step] = session.trajectory.steps
    assert step.status is StepStatus.SYNTH


class Node(BaseModel):
    child: Node


@tool
def tree() -> Node:
    raise AssertionError("never runs")


def test_synth_reports_a_return_type_it_cannot_stub() -> None:
    session = ReplaySession(record_pings(), policy="synth")
    with session, pytest.raises(ReplayError, match="cannot synthesise"):
        tree()
    assert session.fatal is not None


def test_policy_does_not_apply_to_kinds_that_run_live(no_orders: None) -> None:
    """In tools-live a tool call is never 'unrecorded': it simply runs."""
    session = ReplaySession(record_pings(), mode="tools-live", policy="strict")
    with session:
        assert ping(1) == 2
    [step] = session.trajectory.steps
    assert step.status is StepStatus.LIVE


# --- clock and randomness -------------------------------------------------------


def world() -> dict[str, Any]:
    return {
        "now": capture.now().isoformat(),
        "time": capture.time(),
        "random": capture.random(),
        "randint": capture.randint(1, 6),
        "choice": capture.choice(["red", "green", "blue"]),
        "uuid": str(capture.uuid4()),
    }


def test_clock_and_random_are_recorded_and_replayed() -> None:
    capture.seed(1234)
    session = recorder()
    with session:
        recorded = world()
    cassette = Cassette.loads(session.finish().dumps())
    kinds = [e.kind for e in cassette.envelopes]
    assert kinds == [Kind.CLOCK, Kind.CLOCK] + [Kind.RANDOM] * 4

    for mode in ("full", "model-live", "tools-live"):
        replayed = ReplaySession(cassette, mode=mode)
        with replayed:
            assert world() == recorded
        assert {s.status for s in replayed.trajectory.steps} == {StepStatus.EXACT}


def test_capture_passes_through_without_a_session() -> None:
    before = datetime.now(UTC)
    assert before <= capture.now() <= datetime.now(UTC)
    assert 1 <= capture.randint(1, 2) <= 2
    assert isinstance(capture.uuid4(), uuid.UUID)
    with pytest.raises(IndexError):
        capture.choice([])


def test_seeded_capture_is_reproducible() -> None:
    capture.seed(7)
    first = [capture.random(), capture.randint(0, 100)]
    capture.seed(7)
    assert [capture.random(), capture.randint(0, 100)] == first


def test_choice_over_a_different_length_is_served_approximately() -> None:
    capture.seed(1)
    session = recorder()
    with session:
        capture.choice(list("abcde"))
    cassette = session.finish()
    drawn = cassette.envelopes[0].response
    assert isinstance(drawn, int)
    assert drawn < 3  # still in range for the 3-item sequence

    replayed = ReplaySession(cassette)
    with replayed:
        assert capture.choice(["x", "y", "z"]) == "xyz"[drawn]
    [step] = replayed.trajectory.steps
    assert step.status is StepStatus.APPROXIMATE


def test_unrecorded_clock_read_fails_under_strict_and_runs_under_synth() -> None:
    empty = record_pings()
    strict = ReplaySession(empty)
    with strict, pytest.raises(UnrecordedCallError):
        capture.now()

    synth = ReplaySession(empty, policy="synth")
    with synth:
        assert isinstance(capture.now(), datetime)
    [step] = synth.trajectory.steps
    assert step.status is StepStatus.FALLBACK


def test_recorded_draw_outside_the_new_range_fails_the_run() -> None:
    session = recorder()
    with session:
        drawn = capture.randint(5, 9)
    cassette = session.finish()

    replayed = ReplaySession(cassette)
    with replayed, pytest.raises(ReplayError, match="no longer fits"):
        capture.randint(0, drawn - 1)  # approximate match, value out of range
    assert replayed.fatal is not None


# --- nesting across modes -------------------------------------------------------


def record_nested() -> tuple[Cassette, Any]:
    client = wrap(make_client())

    @tool
    def summarise(order_id: str) -> str:
        """A tool that asks the model (a nested model call)."""
        reply = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=64,
            messages=[{"role": "user", "content": f"Order {order_id}"}],
        )
        return toy.final_text(reply.content)

    def agent(model_client: Any, _task: str) -> str:
        return summarise("A1001")

    session = recorder()
    with session:
        agent(client, "t")
    return session.finish(), agent


def test_nested_model_call_is_served_when_its_parent_tool_runs_live() -> None:
    cassette, agent = record_nested()
    [model] = [e for e in cassette.envelopes if e.kind is Kind.MODEL]
    assert model.parent_span_id is not None

    result = replay(cassette, agent, mode="tools-live")

    assert result.ok, result.error
    assert [(s.kind, s.status) for s in result.trajectory.steps] == [
        (Kind.TOOL, StepStatus.LIVE),
        (Kind.MODEL, StepStatus.EXACT),
    ]
    assert result.unconsumed == ()


@pytest.mark.parametrize("mode", ["full", "model-live"])
def test_nested_calls_of_a_served_parent_are_not_servable(mode: str) -> None:
    cassette, agent = record_nested()

    result = replay(cassette, agent, mode=mode, client=make_client())  # type: ignore[arg-type]

    assert result.ok, result.error
    assert [s.kind for s in result.trajectory.steps] == [Kind.TOOL]
    assert result.unconsumed == ()  # the nested model call is not "left over"


# --- threads --------------------------------------------------------------------


@tool
def leaf(n: int) -> int:
    return n * 2


@tool
def fan_out(ns: list[int]) -> list[int]:
    with ContextThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(leaf, ns))


@tool
def fan_out_plain(ns: list[int]) -> list[int]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(leaf, ns))


def test_context_pool_keeps_fanned_out_calls_nested() -> None:
    session = recorder()
    with session:
        assert fan_out([1, 2, 3]) == [2, 4, 6]
    cassette = session.finish()
    outer = next(e for e in cassette.envelopes if e.name == "fan_out")
    leaves = [e for e in cassette.envelopes if e.name == "leaf"]
    assert {e.parent_span_id for e in leaves} == {outer.span_id}

    replayed = ReplaySession(cassette)
    with replayed:
        assert fan_out([1, 2, 3]) == [2, 4, 6]
    assert replayed.unconsumed() == []
    assert replayed.trajectory.tool_names() == ["fan_out"]


def test_plain_pool_records_fanned_out_calls_top_level() -> None:
    """The documented limitation the context pool exists for."""
    session = recorder()
    with session:
        fan_out_plain([1, 2])
    cassette = session.finish()
    assert all(e.parent_span_id is None for e in cassette.envelopes)

    replayed = ReplaySession(cassette)
    with replayed:
        fan_out_plain([1, 2])
    assert [e.name for e in replayed.unconsumed()] == ["leaf", "leaf"]


# --- full mode stays bit-stable --------------------------------------------------


def test_full_replay_output_cassette_matches_the_recording() -> None:
    session = ReplaySession(eligible())
    with session:
        output = toy.run(wrap(make_client()), ELIGIBLE)
    replayed = session.finish(output=output)

    def comparable(c: Cassette) -> list[dict[str, Any]]:
        return [e.model_dump(exclude={"seq"}) for e in c.in_start_order()]

    assert comparable(replayed) == comparable(eligible())
    assert json.loads(replayed.dumps().split("\n")[0])["metadata"]["mode"] == "full"


# --- review regressions: ambiguous threads, swallowed offline errors ---------------


def test_a_worker_thread_call_with_two_active_sessions_fails_both() -> None:
    """Passing it through would run the tool live inside a full replay."""
    cassette = record_pings(1)
    first, second = ReplaySession(cassette), ReplaySession(cassette)
    entered, done = threading.Event(), threading.Event()

    def parallel_replay() -> None:  # a second replay running in another thread
        with second:
            entered.set()
            assert done.wait(5)

    other = threading.Thread(target=parallel_replay)
    other.start()
    try:
        assert entered.wait(5)
        with (
            first,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(ReplayError, match="cannot be attributed"),
        ):
            pool.submit(ping, 1).result()
    finally:
        done.set()
        other.join()
    assert first.fatal is not None
    assert second.fatal is not None
    assert first.trajectory.steps == ()


def test_fallback_without_a_client_fails_even_if_the_agent_swallows_it() -> None:
    def forgiving(client: Any, task: str) -> str:
        answer = toy.run(client, task)
        try:
            client.messages.create(model="m", max_tokens=1, messages=[])
        except Exception:  # an agent that retries or ignores model errors
            return answer
        return answer

    result = replay(eligible(), forgiving, policy="fallback")

    assert not result.ok
    assert result.error is not None
    assert "no live model client" in result.error.message


def test_a_cancelled_call_is_recorded_and_replays() -> None:
    import asyncio  # noqa: PLC0415

    @tool
    async def slow() -> str:
        await asyncio.sleep(10)
        return "late"

    async def agent() -> str:
        try:
            async with asyncio.timeout(0.01):
                return await slow()
        except TimeoutError:
            return "gave up"

    session = recorder()
    with session:
        assert asyncio.run(agent()) == "gave up"
    cassette = session.finish(output="gave up")
    [envelope] = cassette.envelopes
    assert envelope.error is not None
    assert envelope.error.type == "CancelledError"
    assert session.trajectory.steps[0].error is not None

    # Nothing is really cancelled on replay, so the recorded cancellation comes
    # back as a RecordedException naming it, not as an unrecorded call.
    replayed = ReplaySession(cassette)
    with replayed, pytest.raises(RecordedException) as info:
        asyncio.run(agent())
    assert info.value.type_name == "CancelledError"
    assert replayed.fatal is None
    assert replayed.trajectory.steps[0].status is StepStatus.EXACT


class Opaque:
    """Not JSON-schema-able: pydantic needs arbitrary_types_allowed for it."""


class Holder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thing: Opaque


@tool
def make_holder() -> Holder:
    raise AssertionError("never runs")


def test_synth_failure_for_an_unschematisable_type_is_fatal_and_named() -> None:
    session = ReplaySession(record_pings(), policy="synth")
    with session:
        try:
            make_holder()
        except Exception as exc:  # the agent swallows it
            swallowed = exc
    assert isinstance(swallowed, ReplayError)
    assert "cannot synthesise" in str(swallowed)
    assert "--policy fallback" in str(swallowed)
    assert session.fatal is swallowed


def test_envelope_seq_points_into_the_source_cassette_only() -> None:
    result = replay(eligible(), toy.run, mode="tools-live")

    source = eligible()
    for step in result.trajectory.steps:
        if step.kind is Kind.MODEL:  # served
            assert step.envelope_seq is not None
            assert source.envelopes[step.envelope_seq].fingerprint == step.fingerprint
        else:  # performed live
            assert step.envelope_seq is None


def test_a_sync_agent_returning_a_coroutine_is_awaited() -> None:
    async def body(client: Any, task: str) -> str:
        return f"{ping(1)}:{task}"

    def agent(client: Any, task: str) -> Any:
        return body(client, task)  # e.g. a lambda or functools.partial around async code

    result = replay(record_pings(1), agent, task="t")
    assert result.ok, result.error
    assert result.output == "2:t"
