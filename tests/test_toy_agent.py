"""End to end on the toy agent: committed cassettes replay offline, mutants fail."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from agent_replay import Cassette, StepStatus, wrap
from agent_replay.checks import tool_sequence
from agent_replay.runner import load_agent, record, replay
from examples.toy_agent import agent as toy
from examples.toy_agent.fake_model import make_client
from examples.toy_agent.mutants import first_tool_only

CASSETTES = Path(__file__).parents[1] / "examples" / "toy_agent" / "cassettes"
EXPECTED_TOOLS = {
    "refund_eligible.jsonl": ["lookup_order", "get_refund_policy", "issue_refund"],
    "refund_denied.jsonl": ["lookup_order", "get_refund_policy"],
}


def test_sockets_are_blocked_in_tests():
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 443))


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_committed_cassette_replays_offline_and_passes(name: str):
    cassette = Cassette.load(CASSETTES / name)
    assert cassette.header.agent is not None

    result = replay(cassette, load_agent(cassette.header.agent))

    assert result.ok, result.error
    assert result.output == cassette.footer.output
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}
    assert result.unconsumed == ()
    [check] = result.check(tool_sequence(EXPECTED_TOOLS[name]))
    assert check.passed, check.message


def test_replay_is_bit_stable():
    cassette = Cassette.load(CASSETTES / "refund_eligible.jsonl")
    first = replay(cassette, toy.run)
    second = replay(cassette, toy.run)
    assert first == second


def test_mutant_that_drops_parallel_tool_calls_fails_tool_sequence():
    cassette = Cassette.load(CASSETTES / "refund_eligible.jsonl")

    result = replay(cassette, first_tool_only)
    [check] = result.check(tool_sequence(cassette.tool_names()))

    assert not check.passed
    assert check.message == "tool call #2 differs; missing get_refund_policy"
    assert check.details["diff"] == ["  lookup_order", "- get_refund_policy", "  issue_refund"]
    # The model saw a different conversation from the recorded one, and the
    # recorded policy lookup was never asked for: both are surfaced.
    assert result.trajectory.count(StepStatus.APPROXIMATE) == 2
    assert [e.name for e in result.unconsumed] == ["get_refund_policy"]


def test_changed_prompt_is_served_approximately_but_keeps_the_tool_sequence(monkeypatch):
    """A prompt edit changes every model request; tool_sequence alone does not catch it."""
    monkeypatch.setattr(toy, "SYSTEM", toy.SYSTEM + " Be terse.")
    cassette = Cassette.load(CASSETTES / "refund_eligible.jsonl")

    result = replay(cassette, toy.run)

    model_steps = [s for s in result.trajectory.steps if s.kind.value == "model"]
    assert {s.status for s in model_steps} == {StepStatus.APPROXIMATE}
    assert result.check(tool_sequence(cassette.tool_names()))[0].passed


def test_agent_making_an_unrecorded_call_fails_with_a_clear_error():
    cassette = Cassette.load(CASSETTES / "refund_denied.jsonl")

    def refunds_anyway(client, task):
        toy.run(client, task)
        return toy.issue_refund("A1002", 1.0)

    result = replay(cassette, refunds_anyway)

    assert result.error is not None
    assert result.error.type == "UnrecordedCallError"
    assert "replay record" in result.error.message
    assert result.trajectory.steps[-1].status is StepStatus.UNRECORDED
    assert not result.check(tool_sequence(cassette.tool_names()))[0].passed


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_committed_cassettes_match_a_fresh_recording(name: str):
    """Fails when the agent or fake model changed without re-recording (`make cassettes`)."""
    committed = Cassette.load(CASSETTES / name)
    assert committed.header.task is not None

    _, fresh = record(toy.run, committed.header.task, client=make_client())

    def comparable(c: Cassette) -> list[dict[str, object]]:
        return [e.model_dump(exclude={"latency_ms"}) for e in c.envelopes]

    assert comparable(fresh) == comparable(committed), "re-record with `make cassettes`"
    assert fresh.footer == committed.footer


def test_agent_runs_unchanged_without_a_session():
    """Instrumented code is production code: no session, no interception."""
    output = toy.run(wrap(make_client()), "Order A1001 arrived damaged. Can I get a refund?")
    assert output == "Done: I refunded $42.50 for order A1001 (reference RF-A1001)."


def test_unknown_order_becomes_an_error_tool_result_and_replays():
    result, cassette = record(toy.run, "Refund order Z9999 please", client=make_client())
    assert result.ok
    [lookup] = [e for e in cassette.envelopes if e.name == "lookup_order"]
    assert lookup.error is not None
    assert lookup.error.type == "KeyError"

    replayed = replay(Cassette.loads(cassette.dumps()), toy.run)
    assert replayed.ok
    assert {s.status for s in replayed.trajectory.steps} == {StepStatus.EXACT}
    assert replayed.output == result.output
