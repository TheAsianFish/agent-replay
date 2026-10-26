"""The pydantic-ai travel desk: its suites replay offline, its rules judge outcomes."""

from __future__ import annotations

import asyncio
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from agent_replay.cassette import Cassette, Kind
from agent_replay.ci import Overrides, run_suite
from agent_replay.fake import FakeAsyncAnthropic, Reply, ToolCall, scripted
from agent_replay.runner import record, replay
from agent_replay.trajectory import StepStatus
from examples.pydantic_ai_agent import agent, environment, fake_model, rules
from examples.pydantic_ai_agent.tasks import TASKS
from examples.pydantic_ai_agent.tools import BOOKINGS, book_flight, cancel_booking, reset

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "pydantic_ai_agent"
SUITES = ["replay.toml", "replay-qwen.toml"]


@pytest.mark.parametrize("suite", SUITES)
@pytest.mark.parametrize("entry", ["run", "run_streaming"])
def test_suite_replays_in_full_with_every_step_exact(suite, entry):
    # The committed cassettes were recorded with agent.run (no streaming); stream is not
    # part of a request's identity, so run_stream replays them exactly too.
    report = run_suite(
        EXAMPLE / suite,
        overrides=Overrides(mode="full", agent=f"examples.pydantic_ai_agent.agent:{entry}"),
    )
    assert report.passed, [
        (s.name, s.error, [c for c in s.checks if not c.passed]) for s in report.scenarios
    ]
    for scenario in report.scenarios:
        assert scenario.steps.by_status == {"exact": scenario.steps.total}
        assert scenario.unconsumed == ()


def test_qwen_cassettes_name_their_model():
    for path in sorted((EXAMPLE / "cassettes" / "qwen").glob("*.jsonl")):
        cassette = Cassette.load(path)
        assert cassette.header.metadata["model"] == "qwen2.5:7b"
        assert cassette.header.metadata["provider"] == "ollama"
        models = {e.response["model"] for e in cassette.envelopes if e.kind is Kind.MODEL}
        assert models == {"qwen2.5:7b"}
        names = {e.name for e in cassette.envelopes if e.kind is Kind.MODEL}
        assert names == {"anthropic.beta.messages.create"}
        # A local model has no price: unpriced, never a measured-looking $0.
        assert all(e.cost_usd is None for e in cassette.envelopes if e.kind is Kind.MODEL)


@pytest.mark.parametrize("name", list(TASKS))
def test_fake_recording_matches_the_committed_cassette(name):
    """Re-recording with the fake model makes the same requests, so the suite is current."""
    committed = Cassette.load(EXAMPLE / "cassettes" / f"{name}.jsonl")
    _, fresh = record(agent.run, TASKS[name].text, client=fake_model.make_client())
    assert [(e.kind, e.name, e.fingerprint) for e in fresh.envelopes] == [
        (e.kind, e.name, e.fingerprint) for e in committed.envelopes
    ]


def test_a_prompt_change_shows_up_as_approximate_steps():
    """Replay notices when the agent's requests drift from the recording."""
    cassette = Cassette.load(EXAMPLE / "cassettes" / "cheapest_fare_in_euros.jsonl")

    async def reworded(client, task):
        reset()
        result = await agent.build_agent(client, model="qwen2.5:14b").run(task)
        return result.output

    result = replay(cassette, reworded, mode="full")
    assert result.ok
    model_steps = [s for s in result.trajectory.steps if s.kind is Kind.MODEL]
    assert {s.status for s in model_steps} == {StepStatus.APPROXIMATE}


def test_tools_act_on_bookings_that_reset_every_run():
    reset()
    assert book_flight("dl2207", "Ana Lima").booking_ref == "NEW001"
    assert cancel_booking("QX7Y2K").status == "cancelled"
    reset()
    assert set(BOOKINGS) == {"QX7Y2K", "HB4M9T"}
    assert BOOKINGS["QX7Y2K"].status == "confirmed"
    with pytest.raises(ModelRetry, match="full"):
        book_flight("AA178", "Ana Lima")


class _Run:
    """The parts of a RunResult the rules read."""

    def __init__(self, calls, output):
        steps = [SimpleNamespace(name=n, request=a) for n, a in calls]
        self.trajectory = SimpleNamespace(tool_steps=lambda: steps)
        self.output = output


@pytest.mark.parametrize(
    ("output", "passed"),
    [
        ("Cancelled; you get $243.00 back.", True),
        ("You get 243 USD back.", True),
        ("You get $1,243 back.", False),
        ("You get $24.30 back.", False),
        ("You get $243.50 back.", False),
        (None, False),
    ],
)
def test_answer_states_matches_the_figure_exactly(output, passed):
    ok, _ = rules.answer_states(_Run([], output), task="cancel_refundable_booking")
    assert ok is passed


def test_writes_match_ignores_case_and_spacing_but_not_the_flight():
    good = [
        ("search_flights", {}),
        ("book_flight", {"flight_id": "dl2207", "passenger": "ana lima "}),
    ]
    assert rules.writes_match(_Run(good, ""), task="book_earliest_flight")[0]
    wrong = [
        ("search_flights", {}),
        ("book_flight", {"flight_id": "WN1849", "passenger": "Ana Lima"}),
    ]
    ok, message = rules.writes_match(_Run(wrong, ""), task="book_earliest_flight")
    assert not ok
    assert "WN1849" in message
    assert rules.writes_match(_Run([], ""), task="cheapest_fare_in_euros")[0]


def test_looked_up_before_writing():
    assert rules.looked_up_before_writing(_Run([("get_booking", {}), ("cancel_booking", {})], ""))[
        0
    ]
    ok, message = rules.looked_up_before_writing(_Run([("cancel_booking", {})], ""))
    assert not ok
    assert "before any lookup" in message


def test_ollama_model_is_none_without_a_server_or_the_model(monkeypatch):
    def refused(path):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(environment, "_get", refused)
    assert environment.ollama_model() is None
    answers = {
        "/api/version": {"version": "0.40.11"},
        "/api/tags": {"models": [{"name": "llama3:8b"}]},
    }
    monkeypatch.setattr(environment, "_get", answers.__getitem__)
    assert environment.ollama_model() is None
    answers["/api/tags"]["models"].append({"name": "qwen2.5:7b", "digest": "845d"})
    assert environment.ollama_model() == {
        "ollama": "0.40.11",
        "model": "qwen2.5:7b",
        "digest": "845d",
    }


def test_a_tool_retry_records_and_replays_as_the_same_retry_prompt():
    """A ModelRetry from a tool is recorded as an error and re-raised as a ModelRetry."""
    policy = scripted(
        [
            Reply(tool_calls=[ToolCall("get_booking", {"booking_ref": "NOPE00"})]),
            Reply(tool_calls=[ToolCall("get_booking", {"booking_ref": "QX7Y2K"})]),
            Reply(text="QX7Y2K is refundable; $243.00 back if cancelled."),
        ]
    )
    task = TASKS["cancel_refundable_booking"].text
    recorded, cassette = record(agent.run, task, client=FakeAsyncAnthropic(policy))
    assert recorded.ok, recorded.error
    [failed, _] = [e for e in cassette.envelopes if e.kind is Kind.TOOL]
    assert failed.error is not None
    assert failed.error.type == "ModelRetry"
    result = replay(cassette, agent.run, mode="full")
    assert result.ok, result.error
    assert result.output == recorded.output
    assert result.trajectory.count(StepStatus.APPROXIMATE) == 0


def test_a_client_made_for_the_run_is_closed_on_its_own_loop():
    """replay ci / flake close the async client they made, inside the run's event loop."""
    closed: list[bool] = []

    class Closable(FakeAsyncAnthropic):
        async def close(self) -> None:
            asyncio.get_running_loop()  # raises if the loop is already gone
            closed.append(True)

    cassette = Cassette.load(EXAMPLE / "cassettes" / "cheapest_fare_in_euros.jsonl")
    client = Closable(fake_model.policy)
    replay(cassette, agent.run, mode="model-live", client=client)
    assert closed == []  # a caller's own client is left alone
    result = replay(cassette, agent.run, mode="model-live", client=client, close_client=True)
    assert result.ok, result.error
    assert closed == [True]


def test_a_client_that_fails_to_close_keeps_the_runs_result():
    class Broken(FakeAsyncAnthropic):
        async def close(self) -> None:
            raise RuntimeError("pool already closed")

    cassette = Cassette.load(EXAMPLE / "cassettes" / "cheapest_fare_in_euros.jsonl")
    client = Broken(fake_model.policy)
    result = replay(cassette, agent.run, mode="model-live", client=client, close_client=True)
    assert result.ok, result.error


async def _run_with_max_retries(client, task):
    # ModelSettings(max_retries=...) sends each request through client.with_options(
    # http_client=client._client, max_retries=...), a copy the replay must still serve.
    reset()
    built = agent.build_agent(client)
    result = await built.run(task, model_settings={"max_retries": 0})
    return result.output


def test_an_agent_that_sets_max_retries_replays_in_full_without_a_client():
    path = EXAMPLE / "cassettes" / "cheapest_fare_in_euros.jsonl"
    cassette = Cassette.load(path)
    result = replay(cassette, _run_with_max_retries, mode="full")
    assert result.error is None, result.error
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}


def test_an_agent_that_sets_max_retries_records_through_the_wrapped_fake():
    task = TASKS["cheapest_fare_in_euros"].text
    _, cassette = record(_run_with_max_retries, task, client=fake_model.make_client())
    assert sum(e.kind is Kind.MODEL for e in cassette.envelopes) == 3
