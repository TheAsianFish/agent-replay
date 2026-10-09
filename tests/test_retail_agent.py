"""The retail example: τ-bench reward on fresh recordings, committed cassettes, mutants."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_replay import Cassette, StepStatus
from agent_replay.checks import tool_sequence
from agent_replay.runner import load_agent, record, replay
from examples.retail_agent import agent as retail
from examples.retail_agent import rules, tools
from examples.retail_agent.fake_model import make_client
from examples.retail_agent.mutants import first_tool_only, skips_order_lookup, wrong_write_argument
from examples.retail_agent.tasks import TASKS, expected_db_hash, task_for

CASSETTES = Path(__file__).parents[1] / "examples" / "retail_agent" / "cassettes"
TASK_IDS = sorted(TASKS)
WRITE_TASK_IDS = sorted(
    t.id for t in TASKS.values() if any(a.name in tools.WRITE_TOOLS for a in t.expected_actions)
)
# Tasks whose recording has a turn with several tool calls (all but the single-order lookup).
PARALLEL_TASK_IDS = sorted(set(TASK_IDS) - {"lookup_latest_order_total"})


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    tools.reset_db()
    yield
    tools.reset_db()


def load(task_id: str) -> Cassette:
    return Cassette.load(CASSETTES / f"{task_id}.jsonl")


def test_six_tasks_cover_every_kind_of_request():
    kinds = {a.name for t in TASKS.values() for a in t.expected_actions}
    assert kinds == tools.WRITE_TOOLS | tools.TERMINAL_TOOLS
    assert [t.id for t in TASKS.values() if not t.expected_actions] == ["lookup_latest_order_total"]
    assert {t.tau_bench_index for t in TASKS.values()} == {10, 17, 27, 53, 67, 69}


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_fake_model_run_reaches_the_expected_database(task_id: str):
    """The τ-bench objective reward: final DB state equals the ground truth's."""
    task = TASKS[task_id]
    expected = expected_db_hash(task)
    tools.reset_db()
    initial = tools.db_hash()

    result, _ = record(retail.run, task.instruction, client=make_client())

    assert result.ok, result.error
    assert tools.db_hash() == expected
    assert (expected != initial) == any(a.name in tools.WRITE_TOOLS for a in task.expected_actions)
    calls = rules.tool_calls(result)
    assert rules.write_actions_match(calls, task.expected_actions)[0]
    assert rules.authenticated_before_write(calls)[0]
    assert rules.order_checked_before_write(calls)[0]
    assert rules.outputs_match(result.output, task.expected_outputs)[0]


def test_lookup_task_answers_with_tau_benchs_expected_output():
    result, _ = record(
        retail.run, TASKS["lookup_latest_order_total"].instruction, client=make_client()
    )
    assert result.output == "You paid $829.43 for your most recent order, #W6729841."


def test_policy_violation_is_transferred_without_writing():
    task = TASKS["transfer_swapped_refunds"]
    result, _ = record(retail.run, task.instruction, client=make_client())
    calls = rules.tool_calls(result)
    assert calls[-1][0] == "transfer_to_human_agents"
    assert rules.no_write_actions(calls) == (True, "no writes")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_committed_cassette_replays_offline_and_passes(task_id: str):
    cassette = load(task_id)
    assert cassette.header.agent == "examples.retail_agent.agent:run"
    assert task_for(cassette.header.task or "") is TASKS[task_id]

    result = replay(cassette, load_agent(cassette.header.agent))

    assert result.ok, result.error
    assert result.output == cassette.footer.output
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}
    assert result.unconsumed == ()
    [check] = result.check(tool_sequence(cassette.tool_names()))
    assert check.passed, check.message
    passed, message = rules.write_actions_match(
        rules.tool_calls(result), TASKS[task_id].expected_actions
    )
    assert passed, message


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_committed_cassettes_match_a_fresh_recording(task_id: str):
    """Fails when the agent, tools, data or fake model changed without `make retail-cassettes`."""
    committed = load(task_id)
    assert committed.header.task is not None

    _, fresh = record(retail.run, committed.header.task, client=make_client())

    def comparable(c: Cassette) -> list[dict[str, object]]:
        return [e.model_dump(exclude={"latency_ms"}) for e in c.envelopes]

    assert comparable(fresh) == comparable(committed), "re-record with `make retail-cassettes`"
    assert fresh.footer == committed.footer


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_mutant_that_skips_order_lookups_fails_tool_sequence(task_id: str):
    cassette = load(task_id)

    result = replay(cassette, skips_order_lookup)
    [check] = result.check(tool_sequence(cassette.tool_names()))

    assert not check.passed
    assert check.message.startswith("tool call #3 differs; missing get_order_details")
    assert "get_order_details" not in result.trajectory.tool_names()
    assert {e.name for e in result.unconsumed} == {"get_order_details"}
    assert result.trajectory.count(StepStatus.APPROXIMATE) >= 1


@pytest.mark.parametrize("task_id", WRITE_TASK_IDS)
def test_mutant_that_changes_a_write_argument_is_served_approximately(task_id: str):
    """Names match, so tool_sequence passes; the write step and the rule catch it."""
    cassette = load(task_id)
    task = TASKS[task_id]

    result = replay(cassette, wrong_write_argument)

    assert result.check(tool_sequence(cassette.tool_names()))[0].passed
    [approximate] = [s for s in result.trajectory.steps if s.status is StepStatus.APPROXIMATE]
    assert approximate.name in tools.WRITE_TOOLS
    passed, message = rules.write_actions_match(rules.tool_calls(result), task.expected_actions)
    assert not passed
    assert message.startswith("action #1: expected")


@pytest.mark.parametrize("task_id", PARALLEL_TASK_IDS)
def test_mutant_that_drops_parallel_tool_calls_fails_tool_sequence(task_id: str):
    cassette = load(task_id)

    result = replay(cassette, first_tool_only)
    [check] = result.check(tool_sequence(cassette.tool_names()))

    assert not check.passed
    assert "missing get_order_details" in check.message
    assert result.trajectory.count(StepStatus.APPROXIMATE) >= 1
    assert result.unconsumed
    assert {e.name for e in result.unconsumed} == {"get_order_details"}


def test_first_tool_only_is_invisible_without_parallel_calls():
    """A known gap, recorded so it stays visible: one call per turn hides the bug."""
    cassette = load("lookup_latest_order_total")
    result = replay(cassette, first_tool_only)
    assert result.check(tool_sequence(cassette.tool_names()))[0].passed
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}


def test_expected_db_hash_differs_per_write_and_resets_first():
    first = expected_db_hash(TASKS["cancel_pending_laptop"])
    again = expected_db_hash(TASKS["cancel_pending_laptop"])
    other = expected_db_hash(TASKS["return_damaged_bicycle"])
    assert first == again != other


def test_cancel_refunds_a_gift_card_immediately():
    before = tools.get_user_details("emma_smith_8564")["payment_methods"]["gift_card_8541487"]
    order = tools.cancel_pending_order("#W2417020", "no longer needed")
    after = tools.get_user_details("emma_smith_8564")["payment_methods"]["gift_card_8541487"]
    assert order["status"] == "cancelled"
    assert after["balance"] == round(before["balance"] + 2674.4, 2)


def test_cancelling_a_non_pending_order_fails_like_tau_bench():
    with pytest.raises(ValueError, match=r"^non-pending order cannot be cancelled$"):
        tools.cancel_pending_order("#W3916020", "no longer needed")


def test_invalid_cancel_reason_is_rejected():
    with pytest.raises(ValueError, match=r"^invalid reason$"):
        tools.cancel_pending_order("#W2417020", "found it cheaper")  # type: ignore[arg-type]


def test_unknown_email_is_user_not_found():
    with pytest.raises(ValueError, match=r"^user not found$"):
        tools.find_user_id_by_email("nobody@example.com")


def test_email_lookup_is_case_insensitive():
    assert tools.find_user_id_by_email("MIA.Garcia2723@example.com") == "mia_garcia_4516"


def test_return_to_another_card_violates_policy():
    with pytest.raises(ValueError, match="original payment method or a gift card"):
        tools.return_delivered_order_items("#W5490111", ["4579334072"], "paypal_9497703")


def test_exchange_to_an_unavailable_variant_fails():
    with pytest.raises(ValueError, match=r"^new item 4694984344 not found or available$"):
        tools.exchange_delivered_order_items(
            "#W7181492", ["8118291112"], ["4694984344"], "paypal_3024827"
        )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("652.61 - 642.72", "9.89"), ("(1 + 2) * 3 / 4", "2.25"), ("-2 ** 2", "-4.0")],
)
def test_calculate_matches_tau_bench(expression: str, expected: str):
    assert tools.calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os')", "invalid characters in expression"),
        ("1 / 0", "division by zero"),
        ("9 ** 9 ** 9", "exponent 387420489 is too large"),
    ],
)
def test_calculate_errors(expression: str, message: str):
    with pytest.raises(ValueError, match=message):
        tools.calculate(expression)


def test_tool_errors_become_tau_bench_worded_error_results():
    result = retail.call_tool("toolu_1", "get_order_details", {"order_id": "#W0000000"})
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "Error: order not found",
        "is_error": True,
    }
    assert retail.call_tool("toolu_2", "nope", {})["content"] == "Error: unknown tool nope"


def test_reset_db_discards_changes_in_place():
    db = tools.DB
    initial = tools.db_hash()
    tools.cancel_pending_order("#W2417020", "no longer needed")
    assert tools.db_hash() != initial
    tools.reset_db()
    assert tools.DB is db
    assert tools.db_hash() == initial


def test_rules_report_violations():
    calls = [
        ("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}),
        ("find_user_id_by_email", {"email": "a@example.com"}),
    ]
    assert rules.authenticated_before_write(calls) == (
        False,
        "cancel_pending_order called before authenticating the user",
    )
    assert rules.order_checked_before_write(calls)[0] is False
    assert rules.no_write_actions(calls)[0] is False
    assert rules.write_actions_match(calls, []) == (
        False,
        "unexpected extra action ('cancel_pending_order', "
        "{'order_id': '#W1', 'reason': 'no longer needed'})",
    )
    assert rules.outputs_match("It cost $1,093.34.", ["1093.34"])[0]
    assert not rules.outputs_match("It cost $1093.", ["1093.34"])[0]
