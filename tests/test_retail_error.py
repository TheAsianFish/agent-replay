"""M18: the hand-written retail task where a tool legitimately fails, its rule, its
recordings, and the live harness (``eval/retail_error.py``) run against the fake model."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_replay import Cassette
from agent_replay.ci import Overrides, run_suite
from agent_replay.runner import make_client, record
from eval import mutation_study as study
from eval.mutations import MUTATIONS
from examples.retail_agent import agent as retail
from examples.retail_agent import fake_model, rules, tools
from examples.retail_agent.tasks import ACCOUNT_NOT_FOUND, ALL_TASKS, HANDWRITTEN, TASKS, task_for

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "retail_agent"
SUITE = EXAMPLE / "replay-error.toml"
FAKE = "examples.retail_agent.fake_model:make_client"
TASK = HANDWRITTEN[ACCOUNT_NOT_FOUND]


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    tools.reset_db()
    yield
    tools.reset_db()


# --- the task -------------------------------------------------------------------------


def test_the_hand_written_task_stays_out_of_the_vendored_six():
    assert len(TASKS) == 6
    assert ACCOUNT_NOT_FOUND not in TASKS
    assert ALL_TASKS == {**TASKS, **HANDWRITTEN}
    assert TASK.tau_bench_index is None
    assert TASK.expected_actions == ()
    assert task_for(TASK.instruction) is TASK


def test_the_zip_is_wrong_for_a_user_who_exists():
    with pytest.raises(ValueError, match="user not found"):
        tools.find_user_id_by_name_zip("Noah", "Ito", "98178")
    assert tools.find_user_id_by_name_zip("Noah", "Ito", "98187") == TASK.user_id


def test_the_fake_cassette_carries_one_failed_tool_result():
    path = EXAMPLE / "cassettes" / "error" / f"{TASK.id}.jsonl"
    cassette = Cassette.load(path)
    text = path.read_text(encoding="utf-8")

    assert text.count('"is_error":true') == 1
    assert [e.name for e in cassette.envelopes] == [
        "anthropic.messages.create",
        "find_user_id_by_name_zip",
        "anthropic.messages.create",
    ]


@pytest.mark.parametrize("mode", ["full", "tools-live", "model-live"])
def test_the_suite_passes_the_unmutated_agent(mode: str):
    report = run_suite(
        SUITE,
        overrides=Overrides(mode=mode, client=FAKE if mode == "model-live" else None),  # type: ignore[arg-type]
    )

    assert report.passed, [c.check for s in report.scenarios for c in s.failed_checks]


def test_the_fake_model_cancels_once_the_lookup_succeeds(monkeypatch: pytest.MonkeyPatch):
    """The fake model's plan past authentication, which only a run that hides the failure
    reaches: given the right zip it reads the order and cancels it."""
    plan = fake_model.PLANS[ACCOUNT_NOT_FOUND]
    monkeypatch.setitem(fake_model.PLANS, ACCOUNT_NOT_FOUND, dataclasses.replace(plan, zip="98187"))

    result, _ = record(retail.run, TASK.instruction, client=make_client(FAKE))

    assert result.ok, result.error
    assert rules.tool_calls(result)[-1] == (
        "cancel_pending_order",
        {"order_id": "#W6729841", "reason": "no longer needed"},
    )


# --- the rule -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I couldn't find your account; please check your details.",
        "I could not locate an account with that name and zip code.",
        "Sorry, I was unable to verify your identity.",
        "The user was not found.",
        "There is no account matching those details.",
        "Your name and zip code don't match our records.",
        "I can\u2019t find a user with the provided details.",
    ],
)
def test_not_found_answers_are_recognised(answer: str):
    assert rules.NOT_FOUND.search(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Done: order #W6729841 is now cancelled.",
        "Your order has been cancelled and refunded.",
        "",
    ],
)
def test_other_answers_are_not(answer: str):
    assert rules.NOT_FOUND.search(answer) is None


def _run(output: str, calls: list[tuple[str, dict[str, Any]]]) -> Any:
    steps = [SimpleNamespace(name=n, request=a) for n, a in calls]
    return SimpleNamespace(output=output, trajectory=SimpleNamespace(tool_steps=lambda: steps))


def test_the_rule_fails_a_write_even_with_a_not_found_answer():
    run = _run(
        "I couldn't find your account, but I cancelled the order.",
        [("cancel_pending_order", {"order_id": "#W6729841", "reason": "no longer needed"})],
    )
    assert rules.reports_account_not_found(run)[0] is False
    assert rules.reports_account_not_found(_run("User not found.", []))[0] is True
    assert rules.reports_account_not_found(_run("Your order is cancelled.", []))[0] is False


# --- the mutation study's section -----------------------------------------------------

COMMITTED: dict[str, Any] = json.loads(
    (ROOT / "eval" / "results" / "mutations.json").read_text(encoding="utf-8")
)


def test_the_committed_error_task_section():
    section = COMMITTED["error_task"]
    agent_mutations = [m.id for m in MUTATIONS if m.target == "agent"]

    assert [r["id"] for r in section["mutations"]] == agent_mutations
    assert section["unmutated_failed"] == []
    caught = [c for c, ids in section["detected"].items() if "drops_error_flag" in ids]
    assert len(caught) >= 2
    assert {"model-live", "model-live+fallback", "tools-live+strict"} <= set(caught)
    assert not {"full", "tools-live+drift"} & set(caught)


def test_drops_error_flag_on_the_error_task_is_seen_where_the_model_runs():

    configs = {c.id: c for c in study.CONFIGS}
    agent = "eval.mutations.agents:drops_error_flag"
    full = study.replay_suite(agent, FAKE, configs["full"], suite=study.ERROR_SUITE)
    live = study.replay_suite(agent, FAKE, configs["model-live"], suite=study.ERROR_SUITE)

    # In full the agent's request lacks the flag (one approximate step), but the
    # recorded reply is served all the same.
    assert full.passed
    assert full.approximate_steps == 1
    # The fake model takes the error text for a user id; the recording has no such call.
    assert "run_error:UnrecordedCallError" in live.caught_by
