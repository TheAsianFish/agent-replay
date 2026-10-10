"""The mutation study's ingredients and outcomes: the retail database rule, the catalogue
of mutations and benign refactors, and a fast slice of the study itself."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_replay import Cassette
from agent_replay.runner import replay
from examples.retail_agent import agent as retail
from examples.retail_agent import rules, tools
from examples.retail_agent.mutants import wrong_write_argument
from examples.retail_agent.tasks import TASKS

ROOT = Path(__file__).parents[1]
CASSETTES = ROOT / "examples" / "retail_agent" / "cassettes"
TASK_IDS = sorted(TASKS)


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    tools.reset_db()
    yield
    tools.reset_db()


def load(task_id: str) -> Cassette:
    return Cassette.load(CASSETTES / f"{task_id}.jsonl")


# --- the database rule ------------------------------------------------------------


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_db_rule_is_vacuous_when_the_tools_were_served(task_id: str):
    tools.DB.orders.clear()  # whatever the database holds, a served run never touched it
    result = replay(load(task_id), retail.run)
    tools.DB.orders.clear()

    passed, message = rules.db_matches_ground_truth(result, task_id)

    assert passed
    assert message.startswith("not applicable:")
    assert "served from the cassette" in message


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_db_rule_passes_the_real_agent_in_tools_live(task_id: str):
    result = replay(load(task_id), retail.run, mode="tools-live")

    assert result.ok, result.error
    assert rules.db_matches_ground_truth(result, task_id) == (
        True,
        "the database equals the ground truth's",
    )


def test_db_rule_catches_a_wrong_database_in_tools_live():
    result = replay(load("cancel_pending_laptop"), wrong_write_argument, mode="tools-live")

    passed, message = rules.db_matches_ground_truth(result, "cancel_pending_laptop")

    assert not passed
    assert message == "the database differs from the ground truth's in orders #W2417020"


def test_db_rule_leaves_the_database_as_the_run_left_it():
    result = replay(load("modify_address_suite"), retail.run, mode="tools-live")
    after_run = tools.db_hash()
    rules._expected_db.cache_clear()  # computing the expected database resets and replays it

    assert not rules.db_matches_ground_truth(result, "cancel_pending_laptop")[0]
    assert tools.db_hash() == after_run


def test_every_run_starts_from_the_vendored_database():
    tools.DB.orders.clear()

    result = replay(load("lookup_latest_order_total"), retail.run, mode="tools-live")

    assert result.ok, result.error
    assert rules.db_matches_ground_truth(result, "lookup_latest_order_total")[0]
