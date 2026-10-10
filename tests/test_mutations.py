"""The mutation study's ingredients and outcomes: the retail database rule, the catalogue
of mutations and benign refactors, and a fast slice of the study itself."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_replay import Cassette
from agent_replay.runner import load_agent, make_client, replay
from eval.mutations import MUTATIONS, REFACTORS, Mutation, Refactor
from eval.mutations.tools import MUTANT_TOOLS
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


# --- the catalogue ----------------------------------------------------------------


def test_catalogue_has_enough_mutations_of_every_kind_and_five_refactors():
    assert len(MUTATIONS) >= 25
    assert len(REFACTORS) == 5
    ids = [m.id for m in MUTATIONS] + [r.id for r in REFACTORS]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", i) for i in ids)
    assert {m.target for m in MUTATIONS} == {"agent", "model", "tool"}
    assert all(m.description for m in MUTATIONS)
    assert {m.id for m in MUTATIONS if m.target == "tool"} == set(MUTANT_TOOLS)


@pytest.mark.parametrize("variant", [*MUTATIONS, *REFACTORS], ids=lambda v: v.id)
def test_every_reference_loads(variant: Mutation | Refactor):
    assert callable(load_agent(variant.agent))
    assert make_client(variant.client) is not None


@pytest.mark.parametrize("mutation_id", sorted(MUTANT_TOOLS))
def test_tool_mutants_keep_the_real_tools_name_and_schema(mutation_id: str):
    mutant = MUTANT_TOOLS[mutation_id]
    real = retail.TOOLS[mutant.name]
    assert mutant is not real
    assert mutant.to_anthropic() == real.to_anthropic()

    agent = getattr(load_agent(f"eval.mutations.tools:{mutation_id}"), "__self__", None)
    assert isinstance(agent, retail.RetailAgent)
    assert agent.tools[mutant.name] is mutant
    # The model is offered exactly what the recordings offered it.
    assert [t.to_anthropic() for t in agent.tools.values()] == [
        t.to_anthropic() for t in retail.TOOLS.values()
    ]
