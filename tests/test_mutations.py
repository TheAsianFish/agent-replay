"""The mutation study's ingredients and outcomes: the retail database rule, the catalogue
of mutations and benign refactors, and a fast slice of the study itself."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from agent_replay import Cassette
from agent_replay.runner import load_agent, make_client, replay
from eval import mutation_study as study
from eval.mutations import FAKE_MODEL, MUTATIONS, REFACTORS, Mutation, Refactor
from eval.mutations.tools import MUTANT_TOOLS
from examples.retail_agent import agent as retail
from examples.retail_agent import rules, tools
from examples.retail_agent.mutants import wrong_write_argument
from examples.retail_agent.tasks import TASKS

ROOT = Path(__file__).parents[1]
CASSETTES = ROOT / "examples" / "retail_agent" / "cassettes"
TASK_IDS = sorted(TASKS)
COMMITTED: dict[str, Any] = json.loads(
    (ROOT / "eval" / "results" / "mutations.json").read_text(encoding="utf-8")
)
COMMITTED_ROWS = {row["id"]: row for row in [*COMMITTED["mutations"], *COMMITTED["refactors"]]}
CONFIG = {c.id: c for c in study.CONFIGS}
CATALOGUE = {m.id: m for m in MUTATIONS}
REFACTOR = {r.id: r for r in REFACTORS}


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


def test_expected_database_is_stable_under_concurrent_resets():
    """Concurrent ``model-live`` runs reset the database when they start; a reset in the
    middle of replaying a task's ground truth must not poison the cached expected state."""
    rules._expected_db.cache_clear()
    expected = {task_id: rules._expected_db(task_id) for task_id in TASK_IDS}
    rules._expected_db.cache_clear()
    workers = len(TASK_IDS) * 2
    barrier = threading.Barrier(workers)

    def compute(task_id: str) -> tuple[str, dict[str, Any]]:
        barrier.wait()
        for _ in range(20):
            tools.reset_db()
        return rules._expected_db(task_id)

    def churn(_: int) -> tuple[str, dict[str, Any]]:
        barrier.wait()
        for _ in range(200):
            tools.reset_db()
        return "", {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        churning = [pool.submit(churn, i) for i in range(len(TASK_IDS))]
        computed = list(pool.map(compute, TASK_IDS))
        for future in churning:
            future.result()
    rules._expected_db.cache_clear()

    assert computed == [expected[task_id] for task_id in TASK_IDS]
    for task_id in TASK_IDS:
        assert rules._expected_db(task_id) == expected[task_id]


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


# --- the study --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation_id", "config", "caught_by"),
    [
        # A model regression is invisible where the model is served...
        ("wrong_cancel_reason", "full", []),
        ("wrong_cancel_reason", "tools-live", []),
        # ...and caught by the ground truth where it runs.
        ("wrong_cancel_reason", "model-live", ["rule:matches_ground_truth"]),
        # A tool regression shows only where the tools run, and only in the database.
        ("cancel_keeps_status", "tools-live", ["rule:db_matches_ground_truth"]),
        ("cancel_keeps_status", "model-live", []),
        # A read-only tool bug changes only what the tool returns: the drift check sees it...
        ("order_details_stale_status", "tools-live", []),
        ("order_details_stale_status", "tools-live+drift", ["no_tool_drift"]),
        # ...and a bug that only changes the database is no drift (its result is unchanged).
        ("cancel_keeps_status", "tools-live+drift", ["rule:db_matches_ground_truth"]),
        # An orchestration bug shows in every mode; strict ends the run at the extra call.
        (
            "repeats_write_calls",
            "full",
            ["rule:matches_ground_truth", "run_error:UnrecordedCallError", "tool_sequence"],
        ),
        (
            "repeats_write_calls",
            "model-live+fallback",
            ["rule:matches_ground_truth", "tool_sequence"],
        ),
    ],
)
def test_study_outcomes(mutation_id: str, config: str, caught_by: list[str]):
    mutation = CATALOGUE[mutation_id]

    outcome = study.replay_suite(mutation.agent, mutation.client, CONFIG[config])

    assert outcome.passed is not bool(caught_by)
    assert list(outcome.caught_by) == caught_by
    assert COMMITTED_ROWS[mutation_id]["results"][config] == outcome.to_json()


def test_compact_json_refactor_is_served_approximately_and_passes():
    refactor = REFACTOR["compact_tool_json"]

    outcome = study.replay_suite(refactor.agent, refactor.client, CONFIG["full"])

    assert outcome.passed
    assert outcome.approximate_steps > 0
    assert COMMITTED_ROWS["compact_tool_json"]["results"]["full"] == outcome.to_json()


def test_a_scenario_that_cannot_run_is_a_harness_error():
    with pytest.raises(study.HarnessError, match=r"did not run: .*Fix the catalogue entry"):
        study.replay_suite("eval.mutations.agents:no_such_agent", FAKE_MODEL, CONFIG["full"])


def test_main_exits_2_on_a_harness_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    broken = Mutation("broken", "agent", "missing", agent="eval.mutations.agents:missing")
    real_run = study.run
    monkeypatch.setattr(study, "run", lambda: real_run(mutations=(broken,), refactors=()))

    assert study.main([]) == 2
    assert "harness bug" in capsys.readouterr().err


def test_strict_configuration_fails_on_approximate_steps():
    """The ``tools-live+strict`` configuration turns the drift signal (approximate steps)
    into a failure: the JSON-layout refactor passes without it and fails with it."""
    refactor = REFACTOR["compact_tool_json"]
    lenient = study.replay_suite(refactor.agent, refactor.client, CONFIG["tools-live"])
    strict = study.replay_suite(refactor.agent, refactor.client, CONFIG["tools-live+strict"])

    assert lenient.passed
    assert lenient.approximate_steps == 17
    assert not strict.passed
    assert strict.caught_by == ("no_approximate_steps",)
    assert strict.approximate_steps == 17


def test_drift_configuration_ignores_the_request_bytes_strict_fails_on():
    """The JSON-layout refactor changes every model request (17 approximate steps) but no
    tool result, so ``tools-live+drift`` passes it where ``tools-live+strict`` fails it."""
    refactor = REFACTOR["compact_tool_json"]

    drift = study.replay_suite(refactor.agent, refactor.client, CONFIG["tools-live+drift"])

    assert drift.passed
    assert drift.approximate_steps == 17
    assert COMMITTED_ROWS["compact_tool_json"]["results"]["tools-live+drift"] == drift.to_json()


def test_check_types_keep_rule_names_and_drop_arguments():
    assert study.check_type("arg_schema:cancel_pending_order") == "arg_schema"
    assert study.check_type("rule:matches_ground_truth") == "rule:matches_ground_truth"
    assert study.check_type("tool_sequence") == "tool_sequence"


def test_committed_results_cover_the_catalogue_and_summarise_consistently():
    """The full study takes about 20 s, so CI re-runs it (`make eval-mutations`) and diffs
    the committed file; here only its shape and its summary are checked."""
    assert [m["id"] for m in COMMITTED["mutations"]] == [m.id for m in MUTATIONS]
    assert [r["id"] for r in COMMITTED["refactors"]] == [r.id for r in REFACTORS]
    assert [c["id"] for c in COMMITTED["configurations"]] == list(CONFIG)
    summary = study.summarise(
        COMMITTED["mutations"], COMMITTED["refactors"], list(CONFIG), study.suite_check_types()
    )
    assert summary == COMMITTED["summary"]
    markdown = study.to_markdown(COMMITTED)
    assert "stub-everything baseline" in markdown
    rows = len(MUTATIONS) + len(REFACTORS) + len(summary["check_types"])
    assert markdown.count("\n| `") == rows


def test_every_configured_check_type_has_a_row_even_without_catches():
    configured = study.suite_check_types()
    assert "forbidden_tools" in configured
    assert "rule:db_matches_ground_truth" in configured
    assert set(configured) <= set(COMMITTED["summary"]["check_types"])
    zero = COMMITTED["summary"]["check_types"]["forbidden_tools"]
    assert all(cell == {"caught": 0, "alone": 0} for cell in zero.values())
