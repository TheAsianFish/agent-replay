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
from eval import retail_error
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
        "Error: user not found",
        "I couldn't find your user ID, Noah.",
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
        # A failure that is not the account's: no write, but not the right reply.
        "The payment did not match the order total.",
        "I couldn't find a laptop in your orders.",
        "I was unable to verify the order status, but I noted your request.",
        "Order #W6729841 was not found.",
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


# --- the live harness, against the fake model -----------------------------------------

FAKE_ARMS = (
    retail_error.Arm("unmutated", "examples.retail_agent.agent:run"),
    retail_error.Arm("drops_error_flag", "eval.mutations.agents:drops_error_flag"),
)


def test_recording_keeps_the_first_passing_attempt(tmp_path: Path):
    log = retail_error.record_task(
        3,
        tmp_path / "log.json",
        tmp_path / "qwen",
        agent="examples.retail_agent.agent:run",
        client=FAKE,
        environment={"cpu": "test"},
        metadata={"model": "fake"},
    )

    assert log["kept"] == str(tmp_path / "qwen" / f"{TASK.id}.jsonl")
    assert len(log["attempts"]) == 1
    attempt = log["attempts"][0]
    assert attempt["passed"] is True
    assert attempt["not_found_no_write"] is True
    assert attempt["error_flags_sent"] == 1
    assert len(attempt["input_tokens"]) == attempt["model_calls"] == 2
    assert json.loads((tmp_path / "log.json").read_text(encoding="utf-8")) == log


def test_the_arms_on_the_fake_model(tmp_path: Path):
    report = retail_error.run_arms(
        2,
        tmp_path / "out.json",
        arms=FAKE_ARMS,
        client=FAKE,
        environment={"cpu": "test"},
        recording_log=tmp_path / "missing.json",
        qwen_cassettes=tmp_path / "no-qwen",
    )

    assert report["cassette_source"].startswith("fake-model recording")
    unmutated = report["arms"]["unmutated"]["scenarios"][0]
    mutant = report["arms"]["drops_error_flag"]["scenarios"][0]
    assert (unmutated["passes"], unmutated["not_found_no_write"]) == (2, 2)
    assert unmutated["runs_with_an_error_flag_sent"] == 2
    # The fake model branches on the flag: without it, it reads the error text as a user id.
    assert (mutant["passes"], mutant["not_found_no_write"]) == (0, 0)
    assert mutant["runs_with_an_error_flag_sent"] == 0
    assert report["hypothesis"]["verdict"] == "the flag matters to the real model"
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("mutant", "unmutated", "verdict"),
    [
        (9, 10, "near-equivalent"),
        (8, 10, "near-equivalent"),
        (8, 5, "partial"),
        (7, 7, "partial"),
        (5, 10, "the flag matters"),
        (3, 3, "inconclusive"),
        (5, 7, "inconclusive"),
    ],
)
def test_hypothesis_thresholds(mutant: int, unmutated: int, verdict: str):
    def arm(count: int) -> dict[str, Any]:
        return {"scenarios": [{"not_found_no_write": count, "runs": 10}]}

    result = retail_error.hypothesis({"unmutated": arm(unmutated), "drops_error_flag": arm(mutant)})
    assert result["verdict"].startswith(verdict)


def test_hypothesis_needs_both_arms():
    arm = {"scenarios": [{"not_found_no_write": 9, "runs": 10}]}
    assert retail_error.hypothesis({"unmutated": arm})["verdict"] == "incomplete"


@pytest.mark.parametrize(
    ("inputs", "outputs", "prefix", "bound"),
    [
        # The recording: a cold first call, then only what the cache lacked.
        ([2876, 43, 33], [44, 28, 28], None, 2876 + 43 + 33 + 44 + 28),
        # A later run: its first call found the prompt in the cache.
        ([12, 43, 33], [44, 28, 28], 2876, 2876 + 43 + 33 + 44 + 28),
        ([2876], [44], None, 2876),
        ([2876, None], [44, 28], None, None),
        ([], [], 2876, None),
    ],
)
def test_prompt_bound(
    inputs: list[int | None], outputs: list[int | None], prefix: int | None, bound: int | None
):
    assert retail_error.prompt_bound(inputs, outputs, prefix) == bound


def test_the_arms_use_a_qwen_recording_and_its_cold_prefix(tmp_path: Path):
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    name = f"{TASK.id}.jsonl"
    (qwen / name).write_bytes((EXAMPLE / "cassettes" / "error" / name).read_bytes())
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"attempts": [{"input_tokens": [5000, 40]}]}), encoding="utf-8")

    report = retail_error.run_arms(
        1,
        tmp_path / "out.json",
        arms=FAKE_ARMS[:1],
        client=FAKE,
        environment={"cpu": "test"},
        recording_log=log,
        qwen_cassettes=qwen,
    )

    assert report["cassette_source"] == "qwen recording"
    assert report["recording_attempts"] == 1
    assert report["prompt_prefix_tokens"] == 5000
    scenario = report["arms"]["unmutated"]["scenarios"][0]
    run = scenario["outcomes"][0]
    assert run["prompt_tokens_bound"] >= 5000
    assert scenario["runs_that_may_reach_the_context_window"] == 1
    # The failed lookup's error is kept with the call.
    assert run["tool_calls"][0]["error"] == "ValueError: user not found"


def test_the_probe_reads_equal_counts_as_a_flag_that_never_arrives():
    counts = iter([180, 51, 51, 55])
    sent: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        sent.append(request)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=next(counts)))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    probe = retail_error.probe_flag(client, "qwen2.5:7b")  # type: ignore[arg-type]

    assert probe["reading"] == "the flag never reaches the model"
    flags = [r["messages"][-1]["content"][0].get("is_error") for r in sent]
    assert flags == [None, True, None, None]
    firsts = [r["messages"][0]["content"] for r in sent]
    assert len({len(f) for f in firsts}) == 1  # same length, different nonce
    assert len(set(firsts)) == 4


def test_without_ollama_the_fix_is_named(capsys: pytest.CaptureFixture[str]):
    assert retail_error._serving(lambda _model: None) is False
    assert "ollama pull qwen2.5:7b" in capsys.readouterr().err


def test_error_flags_sent_counts_the_last_request():
    request = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "x", "is_error": True},
                    {"type": "tool_result", "tool_use_id": "b", "content": "y"},
                ],
            },
        ]
    }
    steps = [SimpleNamespace(request={"messages": []}), SimpleNamespace(request=request)]
    result = SimpleNamespace(trajectory=SimpleNamespace(model_steps=lambda: steps))
    assert retail_error.error_flags_sent(result) == 1  # type: ignore[arg-type]


# --- the qwen2.5:7b recording -----------------------------------------------------------


def test_the_qwen_recording_replays_in_full():
    qwen = EXAMPLE / "cassettes" / "qwen"
    header = Cassette.load(qwen / f"{TASK.id}.jsonl").header

    # With the agent that recorded it: the default agent asks for claude-opus-5-5, so
    # every model request would differ from the recording (approximate).
    report = run_suite(
        SUITE, overrides=Overrides(cassette_dir=qwen, agent="examples.retail_agent.ollama:run")
    )

    assert header.metadata["model"] == "qwen2.5:7b"
    assert header.metadata["provider"] == "ollama"
    assert report.passed
    assert report.scenarios[0].steps.by_status == {"exact": 5}
