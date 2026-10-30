"""M19: the agent mutations on a real model. The harness (``eval/retail_error.py --task
lookup_latest_order_total``) run against the fake model, the Ollama variant of every
agent mutation, the server-log reader, the id probe and the detection rule."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_replay.runner import load_agent
from eval import retail_error
from eval.mutations import MUTATIONS
from eval.mutations import agents as mutants
from examples.retail_agent import tools
from examples.retail_agent.agent import SINGLE_TURN_NOTE, SYSTEM

FAKE = retail_error.FAKE_CLIENT
LOOKUP = retail_error.SETUPS[retail_error.LOOKUP]
AGENT_MUTATIONS = [m.id for m in MUTATIONS if m.target == "agent"]


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    tools.reset_db()
    yield
    tools.reset_db()


# --- every agent mutation on Ollama ------------------------------------------------------


class _Recorder:
    """A client that records each request and ends the run with a text reply."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        text = SimpleNamespace(type="text", text="done")
        return SimpleNamespace(content=[text], stop_reason="end_turn")


@pytest.mark.parametrize("mutation", AGENT_MUTATIONS)
def test_every_agent_mutation_has_an_ollama_variant_that_asks_for_the_local_model(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLAY_OLLAMA_MODEL", "qwen2.5:3b")
    client = _Recorder()
    run = load_agent(f"eval.mutations.agents:{mutation}_ollama")

    run(client, "I'm Noah Ito, zip code 98187.")

    assert [r["model"] for r in client.requests] == ["qwen2.5:3b"]


def test_the_ollama_variants_keep_the_mutation() -> None:
    copied = _Recorder()
    mutants.strips_policy_ollama(copied, "hi")  # type: ignore[arg-type]
    assert copied.requests[0]["system"] == SINGLE_TURN_NOTE.lstrip() != SYSTEM
    assert mutants.max_turns_too_low_ollama.__self__.max_turns == 3  # type: ignore[attr-defined]
    for mutation in AGENT_MUTATIONS:
        variant = load_agent(f"eval.mutations.agents:{mutation}_ollama").__self__  # type: ignore[attr-defined]
        original = load_agent(next(m.agent for m in MUTATIONS if m.id == mutation)).__self__  # type: ignore[attr-defined]
        assert isinstance(variant, type(original))


def test_the_copied_loop_asks_for_the_agents_model() -> None:
    """Before M19 ``_CopiedLoop`` sent ``claude-opus-5-5`` whatever the agent's model."""
    client = _Recorder()
    mutants.truncates_history(client, "hi")  # type: ignore[arg-type]
    assert client.requests[0]["model"] == "claude-opus-5-5"


def test_arms_for_puts_the_unmutated_agent_first() -> None:
    arms = retail_error.arms_for(["strips_policy"])
    assert arms == (
        retail_error.Arm("unmutated", "examples.retail_agent.ollama:run"),
        retail_error.Arm("strips_policy", "eval.mutations.agents:strips_policy_ollama"),
    )
    # M18's default arms are the same arms.
    assert retail_error.arms_for(retail_error.M18.mutations) == retail_error.ARMS


def test_the_four_that_cannot_act_and_the_eight_that_can_are_every_agent_mutation() -> None:
    assert sorted([*retail_error.M19_MUTATIONS, *retail_error.UNABLE_TO_ACT]) == sorted(
        AGENT_MUTATIONS
    )
    assert set(retail_error.PREDICTED) == set(retail_error.M19_MUTATIONS)


def test_mutations_are_checked_against_the_catalogue() -> None:
    assert retail_error.parse_mutations("strips_policy, truncates_history") == (
        "strips_policy",
        "truncates_history",
    )
    with pytest.raises(Exception, match="agent mutation ids"):
        retail_error.parse_mutations("skips_authentication")  # a model mutation
    with pytest.raises(SystemExit):
        retail_error.main(["--task", "lookup_latest_order_total", "--mutant-runs", "0"])


# --- the scripted model's verdict on the same task ----------------------------------------


def test_the_scripted_model_catches_five_of_the_eight_on_the_lookup_task() -> None:
    scripted = retail_error.scripted_verdicts(retail_error.M19_MUTATIONS)

    caught = sorted(m for m, v in scripted.items() if v["verdict"] == "caught")
    assert caught == [
        "max_turns_too_low",
        "returns_empty_answer",
        "skips_order_lookup",
        "truncates_history",
        "truncates_tool_results",
    ]
    # The one crash catch on this task: the fake model parses the cut order JSON.
    crashes = {m: v["by_a_crash"] for m, v in scripted.items() if v["by_a_crash"]}
    assert crashes == {"truncates_tool_results": ["run_error:JSONDecodeError"]}


def test_the_scripted_verdicts_match_the_committed_mutation_study() -> None:
    study = json.loads((Path(__file__).parents[1] / "eval/results/mutations.json").read_text())
    on_task = {
        m["id"]: retail_error.LOOKUP in m["results"]["model-live"]["failed_scenarios"]
        for m in study["mutations"]
        if m["id"] in retail_error.M19_MUTATIONS
    }
    scripted = retail_error.scripted_verdicts(retail_error.M19_MUTATIONS)
    assert {m: v["verdict"] == "caught" for m, v in scripted.items()} == on_task


# --- the harness, with the fake model playing qwen ----------------------------------------


def _arms(tmp_path: Path, server: retail_error.ServerLog | None = None) -> dict[str, Any]:
    return retail_error.run_arms(
        2,
        tmp_path / "out.json",
        arms=retail_error.arms_for(retail_error.M19_MUTATIONS),
        client=FAKE,
        environment={"cpu": "test"},
        recording_log=tmp_path / "missing.json",
        qwen_cassettes=tmp_path / "no-qwen",
        setup=LOOKUP,
        mutant_runs=1,
        server=server,
    )


def test_the_arms_on_the_fake_model(tmp_path: Path) -> None:
    report = _arms(tmp_path)

    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == report
    assert report["cassette_source"].startswith("fake-model recording")
    assert report["suite"] == "examples/retail_agent/replay-qwen.toml"
    assert list(report["arms"]) == ["unmutated", *retail_error.M19_MUTATIONS]
    assert report["arms"]["unmutated"]["scenarios"][0]["runs"] == 2
    assert report["arms"]["strips_policy"]["scenarios"][0]["runs"] == 1

    detection = report["detection"]
    assert detection["resolvable"] is True
    verdicts = {m: row["verdict"] for m, row in detection["mutations"].items()}
    # The fake reads one order per reply, so the two parallel-call mutations never act.
    assert verdicts == {
        "skips_order_lookup": "caught",
        "first_tool_only": "never acted",
        "truncates_tool_results": "caught",
        "max_turns_too_low": "caught",
        "returns_empty_answer": "caught",
        "mismatches_tool_use_ids": "never acted",
        "strips_policy": "survives",
        "truncates_history": "caught",
    }
    acted = [row for row in detection["mutations"].values() if row["acted"]]
    assert len(acted) == 6
    assert all(row["agrees_with_scripted"] for row in acted)
    assert all(row["as_predicted"] for row in acted)
    # Without the replay.toml checks the crash is still the fake's: it parses the cut JSON.
    truncated = detection["mutations"]["truncates_tool_results"]
    assert truncated["by_a_crash"] == ["run_error:JSONDecodeError"]
    assert detection["mutations"]["max_turns_too_low"]["by_a_crash"] == []

    run = report["arms"]["truncates_tool_results"]["scenarios"][0]["outcomes"][0]
    assert run["cause"] == "model"
    assert run["error"].startswith("JSONDecodeError")
    assert run["most_tool_uses_in_a_reply"] == 1
    assert [c["name"] for c in run["tool_calls"]] == [
        "find_user_id_by_name_zip",
        "get_user_details",
    ]
    assert report["window"]["runs"] == 2 + 8
    assert report["unable_to_act"].keys() == retail_error.UNABLE_TO_ACT.keys()


def test_the_report_is_byte_stable_on_the_fake_model(tmp_path: Path) -> None:
    def without_timings(report: dict[str, Any]) -> str:
        report.pop("arms_wall_s")
        for arm in report["arms"].values():
            arm["wall_s"] = 0
            for s in arm["scenarios"]:
                s["run_seconds_total"] = 0
                for row in s["outcomes"]:
                    row["seconds"] = 0
        return json.dumps(report, sort_keys=True)

    assert without_timings(_arms(tmp_path / "a")) == without_timings(_arms(tmp_path / "b"))


def test_the_arms_read_every_prompt_from_the_server_log(tmp_path: Path) -> None:
    log = tmp_path / "ollama.log"
    log.write_text(
        "srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = 'false'\n",
        encoding="utf-8",
    )
    server = retail_error.ServerLog(log)
    report = _arms(tmp_path, server)

    # The fake model logs nothing, so every run reads no request, and says so.
    assert report["server_log"]["n_ctx_slot_line"].startswith("srv    load_model")
    run = report["arms"]["unmutated"]["scenarios"][0]["outcomes"][0]
    assert run["server"] == {
        "prompt_tokens": [],
        "largest_prompt": None,
        "truncated_requests": 0,
        "n_ctx_slot": [],
    }
    assert report["window"]["runs_with_server_counts"] == 0


# --- the server log -----------------------------------------------------------------------

_REQUEST = (
    "slot   operator(): id  0 | task {t} | new prompt, n_ctx_slot = 8192, n_keep = 4, "
    "task.n_tokens = {n}\n"
    "slot      release: id  0 | task {t} | stop processing: n_tokens = {m}, truncated = {cut}\n"
)


def test_the_server_log_is_read_request_by_request(tmp_path: Path) -> None:
    log = tmp_path / "ollama.log"
    log.write_text("before the run\n" + _REQUEST.format(t=0, n=31, m=35, cut=0), encoding="utf-8")
    server = retail_error.ServerLog(log)
    server.skip()
    with log.open("a", encoding="utf-8") as f:
        f.write(_REQUEST.format(t=1, n=2866, m=2900, cut=0))
        f.write(_REQUEST.format(t=2, n=5012, m=5100, cut=1))
        f.write("slot   operator(): id  0 | task 3 | new prompt, n_ctx_slot")  # mid-line

    requests = server.read()

    assert requests == [
        {"n_ctx_slot": 8192, "prompt_tokens": 2866, "truncated": 0},
        {"n_ctx_slot": 8192, "prompt_tokens": 5012, "truncated": 1},
    ]
    assert retail_error.server_facts(requests) == {
        "prompt_tokens": [2866, 5012],
        "largest_prompt": 5012,
        "truncated_requests": 1,
        "n_ctx_slot": [8192],
    }
    # The half-written line is read whole the next time.
    with log.open("a", encoding="utf-8") as f:
        f.write(" = 8192, n_keep = 4, task.n_tokens = 7\n")
    assert server.read() == [{"n_ctx_slot": 8192, "prompt_tokens": 7}]
    assert server.read() == []


def test_a_missing_server_log_reads_as_nothing(tmp_path: Path) -> None:
    server = retail_error.ServerLog(tmp_path / "none.log")
    server.skip()
    assert server.read() == []
    assert server.slot_line() is None


# --- the id probe -------------------------------------------------------------------------


def test_the_id_probe_reads_equal_counts_as_an_id_that_never_arrives() -> None:
    counts = iter([180, 51, 51, 56])
    sent: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        sent.append(request)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=next(counts)))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    probe = retail_error.probe_ids(client, "qwen2.5:7b")  # type: ignore[arg-type]

    assert probe["reading"] == "the id never reaches the model"
    assert (probe["id_changes_prompt"], probe["control_changes_prompt"]) == (False, True)
    ids = [r["messages"][-1]["content"][0]["tool_use_id"] for r in sent]
    assert ids == ["call_1", "call_1", "toolu_01_mismatched_call_9", "call_1"]
    # A template that printed the id would count the mismatched one differently.
    assert len(ids[2]) > len(ids[1])
    assert {r["messages"][1]["content"][0]["id"] for r in sent} == {"call_1"}
    assert len({r["messages"][0]["content"] for r in sent}) == 4


@pytest.mark.parametrize(
    ("counts", "reading"),
    [
        ([180, 51, 53, 56], "the id reaches the model"),
        ([180, 51, 51, 51], "inconclusive: the control did not move the count"),
    ],
)
def test_the_id_probe_readings(counts: list[int], reading: str) -> None:
    replies = iter(counts)
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(usage=SimpleNamespace(input_tokens=next(replies)))
        )
    )
    assert retail_error.probe_ids(client, "m")["reading"] == reading  # type: ignore[arg-type]


# --- the detection rule -------------------------------------------------------------------


def _arm(passes: int, runs: int, *, most: int = 1, errors: int = 0) -> dict[str, Any]:
    outcomes = [
        {
            "failed_checks": [] if i < passes else ["rule:matches_ground_truth"],
            "error": "RuntimeError: agent did not finish" if i >= runs - errors else None,
        }
        for i in range(runs)
    ]
    scenario = {"runs": runs, "passes": passes, "most_tool_uses_in_a_reply": most}
    return {"scenarios": [{**scenario, "outcomes": outcomes}]}


@pytest.mark.parametrize(
    ("passes", "verdict"),
    [(0, "caught"), (1, "caught"), (2, "unresolved"), (3, "unresolved"), (4, "survives")],
)
def test_a_mutation_is_caught_at_one_of_five_and_survives_at_four(
    passes: int, verdict: str
) -> None:
    result = retail_error.detection({"unmutated": _arm(8, 10), "strips_policy": _arm(passes, 5)})
    assert result["resolvable"] is True
    assert result["mutations"]["strips_policy"]["verdict"] == verdict


def test_fewer_than_eight_unmutated_passes_is_not_resolvable() -> None:
    result = retail_error.detection({"unmutated": _arm(7, 10), "strips_policy": _arm(0, 5)})
    assert result["resolvable"] is False
    assert result["verdict"].startswith("not resolvable on ")
    row = result["mutations"]["strips_policy"]
    assert (row["verdict"], row["rule_verdict"]) == ("not resolvable", "caught")
    assert "agrees_with_scripted" not in row
    assert "as_predicted" not in row


def test_a_parallel_call_mutation_that_never_saw_two_calls_never_acted() -> None:
    arms = {
        "unmutated": _arm(10, 10),
        "first_tool_only": _arm(5, 5, most=1),
        "mismatches_tool_use_ids": _arm(5, 5, most=2),
    }
    rows = retail_error.detection(arms)["mutations"]
    assert (rows["first_tool_only"]["verdict"], rows["first_tool_only"]["acted"]) == (
        "never acted",
        False,
    )
    assert rows["mismatches_tool_use_ids"]["verdict"] == "survives"
    # Failures in runs it never acted on are the baseline's, not a catch: the rule's own
    # reading is kept beside the verdict, and no agreement is claimed.
    scripted = {"first_tool_only": {"verdict": "survives", "caught_by": [], "by_a_crash": []}}
    caught = retail_error.detection(
        {"unmutated": _arm(8, 10), "first_tool_only": _arm(1, 5)}, scripted
    )["mutations"]["first_tool_only"]
    assert (caught["verdict"], caught["rule_verdict"]) == ("never acted", "caught")
    assert "as_predicted" not in caught
    assert "agrees_with_scripted" not in caught


def test_detection_sets_each_verdict_beside_the_prediction_and_the_scripted_model() -> None:
    scripted = {
        "strips_policy": {"verdict": "survives", "caught_by": [], "by_a_crash": []},
        "truncates_tool_results": {
            "verdict": "caught",
            "caught_by": ["run_error:JSONDecodeError"],
            "by_a_crash": ["run_error:JSONDecodeError"],
        },
    }
    arms = {
        "unmutated": _arm(9, 10),
        "strips_policy": _arm(0, 5),
        "truncates_tool_results": _arm(0, 5, errors=0),
    }
    rows = retail_error.detection(arms, scripted)["mutations"]

    policy = rows["strips_policy"]
    assert (policy["predicted"], policy["as_predicted"], policy["agrees_with_scripted"]) == (
        "survives",
        False,
        False,
    )
    cut = rows["truncates_tool_results"]
    assert (cut["as_predicted"], cut["agrees_with_scripted"]) == (True, True)
    # Caught by a check, not a crash, where the scripted model crashed.
    assert cut["caught_by"] == ["rule:matches_ground_truth"]
    assert cut["by_a_crash"] == []
    assert cut["scripted"]["by_a_crash"] == ["run_error:JSONDecodeError"]


def test_the_turn_limit_is_not_a_crash() -> None:
    rows = retail_error.detection(
        {"unmutated": _arm(10, 10), "max_turns_too_low": _arm(0, 5, errors=5)}
    )["mutations"]
    assert rows["max_turns_too_low"]["caught_by"] == [
        "rule:matches_ground_truth",
        "run_error:RuntimeError",
    ]
    assert rows["max_turns_too_low"]["by_a_crash"] == []


def test_without_ollama_the_m19_command_names_the_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    serving = retail_error._serving
    monkeypatch.setattr(retail_error, "_serving", lambda: serving(lambda _model: None))
    assert retail_error.main(["--task", "lookup_latest_order_total"]) == 2
    assert "ollama pull qwen2.5:7b" in capsys.readouterr().err


# --- the committed live report ------------------------------------------------------------

COMMITTED = Path(__file__).parents[1] / "eval" / "results" / "m19_retail_mutations_qwen.json"


def test_the_committed_report_recomputes_from_its_runs() -> None:
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))

    assert (
        retail_error.detection(report["arms"], report["scripted"])["mutations"]
        == (report["detection"]["mutations"])
    )
    assert report["scripted"] == retail_error.scripted_verdicts(retail_error.M19_MUTATIONS)
    assert report["window"] == retail_error.window(report["arms"])
    assert set(report["arms"]) == {"unmutated", *retail_error.M19_MUTATIONS}
    for label, arm in report["arms"].items():
        (scenario,) = arm["scenarios"]
        assert scenario["runs"] == (10 if label == "unmutated" else 5)
        assert scenario["passes"] == sum(r["passed"] for r in scenario["outcomes"])
        # Every request of every run was counted by the server, and none was cut.
        assert all(r["server"]["prompt_tokens"] for r in scenario["outcomes"])
        assert scenario["truncated_requests"] == 0


def test_the_committed_report_is_not_resolvable_and_says_so() -> None:
    report = json.loads(COMMITTED.read_text(encoding="utf-8"))
    detection = report["detection"]

    assert detection["unmutated"] == {"passes": 2, "runs": 10}
    assert detection["verdict"] == "not resolvable on qwen2.5:7b for this task"
    assert {row["verdict"] for row in detection["mutations"].values()} == {"not resolvable"}
    # No reply held two tool_use blocks, so the two parallel-call mutations never acted.
    for mutation in retail_error.ACTS_ON_PARALLEL:
        assert detection["mutations"][mutation]["acted"] is False
    assert report["id_probe"]["reading"] == "the id never reaches the model"
    assert report["server_log"]["n_ctx_slot_line"].endswith(
        "n_ctx_slot = 8192, kv_unified = 'false'"
    )
