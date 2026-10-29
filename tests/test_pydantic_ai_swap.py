"""The M15 swap harness (``eval/pydantic_ai.py --swap``), offline against the fake model."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from anthropic import AsyncAnthropic

from agent_replay.cassette import Cassette, Kind
from agent_replay.fake import FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from eval import pydantic_ai
from examples.pydantic_ai_agent import agent, fake_model
from examples.pydantic_ai_agent.fake_model import make_client
from examples.pydantic_ai_agent.tasks import task_for

FAKE = "examples.pydantic_ai_agent.fake_model:make_client"
ARMS_TESTS = "tests.test_pydantic_ai_arms"


def _swap(
    tmp_path: Path, client: str = FAKE, arms: tuple[pydantic_ai.Arm, ...] | None = None
) -> dict[str, Any]:
    arms = arms or tuple(dataclasses.replace(a, runs=2) for a in pydantic_ai.SWAP_ARMS)
    out = tmp_path / "swap.json"
    result = pydantic_ai.run_swap(arms, out, client=client, environment={}, probe=None)
    assert json.loads(out.read_text(encoding="utf-8")) == result
    return result


def _scenario(result: dict[str, Any], arm: str, name: str) -> dict[str, Any]:
    return next(s for s in result["arms"][arm]["scenarios"] if s["name"] == name)


def test_the_recording_prompt_is_the_one_the_cassettes_were_recorded_with() -> None:
    for name in pydantic_ai.ARM_SCENARIOS:
        cassette = Cassette.load(pydantic_ai.EXAMPLE / "cassettes" / "qwen" / f"{name}.jsonl")
        first = next(e for e in cassette.envelopes if e.kind is Kind.MODEL)
        assert isinstance(first.request, dict)
        system: Any = first.request["system"]
        assert system[0]["text"] == agent.PROMPT_VARIANTS["recording"].strip()
    assert agent.GUARD in agent.INSTRUCTIONS
    no_guard = agent.PROMPT_VARIANTS["no_guard"]
    assert agent.GUARD not in no_guard
    assert no_guard == agent.INSTRUCTIONS.replace(agent.GUARD, "")


def test_the_live_model_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPLAY_OLLAMA_MODEL", raising=False)
    assert agent.live_model() == agent.MODEL
    monkeypatch.setenv("REPLAY_OLLAMA_MODEL", "qwen2.5:3b")
    assert agent.live_model() == "qwen2.5:3b"
    assert agent.build_agent(make_client()).model.model_name == "qwen2.5:3b"  # type: ignore[union-attr]


def test_temperature_is_left_to_the_model_unless_asked() -> None:
    default: Any = agent.build_agent(make_client()).model_settings
    assert "temperature" not in default
    settings: Any = agent.build_agent(make_client(), temperature=0.0).model_settings
    assert settings == {"max_tokens": 1024, "temperature": 0.0}


def test_every_arm_reaches_the_wire_as_configured(tmp_path: Path) -> None:
    result = _swap(tmp_path)
    assert list(result["arms"]) == [a.label for a in pydantic_ai.SWAP_ARMS]
    for arm in pydantic_ai.SWAP_ARMS:
        summary = result["arms"][arm.label]
        assert [s["name"] for s in summary["scenarios"]] == list(arm.scenarios)
        assert (summary["model"], summary["policy"]) == (arm.model, arm.policy)
        for s in summary["scenarios"]:
            # The fake model ignores model, prompt and temperature: every run passes.
            assert (s["runs"], s["passes"], s["distinct_tool_paths"]) == (2, 2, 1)
            for row in s["outcomes"]:
                assert row["request_models"] == [arm.model]
                assert row["request_temperatures"] == [arm.temperature]
                assert row["guard_in_system"] == [arm.prompt_variant == "recording"]
                assert [c["status"] for c in row["tool_calls"]] == ["exact", "exact"]
    assert result["diff"] is None  # nothing failed


def test_a_guessed_identity_is_counted_and_fallback_runs_the_tool(tmp_path: Path) -> None:
    arms = tuple(dataclasses.replace(a, runs=2) for a in pydantic_ai.SWAP_ARMS[:2])
    result = _swap(tmp_path, f"{ARMS_TESTS}:lower_case_passenger_client", arms)
    strict = _scenario(result, "3b_strict", "book_earliest_flight")
    assert strict["causes"]["unrecorded"] == strict["unrecorded_by_identity"] == 2
    assert [c["status"] for c in strict["outcomes"][0]["tool_calls"]] == ["exact", "unrecorded"]
    fallback = _scenario(result, "3b_fallback", "book_earliest_flight")
    assert fallback["causes"]["passed"] == fallback["runs_with_a_fallback_step"] == 2
    # The first failed 3b strict run is diffed against its cassette.
    diff = result["diff"]
    assert (diff["scenario"], diff["run_index"], diff["cause"]) == (
        "book_earliest_flight",
        0,
        "unrecorded",
    )
    markdown = (tmp_path / diff["markdown"]).read_text(encoding="utf-8")
    assert "book_flight" in markdown
    assert "passenger" in markdown
    assert diff["markdown"] == "swap.diff.md"


def test_the_report_directory_is_created_before_the_first_arm(tmp_path: Path) -> None:
    # The ad hoc default is .replay/arms_swapped.json, absent in a fresh clone: the
    # report must not fail to write after an arm's runs have been spent.
    arm = dataclasses.replace(pydantic_ai.SWAP_ARMS[-1], runs=1)
    out = tmp_path / "fresh" / ".replay" / "arms_swapped.json"
    pydantic_ai.run_swap((arm,), out, client=FAKE, environment={}, probe=None)
    assert json.loads(out.read_text(encoding="utf-8"))["arms"][arm.label]["scenarios"]


def test_identical_and_differing_runs_are_counted(tmp_path: Path) -> None:
    arm = dataclasses.replace(pydantic_ai.SWAP_ARMS[-1], runs=3)
    s = _scenario(_swap(tmp_path, arms=(arm,)), arm.label, "book_earliest_flight")
    assert (s["distinct_tool_paths"], s["distinct_outputs"]) == (1, 1)
    paths = {json.dumps(pydantic_ai.tool_path(r)) for r in s["outcomes"]}
    assert json.loads(paths.pop())[1][0] == "book_flight"


def test_an_arm_the_agent_cannot_run_is_refused() -> None:
    arm = pydantic_ai.Arm("x", "strict", prompt_variant="no_guard", temperature=0.0)
    with pytest.raises(ValueError, match="no agent for prompt variant 'no_guard'"):
        _ = arm.agent
    with pytest.raises(SystemExit):
        pydantic_ai.main(["--arms", "strict", "--prompt-variant", "no_guard", "--temperature", "0"])


def test_the_probe_reports_whether_temperature_is_honoured() -> None:
    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/v1/messages"
        text = "same" if body["temperature"] == 0 else f"text {len(seen)}"
        seen.append(body)
        return {"content": [{"type": "text", "text": text}]}

    seen: list[dict[str, Any]] = []
    probe = pydantic_ai.probe_temperature("qwen2.5:7b", post=post)
    assert [b["temperature"] for b in seen] == list(pydantic_ai.PROBE_TEMPERATURES)
    assert all(b["model"] == "qwen2.5:7b" for b in seen)
    assert (probe["identical_at_0"], probe["identical_at_1"]) == (True, False)
    assert [c["text"] for c in probe["calls"]][:2] == ["same", "same"]


def test_swap_flags_need_arms(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pydantic_ai.main(["--model", "qwen2.5:3b"])
    assert "change an --arms run" in capsys.readouterr().err


def test_swap_without_the_models_is_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    served = {"qwen2.5:7b"}
    monkeypatch.setattr(
        pydantic_ai, "ollama_model", lambda m: {"model": m} if m in served else None
    )
    out = tmp_path / "swap.json"
    assert pydantic_ai.main(["--swap", "--swap-out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "not serving qwen2.5:3b" in printed
    assert "ollama pull qwen2.5:3b" in printed
    assert not out.exists()


def test_an_ad_hoc_arm_never_overwrites_the_committed_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[Path] = []
    monkeypatch.setattr(pydantic_ai, "ollama_model", lambda m: {"model": m})

    def run_swap(arms: object, out: Path, **_: object) -> dict[str, Any]:
        written.append(out)
        return {}

    monkeypatch.setattr(pydantic_ai, "run_swap", run_swap)
    assert pydantic_ai.main(["--arms", "strict", "--model", "qwen2.5:3b"]) == 0
    assert written == [(pydantic_ai.ROOT / ".replay" / "arms_swapped.json").resolve()]
    assert written[0] not in {pydantic_ai.ARMS_OUT, pydantic_ai.SWAP_OUT}


def _search_and_convert_at_once(request: FakeRequest) -> Reply:
    """The fake model, except that its first fare reply also converts a fare it has not
    read yet: two tool calls in one reply, the second with a guessed amount."""
    reply = fake_model.policy(request)
    if task_for(request.first_user_text()).name != "cheapest_fare_in_euros":
        return reply
    if request.latest_result("search_flights") is None:
        guess = ToolCall(
            "convert_currency", {"amount": 150.0, "from_currency": "USD", "to_currency": "EUR"}
        )
        return Reply(text=reply.text, tool_calls=[*reply.tool_calls, guess])
    return reply


def search_and_convert_at_once_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_search_and_convert_at_once).as_anthropic()


def test_a_reply_with_several_tool_calls_is_counted(tmp_path: Path) -> None:
    arm = dataclasses.replace(
        pydantic_ai.SWAP_ARMS[2], runs=1, scenarios=("cheapest_fare_in_euros",)
    )
    client = "tests.test_pydantic_ai_swap:search_and_convert_at_once_client"
    s = _scenario(_swap(tmp_path, client, (arm,)), arm.label, "cheapest_fare_in_euros")
    row = s["outcomes"][0]
    assert row["parallel_tool_replies"] == 1
    assert (row["cause"], row["identity"]) == ("unrecorded", True)
    assert [c["status"] for c in row["tool_calls"]] == ["exact", "unrecorded"]


def test_the_full_replay_ignores_an_exported_live_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REPLAY_OLLAMA_MODEL", "qwen2.5:3b")
    assert pydantic_ai.replay_full()
    assert ", 0 approximate" in capsys.readouterr().out
    assert agent.live_model() == "qwen2.5:3b"  # restored after the replay
