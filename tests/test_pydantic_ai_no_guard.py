"""The M17 harness (``eval/pydantic_ai.py --no-guard``), offline against the fake model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anthropic import AsyncAnthropic

from agent_replay.fake import FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from eval import pydantic_ai
from examples.pydantic_ai_agent import fake_model
from examples.pydantic_ai_agent.tasks import task_for

FAKE = "examples.pydantic_ai_agent.fake_model:make_client"
HERE = "tests.test_pydantic_ai_no_guard"
FARE = pydantic_ai.FARE_SCENARIO


def _guess_the_fare(request: FakeRequest) -> Reply:
    """The fake model, except that its first fare reply also converts an amount it has not
    read yet (350 USD), as M15's guardless qwen2.5:7b did; it then answers with whatever
    that conversion returned."""
    reply = fake_model.policy(request)
    if task_for(request.first_user_text()).name != FARE:
        return reply
    if request.latest_result("search_flights") is None:
        guess = ToolCall(
            "convert_currency", {"amount": 350.0, "from_currency": "USD", "to_currency": "EUR"}
        )
        return Reply(text=reply.text, tool_calls=[*reply.tool_calls, guess])
    return reply


def _guess_then_state_the_right_fare(request: FakeRequest) -> Reply:
    """The guess, but the answer states the right fare anyway: a pass the stub never
    earned."""
    reply = _guess_the_fare(request)
    if task_for(request.first_user_text()).name == FARE and reply.text and not reply.tool_calls:
        return Reply(text="The cheapest is B6416 at 173.88 EUR.")
    return reply


def guess_the_fare_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_guess_the_fare).as_anthropic()


def lucky_guess_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_guess_then_state_the_right_fare).as_anthropic()


def _no_guard(tmp_path: Path, client: str = FAKE, runs: int = 2) -> dict[str, Any]:
    out = tmp_path / "no_guard.json"
    result = pydantic_ai.run_no_guard(runs, out, client=client, environment={})
    assert json.loads(out.read_text(encoding="utf-8")) == result
    return result


def _scenario(result: dict[str, Any], arm: str, name: str = FARE) -> dict[str, Any]:
    return next(s for s in result["arms"][arm]["scenarios"] if s["name"] == name)


def test_the_arms_are_the_no_guard_prompt_under_fallback_then_synth(tmp_path: Path) -> None:
    result = _no_guard(tmp_path)
    assert list(result["arms"]) == ["fallback", "synth"]
    for label, arm in result["arms"].items():
        assert (arm["policy"], arm["prompt_variant"], arm["model"]) == (
            label,
            "no_guard",
            pydantic_ai.MODEL,
        )
        assert arm["agent"] == "examples.pydantic_ai_agent.agent:run_no_guard"
        for s in arm["scenarios"]:
            assert (s["runs"], s["passes"]) == (2, 2)
            assert s["no_guard"]["runs_with_a_world_step"] == 0
            for row in s["outcomes"]:
                assert row["guard_in_system"] == [False]
    # The committed M15 strict arm rides along for comparison.
    assert result["m15_strict"][FARE]["passes"] == 2
    assert result["m15_strict"][FARE]["unrecorded_by_identity"] == 18


def test_a_guessed_amount_runs_for_real_under_fallback_and_is_stubbed_under_synth(
    tmp_path: Path,
) -> None:
    result = _no_guard(tmp_path, f"{HERE}:guess_the_fare_client", runs=1)
    fallback = _scenario(result, "fallback")["outcomes"][0]
    assert (fallback["passed"], fallback["cause"]) == (False, "model")
    (converted,) = [c for c in fallback["tool_calls"] if c["status"] == "fallback"]
    assert converted["response"] == 322.0  # 350 USD at 0.92, converted for real
    assert fallback["figures"] == [322.0]
    assert (fallback["stubbed_steps"], fallback["states_stub_figure"]) == (0, False)
    assert fallback["world_steps"] == 1

    synth = _scenario(result, "synth")["outcomes"][0]
    assert (synth["passed"], synth["cause"]) == (False, "synth")
    (stubbed,) = [c for c in synth["tool_calls"] if c["status"] == "synth"]
    assert stubbed["response"] == 0.0
    assert synth["figures"] == [0.0]
    assert synth["stub_numbers"] == [0.0]
    assert synth["states_stub_figure"] is True
    summary = _scenario(result, "synth")["no_guard"]
    assert summary["runs_stating_a_stub_figure"] == summary["runs_with_a_stubbed_step"] == 1
    assert summary["passes_with_a_stubbed_step"] == 0
    assert _scenario(result, "synth")["causes"]["synth"] == 1


def test_a_pass_with_a_stubbed_step_is_reported_with_its_answer(tmp_path: Path) -> None:
    result = _no_guard(tmp_path, f"{HERE}:lucky_guess_client", runs=1)
    synth = _scenario(result, "synth")["no_guard"]
    assert synth["passes_with_a_stubbed_step"] == 1
    (shown,) = synth["stubbed_passes"]
    assert shown["output"] == "The cheapest is B6416 at 173.88 EUR."
    assert "synth" in [c["status"] for c in shown["tool_calls"]]
    assert result["hypotheses"]["passes_with_a_stubbed_step"] == {"runs": 1, "holds": False}


def _arm(passes: int, *, stub_figures: int = 0, stubbed_passes: int = 0) -> dict[str, Any]:
    return {
        "scenarios": [
            {
                "name": FARE,
                "runs": 20,
                "passes": passes,
                "no_guard": {
                    "runs_stating_a_stub_figure": stub_figures,
                    "passes_with_a_stubbed_step": stubbed_passes,
                },
            }
        ]
    }


@pytest.mark.parametrize(
    ("fallback", "verdict"),
    [(0, "stands"), (6, "stands"), (7, "partial"), (14, "partial"), (15, "withdrawn")],
)
def test_the_fallback_fare_reads_m15s_attribution(fallback: int, verdict: str) -> None:
    hypotheses = pydantic_ai.no_guard_hypotheses({"fallback": _arm(fallback)})
    assert hypotheses["fallback_fare"]["m15_attribution"] == verdict
    assert hypotheses["fallback_fare"]["holds"] is (verdict == "stands")
    assert "synth_at_most_fallback" not in hypotheses


def test_the_synth_hypotheses() -> None:
    hypotheses = pydantic_ai.no_guard_hypotheses(
        {"fallback": _arm(3), "synth": _arm(4, stub_figures=10, stubbed_passes=0)}
    )
    assert hypotheses["synth_at_most_fallback"] == {"synth": 4, "fallback": 3, "holds": False}
    assert hypotheses["synth_fare_stub_figures"]["holds"] is True
    assert hypotheses["passes_with_a_stubbed_step"]["holds"] is True
    fewer = pydantic_ai.no_guard_hypotheses({"fallback": _arm(3), "synth": _arm(1, stub_figures=9)})
    assert fewer["synth_at_most_fallback"]["holds"] is True
    assert fewer["synth_fare_stub_figures"]["holds"] is False


@pytest.mark.parametrize(
    ("output", "stated"),
    [
        ("The cheapest is B6416 at $189, or 173.88 EUR.", [189.0, 173.88]),
        ("That comes to €0 (0.00 euros).", [0.0, 0.0]),
        ("Total: USD 1,250.50", [1250.5]),
        ("It departs at 06:10 on 2026-11-05 from gate 12.", []),
        (None, []),
    ],
)
def test_figures_are_numbers_next_to_a_currency(output: str | None, stated: list[float]) -> None:
    row = {"output": output, "tool_calls": []}
    assert pydantic_ai.figures(row)["figures"] == stated


def test_a_stub_figure_is_one_a_stubbed_step_returned() -> None:
    stub = {"name": "book_flight", "input": {}, "status": "synth", "response": {"paid_usd": 0.0}}
    row = {"output": "Booked, you paid $0.00.", "tool_calls": [stub]}
    assert pydantic_ai.figures(row)["states_stub_figure"] is True
    row["output"] = "Booked, you paid $158.00."
    assert pydantic_ai.figures(row)["states_stub_figure"] is False
    # The same figure with no stub behind it is the model's, not the stub's.
    assert pydantic_ai.figures({"output": "0.00 EUR", "tool_calls": []})["states_stub_figure"] is (
        False
    )


def test_m15_reference_is_absent_without_its_report(tmp_path: Path) -> None:
    assert pydantic_ai.m15_no_guard(tmp_path / "missing.json") is None


def test_no_guard_takes_no_other_arms(capsys: pytest.CaptureFixture[str]) -> None:
    for extra in (["--long"], ["--swap"], ["--arms", "strict"], ["--model", "qwen2.5:3b"]):
        with pytest.raises(SystemExit):
            pydantic_ai.main(["--no-guard", *extra])
        assert "--no-guard runs its own arms" in capsys.readouterr().err


def test_no_guard_without_ollama_is_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pydantic_ai, "ollama_model", lambda *_: None)
    out = tmp_path / "no_guard.json"
    assert pydantic_ai.main(["--no-guard", "--no-guard-out", str(out)]) == 0
    assert "skipped the model-live runs" in capsys.readouterr().out
    assert not out.exists()


def test_no_guard_runs_twenty_per_arm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pydantic_ai, "ollama_model", lambda m: {"model": m})
    seen: list[tuple[int, Path]] = []
    monkeypatch.setattr(
        pydantic_ai, "run_no_guard", lambda runs, out, **_: seen.append((runs, out)) or {}
    )
    assert pydantic_ai.main(["--no-guard"]) == 0
    assert seen == [(20, pydantic_ai.NO_GUARD_OUT.resolve())]
