"""The M12 arms harness (``eval/pydantic_ai.py --arms``), offline against the fake model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic

from agent_replay.cassette import ErrorInfo
from agent_replay.fake import FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.flake import RunOutcome
from eval import pydantic_ai
from examples.pydantic_ai_agent import fake_model
from examples.pydantic_ai_agent.tasks import task_for

FAKE = "examples.pydantic_ai_agent.fake_model:make_client"


def _lower_case_passenger(request: FakeRequest) -> Reply:
    """The fake model, except that it books for "ana lima": the same person to the
    tool and the checks, another entity to the suite's identify keys."""
    reply = fake_model.policy(request)
    calls = [
        ToolCall(c.name, {**c.input, "passenger": "ana lima"}) if c.name == "book_flight" else c
        for c in reply.tool_calls
    ]
    return Reply(text=reply.text, tool_calls=calls)


def _wrong_figure(request: FakeRequest) -> Reply:
    """The fake model, except that its euro answer is off: every step exact, a check fails."""
    reply = fake_model.policy(request)
    if task_for(request.first_user_text()).name == "cheapest_fare_in_euros" and reply.text:
        return Reply(text="The cheapest is B6416 at 1.00 EUR.")
    return reply


def lower_case_passenger_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_lower_case_passenger).as_anthropic()


def wrong_figure_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_wrong_figure).as_anthropic()


def _outcome(**fields: object) -> RunOutcome:
    return RunOutcome.model_validate({"index": 0, "seed": 0, "passed": False, **fields})


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"passed": True}, "passed"),
        ({"passed": True, "steps": {"approximate": 1, "live": 3}}, "passed"),
        (
            {
                "error": ErrorInfo(type="UnrecordedCallError", message="no recording"),
                "steps": {"approximate": 1},
                "failed_checks": ("rule:answer_states",),
            },
            "unrecorded",
        ),
        (
            {"steps": {"approximate": 1, "exact": 1}, "failed_checks": ("tool_sequence",)},
            "approximate",
        ),
        ({"steps": {"exact": 2, "live": 3}, "failed_checks": ("rule:answer_states",)}, "model"),
        ({"error": ErrorInfo(type="UnexpectedModelBehavior", message="retries")}, "model"),
        ({"steps": {"fallback": 1, "live": 3}, "failed_checks": ("tool_sequence",)}, "model"),
        ({"steps": {"synth": 1, "live": 3}, "failed_checks": ("rule:answer_states",)}, "synth"),
        ({"passed": True, "steps": {"synth": 1, "live": 3}}, "passed"),
        (
            {"steps": {"approximate": 1, "synth": 1}, "failed_checks": ("tool_sequence",)},
            "approximate",
        ),
    ],
)
def test_cause_precedence(fields: dict[str, object], expected: str) -> None:
    assert pydantic_ai.cause(_outcome(**fields)) == expected


def _arms(tmp_path: Path, client: str) -> dict[str, dict[str, dict[str, object]]]:
    out = tmp_path / "arms.json"
    result = pydantic_ai.run_arms(("strict", "fallback"), 2, out, client=client, environment={})
    assert json.loads(out.read_text(encoding="utf-8")) == result
    return {
        arm: {s["name"]: s for s in summary["scenarios"]} for arm, summary in result["arms"].items()
    }


def test_the_fake_model_passes_both_arms_on_the_qwen_recordings(tmp_path: Path) -> None:
    arms = _arms(tmp_path, FAKE)
    for arm in ("strict", "fallback"):
        assert set(arms[arm]) == set(pydantic_ai.ARM_SCENARIOS)
        for s in arms[arm].values():
            assert (s["runs"], s["passes"], s["decision"]) == (2, 2, "pass")
            assert s["causes"] == {
                "passed": 2,
                "unrecorded": 0,
                "approximate": 0,
                "synth": 0,
                "model": 0,
            }


def test_another_entity_is_unrecorded_under_strict_and_runs_the_tool_under_fallback(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path, "tests.test_pydantic_ai_arms:lower_case_passenger_client")
    strict, fallback = (
        arms["strict"]["book_earliest_flight"],
        arms["fallback"]["book_earliest_flight"],
    )
    assert strict["causes"]["unrecorded"] == 2
    assert strict["unrecorded_by_identity"] == 2
    assert all(o["identity"] and "book_flight" in o["error"] for o in strict["outcomes"])
    # The real tool books "ana lima" and the checks ignore case: the run passes.
    assert fallback["causes"]["passed"] == 2
    assert fallback["runs_with_a_fallback_step"] == 2
    assert arms["strict"]["cheapest_fare_in_euros"]["causes"]["passed"] == 2


def test_a_wrong_answer_on_exact_steps_is_the_models_failure(tmp_path: Path) -> None:
    arms = _arms(tmp_path, "tests.test_pydantic_ai_arms:wrong_figure_client")
    for arm in ("strict", "fallback"):
        s = arms[arm]["cheapest_fare_in_euros"]
        assert s["causes"]["model"] == 2
        assert s["failures"] == {"rule:answer_states": 2}
        assert s["decision"] == "fail"


def test_parse_arms() -> None:
    assert pydantic_ai.parse_arms("strict, fallback") == ("strict", "fallback")
    for bad in ("", "strict,live", ","):
        with pytest.raises(argparse.ArgumentTypeError, match="--arms"):
            pydantic_ai.parse_arms(bad)


def test_arms_without_ollama_are_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pydantic_ai, "ollama_model", lambda *_: None)
    out = tmp_path / "arms.json"
    assert pydantic_ai.main(["--arms", "strict,fallback", "--arms-out", str(out)]) == 0
    assert "skipped the model-live runs" in capsys.readouterr().out
    assert not out.exists()
