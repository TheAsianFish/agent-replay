"""Tests for the judge: rubric, transcript, providers, cache, calibration gate and the check.

Everything runs offline: the Anthropic provider talks to ``FakeAnthropic``, the
Ollama provider to ``FakeOllama``, and suites use the retail example's fake judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import JsonValue

from agent_replay import Cassette, Kind, RunResult, Step, StepStatus, Trajectory, replay
from agent_replay.cassette import ErrorInfo
from agent_replay.errors import ConfigError
from agent_replay.judge import JudgeInput, Rubric
from agent_replay.judge.transcript import clip
from examples.retail_agent import agent as retail_agent
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).parents[1]
RETAIL = ROOT / "examples" / "retail_agent"

RUBRIC_DATA: dict[str, Any] = {
    "name": "refunds",
    "version": 1,
    "description": "Refund handling.",
    "instructions": "Be strict about policy.",
    "criteria": [
        {"id": "resolved", "question": "Was the request resolved?", "pass_when": "it was"},
        {"id": "policy", "question": "Was the policy followed?", "fail_when": "it was not"},
    ],
}


def write_yaml(tmp_path: Path, data: object, name: str = "rubric.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def make_rubric(**changes: Any) -> Rubric:
    return Rubric.model_validate({**RUBRIC_DATA, **changes})


# --- rubric ---------------------------------------------------------------------


def test_a_rubric_loads_from_yaml(tmp_path: Path):
    rubric = Rubric.load(write_yaml(tmp_path, RUBRIC_DATA))

    assert rubric.name == "refunds"
    assert rubric.ids == ("resolved", "policy")
    assert rubric.criteria[0].pass_when == "it was"
    assert rubric.fingerprint.startswith("sha256:")
    text = rubric.render()
    assert "# Rubric: refunds (version 1)" in text
    assert "Be strict about policy." in text
    assert "### policy" in text
    assert "Fail when: it was not" in text


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"name": "Bad Name"}, "name:"),
        ({"version": 0}, "version:"),
        ({"criteria": []}, "criteria:"),
        ({"criteria": [{"id": f"c{i}", "question": "q?"} for i in range(11)]}, "criteria:"),
        ({"criteria": [{"id": "Bad", "question": "q?"}]}, "criteria.0.id:"),
        ({"criteria": [{"id": "a", "question": "  "}]}, "criteria.0.question:"),
        ({"criteria": [{"id": "a"}]}, "criteria.0.question: Field required"),
        (
            {"criteria": [{"id": "a", "question": "q?"}, {"id": "a", "question": "r?"}]},
            "repeated: a",
        ),
        (
            {"criteria": [{"id": "a", "question": "q?", "weight": 2}]},
            "criteria.0.weight: unknown field; use id, question",
        ),
        ({"extra": 1}, "extra: unknown field; use name, version"),
    ],
)
def test_rubric_validation_names_the_file_and_field(tmp_path: Path, changes, fragment):
    path = write_yaml(tmp_path, {**RUBRIC_DATA, **changes})

    with pytest.raises(ConfigError) as info:
        Rubric.load(path)

    assert str(path) in str(info.value)
    assert fragment in str(info.value)


def test_a_missing_or_malformed_rubric_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        Rubric.load(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Rubric.load(bad)
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        Rubric.load(write_yaml(tmp_path, ["a", "list"]))


def test_any_edit_changes_the_rubric_fingerprint():
    base = make_rubric()
    edited_question = make_rubric(
        criteria=[
            {"id": "resolved", "question": "Was it fully resolved?", "pass_when": "it was"},
            RUBRIC_DATA["criteria"][1],
        ]
    )

    assert make_rubric().fingerprint == base.fingerprint
    assert make_rubric(version=2).fingerprint != base.fingerprint
    assert make_rubric(instructions="Be lenient.").fingerprint != base.fingerprint
    assert edited_question.fingerprint != base.fingerprint


# --- transcript -----------------------------------------------------------------


def replay_retail(task_id: str, agent: Any = None) -> RunResult:
    cassette = Cassette.load(RETAIL / "cassettes" / f"{task_id}.jsonl")
    retail_tools.reset_db()
    try:
        return replay(cassette, agent or retail_agent.run)
    finally:
        retail_tools.reset_db()


def test_the_transcript_of_a_replayed_retail_run():
    run = replay_retail("cancel_pending_laptop")
    judge_input = JudgeInput.from_run(run)

    assert judge_input.task == run.task
    assert judge_input.output == run.output
    assert judge_input.error is None
    assert [c.name for c in judge_input.tool_calls] == run.trajectory.tool_names()
    first = judge_input.tool_calls[0]
    assert first.input == {"first_name": "Emma", "last_name": "Smith", "zip": "10192"}
    assert first.result == "emma_smith_8564"
    assert judge_input.tool_call_count == len(run.trajectory.tool_steps())
    assert judge_input.model_call_count == len(run.trajectory.model_steps())
    text = judge_input.render()
    assert "<task>\n" + run.task in text
    assert "## Call 1: find_user_id_by_name_zip" in text
    assert '"first_name": "Emma"' in text  # pretty-printed JSON
    assert "<final_output>\nDone: order #W2417020 is now cancelled.\n</final_output>" in text
    # The same run gives the same transcript, so a cached verdict applies.
    again = JudgeInput.from_run(replay_retail("cancel_pending_laptop"))
    assert again == judge_input
    assert again.fingerprint == judge_input.fingerprint


def test_long_values_are_clipped_with_a_visible_marker():
    run = replay_retail("exchange_waterproof_boots")
    judge_input = JudgeInput.from_run(run)

    product = next(c for c in judge_input.tool_calls if c.name == "get_product_details")
    assert isinstance(product.result, str)
    assert product.result.endswith("…[truncated 891 chars]")
    assert len(product.result) == 2000 + len("…[truncated 891 chars]")
    roomy = JudgeInput.from_run(run, max_chars=10_000)
    roomy_product = next(c for c in roomy.tool_calls if c.name == "get_product_details")
    assert isinstance(roomy_product.result, dict)
    assert roomy.fingerprint != judge_input.fingerprint


def test_clip():
    assert clip("abcdef", 3) == "abc…[truncated 3 chars]"
    assert clip("abc", 3) == "abc"
    assert clip({"a": 1}, 100) == {"a": 1}
    assert clip([1, 2, 3], 4) == "[1, …[truncated 5 chars]"
    with pytest.raises(ValueError, match="max_chars"):
        JudgeInput.from_run(make_run(), max_chars=0)


Call = tuple[str, JsonValue, JsonValue, ErrorInfo | None]


def make_run(
    tools: tuple[Call, ...] = (),
    *,
    output: JsonValue = "done",
    error: ErrorInfo | None = None,
    task: str = "Refund order A1",
) -> RunResult:
    steps = [
        Step(
            index=i,
            kind=Kind.TOOL,
            name=name,
            status=StepStatus.EXACT,
            fingerprint="fp",
            request=request,
            response=response,
            error=step_error,
        )
        for i, (name, request, response, step_error) in enumerate(tools)
    ]
    return RunResult(
        mode="full",
        task=task,
        output=output,
        error=error,
        trajectory=Trajectory(steps=tuple(steps)),
    )


def test_errors_appear_in_the_transcript():
    run = make_run(
        (("lookup", {"id": "A1"}, None, ErrorInfo(type="ValueError", message="no such order")),),
        output=None,
        error=ErrorInfo(type="RuntimeError", message="gave up"),
    )
    judge_input = JudgeInput.from_run(run)

    assert judge_input.tool_calls[0].error is not None
    assert judge_input.tool_calls[0].result is None
    text = judge_input.render()
    assert "Error (ValueError): no such order" in text
    assert "<agent_error>\nThe agent did not finish: RuntimeError: gave up" in text
    assert "(the agent made no tool calls)" in JudgeInput.from_run(make_run()).render()
