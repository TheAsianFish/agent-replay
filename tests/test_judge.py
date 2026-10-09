"""Tests for the judge: rubric, transcript, providers, cache, calibration gate and the check.

Everything runs offline: the Anthropic provider talks to ``FakeAnthropic``, the
Ollama provider to ``FakeOllama``, and suites use the retail example's fake judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_replay.errors import ConfigError
from agent_replay.judge import Rubric

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
