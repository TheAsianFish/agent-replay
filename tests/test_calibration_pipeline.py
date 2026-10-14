"""The calibration pipeline end to end, offline, through the command line.

``replay golden add`` builds a golden set from the retail suite and a seeded
mutant, ``replay label`` takes a human's answers, ``replay calibrate`` measures
the deterministic fake judge against them, and ``replay ci`` runs a suite whose
judge check reads the report: a criterion the labeller agreed with gates the
build, one they disagreed with stays advisory, and a report that may not gate
(baseline statistics) leaves the whole judge advisory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.calibrate.golden import GoldenSet
from agent_replay.cli import app
from agent_replay.judge import Rubric
from examples.retail_agent import tools as retail_tools
from examples.retail_agent.fake_judge import decide

ROOT = Path(__file__).resolve().parents[1]
RETAIL = ROOT / "examples" / "retail_agent"
SUITE = RETAIL / "replay.toml"
RUBRIC = RETAIL / "rubric.yaml"
FAKE_JUDGE = "examples.retail_agent.fake_judge:make_judge"
MUTANT = "examples.retail_agent.mutants:skips_order_lookup"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _retail(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    for name in ("REPLAY_JUDGE_OFFLINE", "REPLAY_JUDGE_PROVIDER", "REPLAY_DB_URL"):
        monkeypatch.delenv(name, raising=False)
    retail_tools.reset_db()
    yield
    retail_tools.reset_db()


def replay(*args: str, input: str | None = None) -> tuple[int, str]:
    result = runner.invoke(app, list(args), input=input, env={"COLUMNS": "200"})
    return result.exit_code, result.output


def build_and_label(golden: Path, rubric: Rubric) -> None:
    """Golden set from the suite and the mutant; the human agrees with the fake judge on
    ``policy_followed`` and answers ``resolved`` against it on every other item."""
    assert replay("golden", "add", str(SUITE), "--out", str(golden))[0] == 0
    code, _ = replay(
        "golden", "add", str(SUITE), "--out", str(golden),
        "--variant", "skips_order_lookup", "--agent", MUTANT,
    )  # fmt: skip
    assert code == 0
    items = sorted(GoldenSet.load(golden).items, key=lambda i: i.id)
    answers = []
    for k, item in enumerate(items):
        verdicts = decide(rubric, item.transcript)
        for criterion in rubric.ids:
            decision = verdicts[criterion]
            passed = decision[0] if isinstance(decision, tuple) else decision
            if criterion == "resolved" and k % 2:
                passed = not passed
            answers.append("p" if passed else "f")
    code, output = replay(
        "label", str(golden), "--rubric", str(RUBRIC), "--labeller", "tester", "--no-shuffle",
        input="\n".join(answers) + "\n",
    )  # fmt: skip
    assert code == 0, output


def suite_with(tmp_path: Path, report: Path) -> Path:
    body = f"""
[defaults]
agent = "examples.retail_agent.agent:run"

[[defaults.checks]]
type = "judge"
rubric = "{RUBRIC}"
provider = "{FAKE_JUDGE}"
cache_dir = "cache"
calibration = "{report}"

[[scenario]]
name = "modify_address"
cassette = "{RETAIL / "cassettes" / "modify_address_suite.jsonl"}"

[[scenario]]
name = "modify_address_without_lookup"
cassette = "{RETAIL / "cassettes" / "modify_address_suite.jsonl"}"
agent = "{MUTANT}"
"""
    path = tmp_path / "replay.toml"
    path.write_text(body, encoding="utf-8")
    return path


def calibrate(golden: Path, out: Path) -> None:
    code, output = replay(
        "calibrate", str(golden), "--rubric", str(RUBRIC), "--provider", FAKE_JUDGE,
        "--labeller", "tester", "--cache-dir", str(out.parent / "cache"), "--online",
        "--no-store", "-o", str(out),
    )  # fmt: skip
    assert code == 0, output


def test_a_calibrated_criterion_gates_the_build(
    tmp_path: Path, core_shim: None, monkeypatch: pytest.MonkeyPatch
):
    rubric = Rubric.load(RUBRIC)
    golden = tmp_path / "golden"
    build_and_label(golden, rubric)
    report_path = tmp_path / "calibration.json"

    calibrate(golden, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reference"] == "human"
    assert report["stats_impl"] == "core"
    assert report["repeats"] == 3
    assert {p["probe"] for p in report["probes"]} == {"position", "verbosity", "self_preference"}
    policy = report["criteria"]["policy_followed"]
    assert policy["kappa"] == 1.0
    assert policy["confusion"]["judge_pass_reference_fail"] == 0
    assert report["criteria"]["resolved"]["ci_low"] < 0.6

    suite = suite_with(tmp_path, report_path)
    code, output = replay("ci", str(suite), "--no-store")

    # The set is small: with the default floor of 20 labelled items nothing gates yet.
    assert policy["n"] < 20
    assert code == 0, output
    assert "advisory: no criterion meets the gate" in output

    monkeypatch.setenv("REPLAY_JUDGE_MIN_ITEMS", str(policy["n"]))
    code, output = replay("ci", str(suite), "--no-store")

    assert code == 1, output  # the mutant skips the order lookup: policy_followed gates
    assert "gating: policy_followed" in output


def test_a_baseline_statistics_report_leaves_the_judge_advisory(tmp_path: Path):
    rubric = Rubric.load(RUBRIC)
    golden = tmp_path / "golden"
    build_and_label(golden, rubric)
    report_path = tmp_path / "calibration.json"

    calibrate(golden, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8"))["stats_impl"] == "baseline"
    code, output = replay("ci", str(suite_with(tmp_path, report_path)), "--no-store")
    assert code == 0, output
    assert "judge:retail-resolution (advisory)" in output
    assert "advisory: calibrated without the core statistics" in output
