"""Tests for baselines in ``replay ci``: relative budget limits, reports and exit codes."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.baseline import BaselineFile, ScenarioBaseline, aggregate, measure
from agent_replay.ci import (
    Overrides,
    RunSink,
    ScenarioReport,
    measure_suite,
    run_suite,
    to_json,
    to_markdown,
)
from agent_replay.cli import app
from agent_replay.errors import ConfigError
from agent_replay.runner import RunResult
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).parents[1]
RETAIL = ROOT / "examples" / "retail_agent"
SUITE = RETAIL / "replay.toml"
LIVE = RETAIL / "cassettes" / "live"
RELATIVE_BUDGET = (
    '{ type = "budget", max_cost_increase_pct = 25, max_tokens_increase_pct = 25, '
    "max_p95_latency_increase_pct = 50 }"
)
CREATED = datetime(2026, 10, 9, tzinfo=UTC)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    retail_tools.reset_db()
    yield
    retail_tools.reset_db()


def invoke(*args: str):
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


def live_baseline(scale: float = 1.0) -> BaselineFile:
    """What the live recordings measure, with cost scaled by ``scale``."""
    results: dict[str, RunResult] = {}

    def save(report: ScenarioReport, result: RunResult | None, _started: datetime) -> None:
        assert result is not None
        results[report.name] = result

    run_suite(SUITE, overrides=Overrides(cassette_dir=LIVE), sink=RunSink(save=save))
    scenarios = {}
    for name, result in results.items():
        b = aggregate([measure(result)])
        scenarios[name] = b.model_copy(update={"cost_usd": b.cost_usd * scale})
    return BaselineFile(
        suite=str(SUITE),
        created_at=CREATED,
        mode="full",
        policy="strict",
        scenarios=scenarios,
    )


def live_suite(tmp_path: Path, *, baseline: str | None = "baseline.json") -> Path:
    """The retail suite over the live recordings, with a relative budget check."""
    text = SUITE.read_text(encoding="utf-8").replace('"cassettes/', f'"{LIVE}/')
    defaults = "[defaults]\n"
    if baseline is not None:
        defaults += f'baseline = "{baseline}"\n'
    text = text.replace("[defaults]\n", defaults).replace(
        '{ type = "budget", max_model_calls = 8 },',
        f'{{ type = "budget", max_model_calls = 8 }},\n  {RELATIVE_BUDGET},',
    )
    path = tmp_path / "replay.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_live_recordings_pass_against_their_own_baseline(tmp_path: Path) -> None:
    live_baseline().dump(tmp_path / "baseline.json")

    report = run_suite(live_suite(tmp_path))

    assert report.passed, to_markdown(report)
    for scenario in report.scenarios:
        assert scenario.baseline is not None
        assert scenario.baseline.cost_usd == pytest.approx(scenario.cost_usd)
        relative = [c for c in scenario.checks if c.details.get("max_cost_increase_pct")]
        assert len(relative) == 1
        assert relative[0].details["cost_usd_delta_pct"] == 0.0
        assert "vs baseline: cost_usd +0.0%" in relative[0].message
    data = json.loads(to_json(report))
    assert data["scenarios"][0]["baseline"]["runs"] == 1
    assert "(+0.0%)" in to_markdown(report)


def test_a_lowered_baseline_fails_the_budget_with_exit_1(tmp_path: Path) -> None:
    live_baseline(scale=0.5).dump(tmp_path / "baseline.json")
    suite = live_suite(tmp_path)

    result = invoke("ci", str(suite), "--no-store", "--markdown", str(tmp_path / "s.md"))

    assert result.exit_code == 1, result.output
    assert "is +100.0% over baseline" in result.output
    assert "(limit +25%)" in result.output
    markdown = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "(+100.0%)" in markdown
    assert "**budget**: over budget: cost_usd" in markdown


def test_the_baseline_option_replaces_the_suites_file(tmp_path: Path) -> None:
    live_baseline(scale=0.5).dump(tmp_path / "baseline.json")
    live_baseline().dump(tmp_path / "good.json")
    suite = live_suite(tmp_path)

    lowered = invoke("ci", str(suite), "--no-store")
    good = invoke("ci", str(suite), "--no-store", "--baseline", str(tmp_path / "good.json"))

    assert lowered.exit_code == 1, lowered.output
    assert good.exit_code == 0, good.output


def test_a_scenario_missing_from_the_baseline_cannot_run(tmp_path: Path) -> None:
    baseline = live_baseline()
    scenarios = dict(baseline.scenarios)
    del scenarios["modify_address_suite"]
    baseline.model_copy(update={"scenarios": scenarios}).dump(tmp_path / "baseline.json")
    suite = live_suite(tmp_path)

    report = run_suite(suite)
    broken = {s.name: s for s in report.broken}
    assert set(broken) == {"modify_address_suite"}
    error = broken["modify_address_suite"].error
    assert error is not None
    assert "has no entry for this scenario" in error.message
    assert f"replay baseline SUITE -o {tmp_path / 'baseline.json'}" in error.message
    assert broken["modify_address_suite"].baseline is None

    result = invoke("ci", str(suite), "--no-store")
    assert result.exit_code == 2, result.output


def test_relative_limits_without_a_baseline_file_cannot_run(tmp_path: Path) -> None:
    suite = live_suite(tmp_path, baseline=None)

    report = run_suite(suite)

    assert len(report.broken) == 6
    error = report.broken[0].error
    assert error is not None
    assert "the suite has none" in error.message
    assert "replay baseline SUITE -o FILE" in error.message
    assert 'baseline = "FILE" in [defaults]' in error.message
    result = invoke("ci", str(suite), "--no-store")
    assert result.exit_code == 2, result.output


def test_a_missing_or_broken_baseline_file_breaks_the_suite(tmp_path: Path) -> None:
    suite = live_suite(tmp_path)
    with pytest.raises(ConfigError, match="does not exist; record one with `replay baseline"):
        run_suite(suite)
    missing = invoke("ci", str(suite), "--no-store")
    assert missing.exit_code == 2
    assert "does not exist" in missing.output

    (tmp_path / "baseline.json").write_text('{"schema_version": 1}', encoding="utf-8")
    broken = invoke("ci", str(suite), "--no-store")
    assert broken.exit_code == 2
    assert "is invalid" in broken.output


def test_baseline_path_is_relative_to_the_suite_and_override_to_the_cwd(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    live_baseline().dump(tmp_path / "sub" / "b.json")
    assert run_suite(live_suite(tmp_path, baseline="sub/b.json")).passed
    with pytest.raises(ConfigError, match="does not exist"):
        run_suite(SUITE, overrides=Overrides(baseline=Path("sub/b.json")))  # cwd is the repo


def test_a_suite_without_relative_limits_reports_the_baseline_without_gating(
    tmp_path: Path,
) -> None:
    live_baseline(scale=0.5).dump(tmp_path / "baseline.json")

    report = run_suite(
        SUITE,
        overrides=Overrides(cassette_dir=LIVE, baseline=tmp_path / "baseline.json"),
    )

    assert report.passed, to_markdown(report)
    assert all(s.baseline is not None for s in report.scenarios)
    assert to_markdown(report).count("(+100.0%)") == 6


def test_markdown_cost_cell_against_a_zero_baseline(tmp_path: Path) -> None:
    zero = ScenarioBaseline(cost_usd=0, tokens=0, model_calls=0, runs=1)
    report = run_suite(SUITE)  # fake cassettes are unpriced: cost 0
    scenarios = tuple(s.model_copy(update={"baseline": zero}) for s in report.scenarios)
    markdown = to_markdown(report.model_copy(update={"scenarios": scenarios}))
    assert "$0.0000 (baseline $0)" in markdown


# --- replay baseline ----------------------------------------------------------------------


def failing_agent(client: object, task: str) -> str:
    raise RuntimeError("the agent broke")


@pytest.fixture
def broken_agent(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("baseline_agents_fixture")
    module.failing_agent = failing_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "baseline_agents_fixture", module)
    return "baseline_agents_fixture:failing_agent"


def comparable(baseline: BaselineFile) -> dict[str, object]:
    return baseline.model_dump(exclude={"created_at", "git_sha", "suite"})


def test_measure_suite_matches_what_the_checks_measure() -> None:
    baseline = measure_suite(SUITE, overrides=Overrides(cassette_dir=LIVE), clock=lambda: CREATED)
    assert baseline.created_at == CREATED
    assert (baseline.mode, baseline.policy) == ("full", "strict")
    assert baseline.suite == str(SUITE)
    assert comparable(baseline) == comparable(live_baseline())
    assert sum(b.cost_usd for b in baseline.scenarios.values()) == pytest.approx(0.295, abs=0.001)


def test_measure_suite_ignores_the_suites_own_baseline_and_checks(tmp_path: Path) -> None:
    """The first baseline of a suite with relative limits can be recorded."""
    suite = live_suite(tmp_path, baseline="not-yet-recorded.json")
    with pytest.raises(ConfigError):
        run_suite(suite)
    baseline = measure_suite(suite)
    assert len(baseline.scenarios) == 6


def test_measure_suite_measures_a_run_that_fails_its_checks() -> None:
    """A mutant that fails checks but finishes is still a measurement: no checks run."""
    baseline = measure_suite(
        SUITE, overrides=Overrides(agent="examples.retail_agent.mutants:wrong_write_argument")
    )
    assert len(baseline.scenarios) == 6


def test_measure_suite_rejects_zero_runs() -> None:
    with pytest.raises(ConfigError, match="at least 1"):
        measure_suite(SUITE, runs=0)


def test_baseline_cli_writes_what_measure_suite_computes(tmp_path: Path) -> None:
    out = tmp_path / "out" / "baseline.json"
    result = invoke("baseline", str(SUITE), "--cassette-dir", str(LIVE), "-o", str(out))

    assert result.exit_code == 0, result.output
    written = BaselineFile.load(out)
    expected = measure_suite(SUITE, overrides=Overrides(cassette_dir=LIVE))
    assert comparable(written) == comparable(expected)
    assert written.suite == str(SUITE)
    assert out.read_text(encoding="utf-8") == written.dumps()
    assert "cancel_pending_laptop" in result.output
    assert "$0.0461" in result.output
    assert " ms" in result.output
    assert "Wrote the baseline of 6 scenario(s)" in result.output


def test_baseline_cli_runs_aggregates_several_replays(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    result = invoke(
        "baseline", str(SUITE), "--cassette-dir", str(LIVE), "-o", str(out), "--runs", "3"
    )

    assert result.exit_code == 0, result.output
    written = BaselineFile.load(out)
    single = measure_suite(SUITE, overrides=Overrides(cassette_dir=LIVE))
    for name, b in written.scenarios.items():
        assert b.runs == 3
        # Full replays are deterministic, so the median of three is the single run.
        assert b.model_copy(update={"runs": 1}) == single.scenarios[name]


def test_baseline_cli_records_the_mode_and_policy(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    result = invoke(
        "baseline", str(SUITE), "-o", str(out), "--mode", "tools-live", "--policy", "synth"
    )
    assert result.exit_code == 0, result.output
    written = BaselineFile.load(out)
    assert (written.mode, written.policy) == ("tools-live", "synth")


@pytest.mark.parametrize("args", [("--runs", "0"), ("--runs", "x"), ("--mode", "nope"), ()])
def test_baseline_cli_bad_arguments_exit_2(tmp_path: Path, args: tuple[str, ...]) -> None:
    out = tmp_path / "baseline.json"
    output = ("-o", str(out)) if args else ()
    result = invoke("baseline", str(SUITE), *output, *args)
    assert result.exit_code == 2, result.output
    assert not out.exists()


def test_baseline_cli_broken_suite_or_scenario_exits_2(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    missing = invoke("baseline", str(tmp_path / "nope.toml"), "-o", str(out))
    assert missing.exit_code == 2
    assert "does not exist" in missing.output

    partial = tmp_path / "cassettes"
    partial.mkdir()
    (partial / "cancel_pending_laptop.jsonl").write_bytes(
        (LIVE / "cancel_pending_laptop.jsonl").read_bytes()
    )
    result = invoke("baseline", str(SUITE), "--cassette-dir", str(partial), "-o", str(out))
    assert result.exit_code == 2, result.output
    assert "could not run" in result.output
    assert "modify_address_suite" in result.output
    assert not out.exists()

    no_client = invoke("baseline", str(SUITE), "--mode", "model-live", "-o", str(out))
    assert no_client.exit_code == 2
    assert "needs a client" in no_client.output
    assert not out.exists()


def test_baseline_cli_agent_error_exits_1_and_writes_nothing(
    tmp_path: Path, broken_agent: str
) -> None:
    out = tmp_path / "baseline.json"
    kept = tmp_path / "kept.json"
    kept.write_text("previous", encoding="utf-8")

    result = invoke("baseline", str(SUITE), "--agent", broken_agent, "-o", str(out))
    again = invoke("baseline", str(SUITE), "--agent", broken_agent, "-o", str(kept))

    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "the agent failed in cancel_pending_laptop (RuntimeError: the agent broke)" in flat
    assert not out.exists()
    assert again.exit_code == 1
    assert kept.read_text(encoding="utf-8") == "previous"


def test_baseline_cli_unwritable_output_exits_2(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    target = tmp_path / "file" / "b.json"
    result = invoke("baseline", str(SUITE), "--cassette-dir", str(LIVE), "-o", str(target))
    assert result.exit_code == 2
    assert "cannot write the baseline" in " ".join(result.output.split())


# --- the committed example ----------------------------------------------------------------


def test_the_committed_live_suite_passes_against_its_committed_baseline() -> None:
    report = run_suite(RETAIL / "replay-live.toml")

    assert report.passed, to_markdown(report)
    relative = [
        c for s in report.scenarios for c in s.checks if c.details.get("max_cost_increase_pct")
    ]
    assert len(relative) == 6
    assert all(c.details["skipped"] == [] for c in relative)


def test_the_committed_live_baseline_is_what_replay_baseline_measures() -> None:
    committed = BaselineFile.load(RETAIL / "baseline-live.json")
    measured = measure_suite(SUITE, overrides=Overrides(cassette_dir=LIVE))
    assert comparable(committed) == comparable(measured)
    assert committed.suite == "examples/retail_agent/replay.toml"
