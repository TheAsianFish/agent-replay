"""Tests for baselines in ``replay ci``: relative budget limits, reports and exit codes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.baseline import BaselineFile, ScenarioBaseline, aggregate, measure
from agent_replay.ci import Overrides, RunSink, ScenarioReport, run_suite, to_json, to_markdown
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
