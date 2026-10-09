"""Tests for ``replay ci``: suites, reports, the store, exit codes, and the seeded regressions."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.ci import Overrides, load_suite, run_suite, to_json, to_markdown
from agent_replay.cli import app
from agent_replay.errors import ConfigError
from agent_replay.store import Store
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay.toml"
RETAIL_CASSETTES = ROOT / "examples" / "retail_agent" / "cassettes"
FAKE_RETAIL = "examples.retail_agent.fake_model:make_client"
MUTANTS = "examples.retail_agent.mutants"
WRITE_SCENARIOS = {
    "cancel_pending_laptop",
    "modify_address_suite",
    "return_damaged_bicycle",
    "exchange_waterproof_boots",
}
ENV = {"COLUMNS": "200"}

runner = CliRunner()


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    retail_tools.reset_db()
    yield
    retail_tools.reset_db()


def invoke(*args: str):
    return runner.invoke(app, list(args), env=ENV)


def write_suite(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "replay.toml"
    path.write_text(body, encoding="utf-8")
    return path


TOY_SUITE = f"""
[defaults]
checks = [{{ type = "tool_sequence" }}]

[[scenario]]
name = "eligible"
cassette = "{ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl"}"
"""


# --- the retail suite -------------------------------------------------------------


def test_retail_suite_passes_in_full_mode() -> None:
    report = run_suite(SUITE)

    assert report.passed, to_markdown(report)
    assert len(report.scenarios) == 6
    for scenario in report.scenarios:
        assert scenario.mode == "full"
        assert set(scenario.steps.by_status) == {"exact"}
        assert all(c.passed for c in scenario.checks)
        assert scenario.unconsumed == ()
    check_types = {c.check.split(":")[0] for s in report.scenarios for c in s.checks}
    assert check_types == {
        "tool_sequence",
        "required_tools",
        "forbidden_tools",
        "max_tool_calls",
        "arg_schema",
        "output_schema",
        "rule",
        "budget",
    }


def test_retail_suite_passes_with_the_model_live() -> None:
    """The fake model plays the live model; tools are served from the cassettes."""
    report = run_suite(SUITE, overrides=Overrides(mode="model-live", client=FAKE_RETAIL))

    assert report.passed, to_markdown(report)
    for scenario in report.scenarios:
        assert set(scenario.steps.by_status) == {"live", "exact"}


def test_retail_suite_passes_with_the_tools_live() -> None:
    report = run_suite(SUITE, overrides=Overrides(mode="tools-live"))

    assert report.passed, to_markdown(report)
    for scenario in report.scenarios:
        assert set(scenario.steps.by_status) == {"exact", "live"}


def failed_checks(report_scenarios: object) -> dict[str, set[str]]:
    return {
        s.name: {c.check for c in s.failed_checks}
        for s in report_scenarios  # type: ignore[attr-defined]
        if not s.passed
    }


def test_wrong_write_argument_is_caught_by_the_ground_truth_rule() -> None:
    report = run_suite(SUITE, overrides=Overrides(agent=f"{MUTANTS}:wrong_write_argument"))

    assert not report.passed
    failures = failed_checks(report.scenarios)
    assert set(failures) == WRITE_SCENARIOS  # the lookup and the transfer write nothing
    for checks in failures.values():
        assert "rule:matches_ground_truth" in checks
        # The tool names are right; only an argument changed.
        assert "tool_sequence" not in checks


def test_skipped_order_lookup_is_caught_everywhere() -> None:
    report = run_suite(SUITE, overrides=Overrides(agent=f"{MUTANTS}:skips_order_lookup"))

    failures = failed_checks(report.scenarios)
    assert set(failures) == {s.name for s in report.scenarios}
    for name in WRITE_SCENARIOS:
        assert "rule:checked_before_writing" in failures[name]


def test_first_tool_only_is_caught_where_the_recording_used_parallel_calls() -> None:
    report = run_suite(SUITE, overrides=Overrides(agent=f"{MUTANTS}:first_tool_only"))

    failures = failed_checks(report.scenarios)
    assert "lookup_latest_order_total" not in failures  # no parallel calls to drop
    assert len(failures) == 5
    assert all("tool_sequence" in checks for checks in failures.values())


# --- reports ------------------------------------------------------------------------


def test_json_report_round_trips_and_names_failures() -> None:
    report = run_suite(SUITE, overrides=Overrides(agent=f"{MUTANTS}:wrong_write_argument"))

    data = json.loads(to_json(report))

    assert data["passed"] is False
    assert sorted(data["failed"]) == sorted(WRITE_SCENARIOS)
    by_name = {s["name"]: s for s in data["scenarios"]}
    assert by_name["cancel_pending_laptop"]["failed_checks"] == ["rule:matches_ground_truth"]
    assert by_name["lookup_latest_order_total"]["failed_checks"] == []


def test_markdown_summary_has_a_row_per_scenario_and_failure_details() -> None:
    report = run_suite(SUITE, overrides=Overrides(agent=f"{MUTANTS}:first_tool_only"))

    markdown = to_markdown(report)

    assert markdown.startswith("## ❌ Replay: 1/6 scenarios passed")
    assert markdown.count("| `") == 6
    assert "### ❌ `cancel_pending_laptop`" in markdown
    assert "```diff" in markdown
    assert "- get_order_details" in markdown


def test_passing_markdown_has_no_failure_sections() -> None:
    markdown = to_markdown(run_suite(SUITE))
    assert markdown.startswith("## ✅ Replay: 6/6 scenarios passed")
    assert "###" not in markdown


# --- suite files ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not toml [", "not valid TOML"),
        ("[defaults]\nmode = 'full'\n", "scenario"),
        (TOY_SUITE + "surprise = 1\n", "surprise"),
        (TOY_SUITE.replace('checks = [{ type = "tool_sequence" }]', 'mode = "half"'), "mode"),
        (TOY_SUITE + TOY_SUITE.split("[defaults]")[1].split("\n\n", 1)[1], "unique"),
    ],
)
def test_broken_suite_files_name_the_problem(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_suite(write_suite(tmp_path, body))


def test_missing_suite_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_suite(tmp_path / "nope.toml")


def test_a_broken_scenario_fails_alone(tmp_path: Path) -> None:
    body = (
        TOY_SUITE
        + """
[[scenario]]
name = "missing"
cassette = "nope.jsonl"

[[scenario]]
name = "bad check"
cassette = "{c}"
checks = [{{ type = "max_tool_calls", limit = -1 }}]
""".format(c=ROOT / "examples/toy_agent/cassettes/refund_denied.jsonl")
    )
    report = run_suite(write_suite(tmp_path, body))

    by_name = {s.name: s for s in report.scenarios}
    assert by_name["eligible"].passed
    assert by_name["missing"].error is not None
    assert "does not exist" in by_name["missing"].error.message
    assert by_name["bad check"].error is not None
    assert "check #2" in by_name["bad check"].error.message  # after the default tool_sequence
    assert not report.passed


def test_fail_on_approximate(tmp_path: Path) -> None:
    body = TOY_SUITE.replace("[defaults]", "[defaults]\nfail_on_approximate = true").replace(
        'cassette = "', 'agent = "examples.toy_agent.mutants:first_tool_only"\ncassette = "'
    )
    report = run_suite(write_suite(tmp_path, body))

    [scenario] = report.scenarios
    assert "no_approximate_steps" in {c.check for c in scenario.failed_checks}


def test_fail_on_approximate_fires_for_a_single_approximate_step(tmp_path: Path) -> None:
    body = f"""
[defaults]
fail_on_approximate = true
checks = [{{ type = "tool_sequence" }}]

[[scenario]]
name = "one_approximate"
cassette = "{RETAIL_CASSETTES / "cancel_pending_laptop.jsonl"}"
"""
    report = run_suite(
        write_suite(tmp_path, body),
        overrides=Overrides(agent=f"{MUTANTS}:wrong_write_argument"),
    )

    [scenario] = report.scenarios
    assert scenario.steps.by_status["approximate"] == 1
    [failed] = scenario.failed_checks
    assert failed.check == "no_approximate_steps"
    assert failed.message.startswith("1 step(s) were served approximately")
    assert not scenario.passed


def test_model_live_without_a_client_fails_the_scenario(tmp_path: Path) -> None:
    report = run_suite(write_suite(tmp_path, TOY_SUITE), overrides=Overrides(mode="model-live"))
    [scenario] = report.scenarios
    assert scenario.error is not None
    assert "needs a client" in scenario.error.message


# --- the command line -------------------------------------------------------------------


def test_ci_exit_codes_and_reports(tmp_path: Path) -> None:
    ok = invoke("ci", str(SUITE), "--no-store", "--json", str(tmp_path / "ok.json"))
    assert ok.exit_code == 0, ok.output
    assert "6/6 scenario(s) passed" in ok.output
    assert json.loads((tmp_path / "ok.json").read_text())["passed"] is True

    regression = invoke(
        "ci",
        str(SUITE),
        "--no-store",
        "--agent",
        f"{MUTANTS}:wrong_write_argument",
        "--markdown",
        str(tmp_path / "out" / "summary.md"),
    )
    assert regression.exit_code == 1, regression.output
    assert "FAIL cancel_pending_laptop" in regression.output
    assert "rule:matches_ground_truth" in regression.output
    assert (tmp_path / "out" / "summary.md").read_text().startswith("## ❌")


def test_ci_usage_errors_exit_2(tmp_path: Path) -> None:
    missing = invoke("ci", str(tmp_path / "nope.toml"), "--no-store")
    assert missing.exit_code == 2
    assert "does not exist" in missing.output

    bad = invoke("ci", str(write_suite(tmp_path, "x = [")), "--no-store")
    assert bad.exit_code == 2
    assert "not valid TOML" in bad.output


def test_ci_appends_to_the_github_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "step_summary.md"
    summary.write_text("earlier step\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    result = invoke("ci", str(SUITE), "--no-store")

    assert result.exit_code == 0, result.output
    text = summary.read_text()
    assert text.startswith("earlier step\n## ✅ Replay: 6/6")


def test_ci_stores_runs_and_runs_lists_them(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'replay.db'}"

    result = invoke("ci", str(SUITE), "--db", db, "--agent", f"{MUTANTS}:first_tool_only")
    assert result.exit_code == 1

    with Store.open(db) as store:
        [suite] = store.list_suites()
        assert (suite.passed, suite.scenario_count, suite.failed_count, suite.run_count) == (
            False,
            6,
            5,
            6,
        )
        runs = store.list_runs(suite_id=suite.id)
        failing = next(r for r in runs if r.scenario == "cancel_pending_laptop")
        record = store.get_run(failing.id)
    assert not record.passed
    assert "tool_sequence" in {c.check for c in record.checks if not c.passed}

    listed = invoke("runs", "--db", db)
    assert listed.exit_code == 0, listed.output
    assert "cancel_pending_laptop" in listed.output
    shown = invoke("runs", failing.id[:8], "--db", db)
    assert shown.exit_code == 0, shown.output
    assert "FAIL tool_sequence" in shown.output
    assert "get_order_details" in shown.output


def test_runs_with_an_unknown_id_exits_2(tmp_path: Path) -> None:
    result = invoke("runs", "deadbeef00", "--db", f"sqlite:///{tmp_path / 'r.db'}")
    assert result.exit_code == 2
    assert "replay runs" in result.output


def test_ci_resolves_paths_relative_to_the_suite(tmp_path: Path) -> None:
    copy = tmp_path / "suite"
    shutil.copytree(SUITE.parent / "cassettes", copy / "cassettes")
    shutil.copy(SUITE, copy / "replay.toml")

    result = invoke("ci", str(copy / "replay.toml"), "--no-store")

    assert result.exit_code == 0, result.output


# --- replay run with modes and policies -------------------------------------------------


def test_run_model_live_needs_and_uses_a_client(tmp_path: Path) -> None:
    cassette = str(ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl")

    without = invoke("run", cassette, "--mode", "model-live")
    assert without.exit_code == 2
    assert "needs a client" in without.output

    saved = tmp_path / "saved.jsonl"
    with_client = invoke(
        "run",
        cassette,
        "--mode",
        "model-live",
        "--client",
        "examples.toy_agent.fake_model:make_client",
        "--save",
        str(saved),
    )
    assert with_client.exit_code == 0, with_client.output
    assert "live" in with_client.output
    assert saved.exists()
    assert invoke("run", str(saved)).exit_code == 0


def test_run_tools_live_and_policy_flags() -> None:
    cassette = str(ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl")
    result = invoke("run", cassette, "--mode", "tools-live", "--policy", "synth")
    assert result.exit_code == 0, result.output
    assert "mode tools-live, policy synth" in result.output


def test_cassette_dir_override_reads_cassettes_by_file_name(tmp_path: Path) -> None:
    shutil.copytree(RETAIL_CASSETTES, tmp_path / "other")
    (tmp_path / "other" / "modify_address_suite.jsonl").unlink()

    report = run_suite(SUITE, overrides=Overrides(cassette_dir=tmp_path / "other"))

    failed = {s.name: s for s in report.scenarios if not s.passed}
    assert set(failed) == {"modify_address_suite"}
    assert failed["modify_address_suite"].error is not None
    assert str(tmp_path / "other") in failed["modify_address_suite"].error.message


def test_live_opus_recordings_pass_the_suite_offline() -> None:
    """Six real claude-opus-5-5 runs (RESULTS.md), replayed with zero network."""
    report = run_suite(SUITE, overrides=Overrides(cassette_dir=RETAIL_CASSETTES / "live"))

    assert report.passed, to_markdown(report)
    for scenario in report.scenarios:
        assert set(scenario.steps.by_status) == {"exact"}
    assert sum(s.cost_usd for s in report.scenarios) == pytest.approx(0.295, abs=0.001)


# --- review regressions: bad input never crashes with a traceback --------------------


def test_a_client_factory_returning_a_non_client(tmp_path: Path) -> None:
    cassette = str(ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl")
    run = invoke("run", cassette, "--mode", "model-live", "--client", "builtins:object")
    assert run.exit_code == 2, run.output
    assert "must return a model client" in run.output

    ci = invoke(
        "ci", str(SUITE), "--no-store", "--mode", "model-live", "--client", "builtins:object"
    )
    assert ci.exit_code == 1, ci.output
    assert "must return a model client" in ci.output


def test_a_cassette_path_that_is_a_directory(tmp_path: Path) -> None:
    run = invoke("run", str(tmp_path))
    assert run.exit_code == 2
    assert "cannot read cassette" in run.output

    body = TOY_SUITE.replace(
        str(ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl"), str(tmp_path)
    )
    ci = invoke("ci", str(write_suite(tmp_path, body)), "--no-store")
    assert ci.exit_code == 1
    assert "cannot read cassette" in ci.output


def test_an_unwritable_step_summary_only_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing" / "summary.md"))
    result = invoke("ci", str(SUITE), "--no-store")
    assert result.exit_code == 0, result.output
    assert "cannot append to GITHUB_STEP_SUMMARY" in result.output


# --- review regressions: claims the mutation sweep showed were untested ---------------


def test_an_agent_error_fails_the_scenario_even_when_every_check_passes(tmp_path: Path) -> None:
    body = TOY_SUITE.replace(
        'cassette = "', 'agent = "examples.toy_agent.mutants:crash_after_answer"\ncassette = "'
    )
    [scenario] = run_suite(write_suite(tmp_path, body)).scenarios

    assert all(c.passed for c in scenario.checks)
    assert scenario.error is not None
    assert not scenario.passed


def test_scenario_mode_and_policy_override_the_defaults(tmp_path: Path) -> None:
    body = TOY_SUITE + 'mode = "tools-live"\npolicy = "synth"\n'
    [scenario] = run_suite(write_suite(tmp_path, body)).scenarios

    assert (scenario.mode, scenario.policy) == ("tools-live", "synth")
    assert scenario.steps.by_status == {"exact": 3, "live": 3}


def test_the_store_counts_approximate_steps(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'r.db'}"
    invoke("ci", str(write_suite(tmp_path, TOY_SUITE)), "--db", db, "--agent",
           "examples.toy_agent.mutants:first_tool_only")  # fmt: skip
    with Store.open(db) as store:
        [run] = store.list_runs()
    assert run.approximate_count == 2
    assert run.unconsumed_count == 1
