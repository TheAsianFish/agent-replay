"""Flake and calibration reports in the run store, run filters, and the ``replay db`` commands."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay import Cassette
from agent_replay.ci import Overrides
from agent_replay.cli import app
from agent_replay.errors import StoreError
from agent_replay.flake import FlakeReport, FlakeSettings, flake_suite
from agent_replay.judge.calibration import CalibrationReport, CriterionCalibration
from agent_replay.runner import replay
from agent_replay.store import Store
from examples.toy_agent import agent as toy

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay.toml"
FLAKY = "examples.retail_agent.flaky_model:make_client"
BOOTS = "exchange_waterproof_boots"
LAPTOP = "cancel_pending_laptop"
TOY = ROOT / "examples" / "toy_agent" / "cassettes" / "refund_eligible.jsonl"
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
ENV = {"COLUMNS": "200"}

runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(f"sqlite:///{tmp_path / 'r.db'}", clock=lambda: T0) as s:
        yield s


@pytest.fixture(scope="module")
def flake_report() -> FlakeReport:
    return flake_suite(
        SUITE,
        settings=FlakeSettings(max_runs=6, concurrency=2, seed=3),
        overrides=Overrides(mode="model-live", client=FLAKY),
        scenarios=[BOOTS, LAPTOP],
    )


def test_a_flake_report_round_trips(store: Store, flake_report: FlakeReport):
    report_id = store.save_flake(flake_report)

    stored = store.get_flake_report(report_id[:8])
    assert stored.id == report_id
    assert stored.suite == flake_report.suite
    assert stored.passed is flake_report.passed
    assert stored.settings == flake_report.settings
    assert stored.run_count == flake_report.runs
    assert stored.flaky_count == len(flake_report.flaky)
    assert stored.failed_count == len(flake_report.failed)
    assert [s.name for s in stored.scenarios] == [s.name for s in flake_report.scenarios]
    for got, want in zip(stored.scenarios, flake_report.scenarios, strict=True):
        assert (got.ci_low, got.ci_high) == want.interval
        assert got.pass_rate == want.pass_rate
        assert got.classification == want.classification
        assert got.failures == dict(want.failures)
        assert tuple(got.outcomes) == want.outcomes
    assert store.list_flake_reports()[0].id == report_id


def test_flake_trend_is_oldest_first_and_keeps_the_newest_points(
    store: Store, flake_report: FlakeReport
):
    for day in range(3):
        report = flake_report.model_copy(update={"started_at": T0 + timedelta(days=day)})
        store.save_flake(report)

    trend = store.flake_trend()
    assert [p.started_at for p in trend] == sorted(p.started_at for p in trend)
    assert len(trend) == 6
    boots = store.flake_trend(scenario=BOOTS)
    assert [p.name for p in boots] == [BOOTS] * 3
    newest = store.flake_trend(limit=2)
    assert {p.started_at for p in newest} == {T0 + timedelta(days=2)}
    # Within a report, scenarios keep their suite order (the laptop task comes first).
    assert [p.name for p in newest] == [LAPTOP, BOOTS]
    assert [p.name for p in trend[:2]] == [LAPTOP, BOOTS]


def test_timestamps_with_an_offset_are_stored_as_the_same_instant(store: Store):
    created = datetime(2026, 10, 10, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    report_id = store.save_calibration(calibration(created=created))
    (listed,) = store.list_calibrations()
    assert listed.created_at == created
    assert listed.created_at.utcoffset() == timedelta(0)
    assert store.get_calibration(report_id).report.created_at == created

    result = replay(Cassette.load(TOY), toy.run)
    run_id = store.save_run(
        result,
        [],
        scenario="s",
        cassette_path=str(TOY),
        agent="examples.toy_agent.agent:run",
        passed=True,
        started_at=created,
        duration_ms=1.0,
    )
    assert store.get_run(run_id).started_at == created


def test_an_in_memory_store_serves_concurrent_readers(tmp_path: Path):
    # The server runs endpoints in a thread pool; the shared in-memory connection
    # must not be used by two threads at once.
    with Store.open("sqlite://") as store:
        result = replay(Cassette.load(TOY), toy.run)
        run_id = store.save_run(
            result,
            [],
            scenario="s",
            cassette_path=str(TOY),
            agent="examples.toy_agent.agent:run",
            passed=True,
            started_at=T0,
            duration_ms=1.0,
        )

        def read(_: int) -> int:
            return len(store.get_run(run_id).steps) + store.count_runs()

        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = set(pool.map(read, range(200)))
    assert counts == {len(result.trajectory.steps) + 1}


def test_unknown_flake_report_names_the_fix(store: Store):
    with pytest.raises(StoreError, match="no flake report with id 'abcdef123'"):
        store.get_flake_report("abcdef123")


def calibration(rubric: str = "retail", created: datetime = T0) -> CalibrationReport:
    return CalibrationReport(
        rubric=rubric,
        rubric_fingerprint="f" * 16,
        judge_id="fake:default",
        prompt_version="1",
        created_at=created,
        n_items=40,
        method="Cohen's kappa, 95% bootstrap CI (2000 resamples)",
        criteria={
            "resolved": CriterionCalibration(
                kappa=0.7, ci_low=0.55, ci_high=0.85, agreement=0.9, n=40
            )
        },
    )


def test_calibration_reports_round_trip_newest_first(store: Store):
    older = store.save_calibration(calibration(created=T0 - timedelta(days=1)))
    newer = store.save_calibration(calibration(rubric="other"))

    assert [c.id for c in store.list_calibrations()] == [newer, older]
    record = store.get_calibration(older)
    assert record.report == calibration(created=T0 - timedelta(days=1))
    assert record.stored_at == T0
    with pytest.raises(StoreError, match="no calibration report"):
        store.get_calibration("zzzzzzzz")


def test_run_filters_offset_and_counts(store: Store):
    result = replay(Cassette.load(TOY), toy.run)
    for i in range(5):
        store.save_run(
            result,
            [],
            scenario=f"s{i % 2}",
            cassette_path=str(TOY),
            agent="examples.toy_agent.agent:run",
            passed=True,
            started_at=T0 + timedelta(minutes=i),
            duration_ms=1.0,
        )

    assert store.count_runs() == 5
    assert store.count_runs(scenario="s0") == 3
    assert store.count_runs(mode="full", passed=True) == 5
    assert store.count_runs(passed=False) == 0
    page = store.list_runs(limit=2, offset=2)
    assert [r.started_at for r in page] == [T0 + timedelta(minutes=2), T0 + timedelta(minutes=1)]
    assert store.scenarios() == ["s0", "s1"]


def test_replay_flake_stores_its_report(tmp_path: Path):
    db = f"sqlite:///{tmp_path / 'r.db'}"
    result = runner.invoke(
        app,
        [
            "flake",
            str(SUITE),
            "-s",
            BOOTS,
            "--max-runs",
            "4",
            "--client",
            FLAKY,
            "--db",
            db,
        ],
        env=ENV,
    )
    assert result.exit_code in (0, 1), result.output
    with Store.open(db) as store:
        (summary,) = store.list_flake_reports()
        assert summary.scenario_count == 1
        assert summary.suite_id is not None
        assert store.count_runs(suite_id=summary.suite_id) == summary.run_count


def test_db_upgrade_and_add_calibration(tmp_path: Path):
    db = f"sqlite:///{tmp_path / 'r.db'}"
    upgraded = runner.invoke(app, ["db", "upgrade", "--db", db], env=ENV)
    assert upgraded.exit_code == 0, upgraded.output
    assert "schema revision 0002" in upgraded.output

    path = tmp_path / "cal.json"
    path.write_text(calibration().model_dump_json(), encoding="utf-8")
    added = runner.invoke(app, ["db", "add-calibration", str(path), "--db", db], env=ENV)
    assert added.exit_code == 0, added.output
    assert "for rubric retail (40 items, judge fake:default)" in added.output
    with Store.open(db) as store:
        assert len(store.list_calibrations()) == 1

    data = json.loads(path.read_text(encoding="utf-8"))
    data["criteria"]["resolved"]["kappa"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    bad = runner.invoke(app, ["db", "add-calibration", str(path), "--db", db], env=ENV)
    assert bad.exit_code == 2
    assert "criteria.resolved.kappa" in bad.output
