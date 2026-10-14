"""The dashboard's demo seed: real replay commands into a store, and the demo calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay.store import Store
from examples import demo_data


def test_the_demo_calibration_compares_judge_and_deterministic_verdicts():
    report = demo_data.calibration_from_judge_smoke()
    rows = json.loads(demo_data.JUDGE_SMOKE.read_text(encoding="utf-8"))
    assert report.n_items == len(rows) == 24
    assert "not human labels" in report.method
    assert report.judge_id == "anthropic:claude-opus-5-5:effort=low"
    resolved = report.criteria["resolved"]
    agree = sum((r["criteria"]["resolved"] == "pass") == r["deterministic_pass"] for r in rows)
    assert resolved.agreement == agree / 24
    assert -1 <= resolved.ci_low <= resolved.ci_high <= 1
    assert "raw agreement in the kappa column" in report.method  # baseline statistics


def test_a_small_seed_fills_every_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(demo_data, "MUTANTS", ("skips_order_lookup",))
    monkeypatch.setattr(demo_data, "FLAKE_SEEDS", (0,))
    db = f"sqlite:///{tmp_path / 'demo.db'}"

    assert demo_data.main(["--db", db]) == 0

    with Store.open(db) as store:
        suites = store.list_suites(limit=100)
        assert len(suites) == 8  # 4 passing ci, 2 regressions, 2 flake
        assert store.count_runs(passed=False) > 0
        assert len(store.list_flake_reports()) == 2
        assert len(store.list_calibrations()) == 1
        total = store.count_runs()
    assert demo_data.main(["--db", db, "--if-empty"]) == 0
    with Store.open(db) as store:
        assert store.count_runs() == total


def test_a_usage_error_is_an_exit_code_not_a_traceback(capsys: pytest.CaptureFixture[str]):
    assert demo_data.run_replay(["runs", "--limit", "abc", "--db", "sqlite://"]) == 2
    assert "abc" in capsys.readouterr().err


def test_an_unexpected_exit_code_stops_the_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        demo_data, "invocations", lambda db: [(["ci", "missing.toml", "--db", db], demo_data.PASS)]
    )
    assert demo_data.main(["--db", f"sqlite:///{tmp_path / 'demo.db'}"]) == 1
