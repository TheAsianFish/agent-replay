"""The dashboard's demo seed: real replay commands into a store."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_replay.store import Store
from examples import demo_data


def test_the_demo_calibrations_are_real_replay_calibrate_runs():
    plan = [args for args, _ in demo_data.invocations("sqlite://")]
    calibrations = [a for a in plan if a[0] == "calibrate"]

    assert len(calibrations) == 2
    assert all(a[a.index("--reference") + 1] == "objective" for a in calibrations)
    real, fake = calibrations
    assert "--offline" in real
    assert "--provider" not in real
    assert fake[fake.index("--provider") + 1] == demo_data.FAKE_JUDGE
    assert demo_data.FAKE_JUDGE_CACHE not in real


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
        assert len(store.list_calibrations()) == 2
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
