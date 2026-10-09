"""Tests for the SPRT gate simulation script (``eval/sprt_sim.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_replay import stats, stats_baseline
from eval import sprt_sim

PARAMS = sprt_sim.GateParams(p_fail=0.6, p_pass=0.9, max_runs=7)
GRID = [0.0, 0.6, 0.9, 1.0]


def test_baseline_always_uses_max_runs() -> None:
    points = sprt_sim.simulate(stats_baseline, PARAMS, grid=GRID, trials=50, seed=3)
    assert [pt.p for pt in points] == GRID
    assert all(pt.mean_runs == PARAMS.max_runs for pt in points)
    by_p = {pt.p: pt.p_says_pass for pt in points}
    assert by_p[0.0] == 0.0
    assert by_p[1.0] == 1.0
    assert all(0.0 <= pt.p_says_pass <= 1.0 for pt in points)


def test_simulation_is_deterministic_for_a_seed() -> None:
    first = sprt_sim.simulate(stats_baseline, PARAMS, grid=GRID, trials=30, seed=1)
    assert sprt_sim.simulate(stats_baseline, PARAMS, grid=GRID, trials=30, seed=1) == first


def test_summary_reads_the_p_fail_and_p_pass_points() -> None:
    points = sprt_sim.simulate(stats_baseline, PARAMS, grid=GRID, trials=50, seed=0)
    summary = sprt_sim.summarize(points, PARAMS)
    assert summary.expected_runs_at_p_fail == summary.expected_runs_at_p_pass == 7
    assert summary.false_pass_rate_at_p_fail == points[1].p_says_pass
    assert summary.false_fail_rate_at_p_pass == pytest.approx(1 - points[2].p_says_pass)


def test_default_grid_includes_p_fail_and_p_pass() -> None:
    grid = sprt_sim.default_grid(sprt_sim.GateParams(p_fail=0.33, p_pass=0.9))
    assert grid[0] == 0.05
    assert grid[-1] == 1.0
    assert 0.33 in grid
    assert len(grid) == 21


def test_main_writes_markdown_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sprt.json"
    argv = ["--impl", "baseline", "--trials", "20", "--max-runs", "5", "--out", str(out)]
    assert sprt_sim.main(argv) == 0
    stdout = capsys.readouterr().out
    assert "| true p | baseline mean runs | baseline P(pass) |" in stdout
    assert "- baseline: expected runs 5.00 at p_fail" in stdout
    payload = json.loads(out.read_text())
    assert payload["parameters"]["max_runs"] == 5
    assert payload["parameters"]["trials"] == 20
    points = payload["results"]["baseline"]["points"]
    assert len(points) == 20
    assert {pt["mean_runs"] for pt in points} == {5}


def test_main_skips_an_unimplemented_core(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unimplemented(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError

    monkeypatch.setattr(stats, "SequentialGate", unimplemented)
    assert sprt_sim.main(["--impl", "core", "--trials", "5"]) == 2
    err = capsys.readouterr().err
    assert "core: agent_replay/stats.py is not implemented yet; skipped" in err
    assert sprt_sim.main(["--trials", "5", "--max-runs", "3"]) == 0
    captured = capsys.readouterr()
    assert "baseline mean runs" in captured.out
    assert "core mean runs" not in captured.out
