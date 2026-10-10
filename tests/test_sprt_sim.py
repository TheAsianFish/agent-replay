"""Tests for the SPRT gate simulation script (``eval/sprt_sim.py``)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from scipy.stats import binom

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


class _StopsAtOnce:
    """A gate that decides on its first outcome, recording what it saw."""

    seen: ClassVar[list[bool]] = []

    def __init__(self, p_fail: float, p_pass: float, **_: float) -> None:
        self.runs = 0
        self.passes = 0
        self.decision = "continue"

    def update(self, passed: bool) -> str:
        self.seen.append(passed)
        self.runs += 1
        self.passes += passed
        self.decision = "pass" if passed else "fail"
        return self.decision


class _NeverDecides(_StopsAtOnce):
    def update(self, passed: bool) -> str:
        self.runs += 1
        return "continue"


def test_an_early_stopping_gate_sees_the_same_stream_as_the_baseline() -> None:
    # Trial k's outcomes are the k-th block of max_runs draws, however many the
    # gate consumed before, so both implementations face the same sequences.
    _StopsAtOnce.seen = []
    rng = random.Random("7:0")
    expected = [[rng.random() < 0.6 for _ in range(PARAMS.max_runs)][0] for _ in range(25)]  # noqa: RUF015 - the block is drawn whole
    sprt_sim.simulate(
        SimpleNamespace(SequentialGate=_StopsAtOnce), PARAMS, grid=[0.6], trials=25, seed=7
    )  # type: ignore[arg-type]
    assert _StopsAtOnce.seen == expected


def test_a_gate_that_never_decides_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="max_runs"):
        sprt_sim.simulate(
            SimpleNamespace(SequentialGate=_NeverDecides), PARAMS, grid=[0.6], trials=1, seed=0
        )  # type: ignore[arg-type]


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


def _scipy_design(params: sprt_sim.GateParams) -> tuple[int, int]:
    for n in range(1, 200):
        for c in range(n + 1):
            false_pass = binom.sf(c - 1, n, params.p_fail)
            false_fail = binom.cdf(c - 1, n, params.p_pass)
            if false_pass <= params.alpha and false_fail <= params.beta:
                return n, c
    raise AssertionError("no design")


@pytest.mark.parametrize(
    "params",
    [
        sprt_sim.GateParams(),
        sprt_sim.GateParams(p_fail=0.5, p_pass=0.8, alpha=0.1, beta=0.05),
        sprt_sim.GateParams(p_fail=0.7, p_pass=0.95, alpha=0.01, beta=0.1),
    ],
)
def test_fixed_sample_design_matches_scipy(params: sprt_sim.GateParams) -> None:
    design = sprt_sim.fixed_sample_design(params)
    assert (design.runs, design.min_passes) == _scipy_design(params)
    assert design.false_pass_rate <= params.alpha
    assert design.false_fail_rate <= params.beta


def test_default_fixed_sample_design() -> None:
    # p_fail 0.6 vs p_pass 0.9 at 5%/5% needs 24 runs; the fixed-20 majority gate misses alpha.
    design = sprt_sim.fixed_sample_design(sprt_sim.GateParams())
    assert (design.runs, design.min_passes) == (24, 19)
    assert design.false_pass_rate == pytest.approx(0.03997, abs=1e-5)
    assert design.false_fail_rate == pytest.approx(0.02766, abs=1e-5)


def test_fixed_sample_design_reports_an_impossible_target() -> None:
    with pytest.raises(ValueError, match="no fixed-sample test of at most 3 runs"):
        sprt_sim.fixed_sample_design(sprt_sim.GateParams(), limit=3)


def test_main_reports_the_fixed_sample_yardstick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sprt.json"
    assert sprt_sim.main(["--impl", "baseline", "--trials", "10", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "fixed-sample test meeting alpha=0.05, beta=0.05 (exact binomial): 24 runs" in stdout
    assert "runs vs the fixed-sample test: 83% at p_fail, 83% at p_pass" in stdout
    design = json.loads(out.read_text())["fixed_sample_design"]
    assert (design["runs"], design["min_passes"]) == (24, 19)
