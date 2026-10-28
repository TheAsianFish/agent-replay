"""Tests for the verdict-stability curve (``eval/verdict_curve.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scipy.stats import hypergeom

from eval import verdict_curve as vc


def _scenario(passes: int, runs: int, name: str = "s") -> vc.Scenario:
    return vc.Scenario("src", name, (True,) * passes + (False,) * (runs - passes), 0.75)


def test_verdict_is_the_baseline_gates_rule() -> None:
    assert vc.verdict(3, 4, 0.75)  # exactly at the threshold passes
    assert not vc.verdict(2, 3, 0.75)
    assert vc.verdict(15, 20, (0.6 + 0.9) / 2)


@pytest.mark.parametrize(("passes", "runs"), [(19, 20), (2, 10), (15, 20), (14, 20), (7, 9)])
def test_exact_agreement_matches_the_hypergeometric_oracle(passes: int, runs: int) -> None:
    s = _scenario(passes, runs)
    target = vc.verdict(passes, runs, 0.75)
    for n in range(1, runs):
        dist = hypergeom(runs, passes, n)
        oracle = sum(dist.pmf(k) for k in range(n + 1) if vc.verdict(k, n, 0.75) == target)
        assert vc.exact_agreement(s, n) == pytest.approx(oracle, abs=1e-12)


def test_sampling_tracks_the_exact_share_and_is_seeded() -> None:
    s = _scenario(15, 20)
    for n in (1, 5, 12, 19):
        share = vc.sampled_agreement(s, n, 4000, seed=1) / 4000
        assert share == pytest.approx(vc.exact_agreement(s, n), abs=0.03)
    assert vc.sampled_agreement(s, 7, 500, 2) == vc.sampled_agreement(s, 7, 500, 2)


def test_unanimous_scenarios_agree_at_every_size() -> None:
    for s in (_scenario(10, 10), _scenario(0, 10)):
        row = vc.scenario_row(s, 200, 0)
        assert row["sampled"] == [1.0] * 9 == row["exact"]
        assert row["stable_from"] == 1


def test_single_run_rates_are_the_observed_rates() -> None:
    passing = vc.scenario_row(_scenario(19, 20), 100, 0)
    assert (passing["single_run_false_pass"], passing["single_run_false_fail"]) == (0.0, 0.05)
    assert passing["exact"][0] == pytest.approx(0.95)
    failing = vc.scenario_row(_scenario(2, 10), 100, 0)
    assert (failing["single_run_false_pass"], failing["single_run_false_fail"]) == (0.2, 0.0)
    assert failing["distance_from_threshold"] == pytest.approx(-0.55)


def test_stable_from_needs_every_larger_size() -> None:
    assert vc.stable_from([0.96, 0.9, 0.97, 1.0], 0.95) == 3
    assert vc.stable_from([1.0, 1.0], 0.95) == 1
    assert vc.stable_from([0.9, 0.99, 0.5], 0.95) is None


def test_a_threshold_hugging_rate_breaks_the_hypothesis() -> None:
    result = vc.curve([_scenario(20, 20, "a"), _scenario(15, 20, "b")], draws=400)
    assert result["scenarios_at_0_or_100"] == 1
    assert not result["hypothesis_holds"]
    assert result["scenarios"][0][f"agreement_at_{vc.HYPOTHESIS_N}"] == 1.0


def test_loads_flake_and_arm_reports(tmp_path: Path) -> None:
    flake = {
        "stats_impl": "baseline",
        "settings": {"p_fail": 0.6, "p_pass": 0.9},
        "scenarios": [
            {
                "name": "x",
                "outcomes": [
                    {"passed": True},
                    {"passed": False},
                    {"passed": True, "counted": False},
                ],
            }
        ],
    }
    arms = {
        "settings": {"p_fail": 0.6, "p_pass": 0.9},
        "arms": {
            "strict": {
                "stats_impl": "baseline",
                "scenarios": [{"name": "y", "outcomes": [{"passed": True}]}],
            },
            "fallback": {
                "stats_impl": "baseline",
                "scenarios": [{"name": "y", "outcomes": [{"passed": False}]}],
            },
        },
    }
    (tmp_path / "f.json").write_text(json.dumps(flake), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(arms), encoding="utf-8")
    [x] = vc.load_flake("F", tmp_path / "f.json")
    assert (x.outcomes, x.threshold) == ((True, False), 0.75)
    loaded = vc.load_arms("A", tmp_path / "a.json")
    assert [(s.source, s.outcomes) for s in loaded] == [
        ("A fallback", (False,)),
        ("A strict", (True,)),
    ]
    arms["arms"]["fallback"]["stats_impl"] = "core"  # type: ignore[index]
    (tmp_path / "a.json").write_text(json.dumps(arms), encoding="utf-8")
    with pytest.raises(ValueError, match="'fallback' ran core"):
        vc.load_arms("A", tmp_path / "a.json")
    flake["stats_impl"] = "core"
    (tmp_path / "f.json").write_text(json.dumps(flake), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline"):
        vc.load_flake("F", tmp_path / "f.json")


def test_the_committed_curve_is_byte_stable(tmp_path: Path) -> None:
    out = tmp_path / "curve.json"
    assert vc.main(["--out", str(out)]) == 0
    assert out.read_bytes() == vc.OUT.read_bytes()
