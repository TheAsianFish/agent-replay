"""M12: how many runs does a flakiness verdict need, on the runs already recorded?

For every scenario of every committed flake report, the verdict on all of its
runs is the fixed-threshold rule those reports ran (the ``baseline`` gate: pass
when the pass rate is at least ``(p_fail + p_pass) / 2``, 0.75; no interval).
For each subset size ``n`` from 1 to ``runs - 1``, :data:`DRAWS` seeded subsets of
``n`` runs are drawn without replacement from the scenario's recorded outcomes,
and the curve is the share of draws whose verdict on those ``n`` runs agrees
with the all-runs verdict. Beside it, the exact share (a hypergeometric count
over every subset, :func:`exact_agreement`) checks the sampling.

A single run's false-pass rate (the all-runs verdict fails, one run passes) is
the observed pass rate, and its false-fail rate is the observed failure rate,
by construction; they are reported, not estimated.

Offline and byte-stable: standard library only, counting and seeded sampling;
``agent_replay.stats`` is not used. The curve says how far each scenario's pass
rate sits from the threshold, not how many runs any agent needs.

Usage::

    uv run python -m eval.verdict_curve [--out eval/results/verdict_curve.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUT = RESULTS / "verdict_curve.json"
DRAWS = 2000
SEED = 0
TARGET = 0.95
"""The agreement the summary asks of a subset size (the plan's hypothesis: n = 5)."""
HYPOTHESIS_N = 5

FLAKE_REPORTS: tuple[tuple[str, str], ...] = (
    ("claude-haiku-5-5 (M4)", "m4_flake_haiku.json"),
    ("claude-opus-5-5 (M4)", "m4_flake_opus.json"),
    ("qwen2.5:7b (M9)", "m9_flake_qwen.json"),
)
"""Committed ``replay flake`` reports: (label, file under ``eval/results``)."""
ARMS_REPORTS: tuple[tuple[str, str], ...] = (("qwen2.5:7b (M12)", "m12_arms_qwen.json"),)
"""Committed M12 arm reports (``eval.pydantic_ai --arms``): one row per arm and scenario."""


@dataclass(frozen=True)
class Scenario:
    """One scenario's recorded outcomes and the threshold its gate used."""

    source: str
    name: str
    outcomes: tuple[bool, ...]
    threshold: float

    @property
    def runs(self) -> int:
        return len(self.outcomes)

    @property
    def passes(self) -> int:
        return sum(self.outcomes)


def verdict(passes: int, runs: int, threshold: float) -> bool:
    """The baseline gate's rule: pass when ``passes / runs`` reaches ``threshold``."""
    return passes / runs >= threshold


def _threshold(settings: dict[str, Any]) -> float:
    return (float(settings["p_fail"]) + float(settings["p_pass"])) / 2


def load_flake(label: str, path: Path) -> list[Scenario]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("stats_impl") != "baseline":
        raise ValueError(f"{path}: the curve replays the baseline gate, not {report['stats_impl']}")
    threshold = _threshold(report["settings"])
    return [
        Scenario(
            label,
            s["name"],
            tuple(bool(o["passed"]) for o in s["outcomes"] if o.get("counted", True)),
            threshold,
        )
        for s in report["scenarios"]
    ]


def load_arms(label: str, path: Path) -> list[Scenario]:
    report = json.loads(path.read_text(encoding="utf-8"))
    for arm, summary in report["arms"].items():
        if summary.get("stats_impl") != "baseline":
            raise ValueError(
                f"{path}: arm {arm!r} ran {summary.get('stats_impl')}, and the curve replays "
                "the baseline gate"
            )
    threshold = _threshold(report["settings"])
    return [
        Scenario(
            f"{label} {arm}",
            s["name"],
            tuple(bool(o["passed"]) for o in s["outcomes"]),
            threshold,
        )
        for arm, summary in sorted(report["arms"].items())
        for s in summary["scenarios"]
    ]


def sampled_agreement(scenario: Scenario, n: int, draws: int, seed: int) -> int:
    """How many of ``draws`` seeded size-``n`` subsets (without replacement) give the
    all-runs verdict."""
    target = verdict(scenario.passes, scenario.runs, scenario.threshold)
    rng = random.Random(f"{seed}:{scenario.source}:{scenario.name}:{n}")  # noqa: S311 - sampling, not secrets
    agree = 0
    for _ in range(draws):
        passes = sum(rng.sample(scenario.outcomes, n))
        agree += verdict(passes, n, scenario.threshold) == target
    return agree


def exact_agreement(scenario: Scenario, n: int) -> float:
    """The share of all size-``n`` subsets that give the all-runs verdict (hypergeometric)."""
    target = verdict(scenario.passes, scenario.runs, scenario.threshold)
    fails = scenario.runs - scenario.passes
    agreeing = sum(
        comb(scenario.passes, k) * comb(fails, n - k)
        for k in range(max(0, n - fails), min(n, scenario.passes) + 1)
        if verdict(k, n, scenario.threshold) == target
    )
    return agreeing / comb(scenario.runs, n)


def stable_from(curve: Sequence[float], target: float) -> int | None:
    """The smallest ``n`` from which every larger subset size agrees at least ``target``
    (``curve[i]`` is size ``i + 1``); ``None`` if not even ``runs - 1`` does."""
    first: int | None = None
    for i, share in enumerate(curve):
        if share < target:
            first = None
        elif first is None:
            first = i + 1
    return first


def scenario_row(scenario: Scenario, draws: int, seed: int) -> dict[str, Any]:
    all_runs = verdict(scenario.passes, scenario.runs, scenario.threshold)
    sizes = range(1, scenario.runs)
    sampled = [round(sampled_agreement(scenario, n, draws, seed) / draws, 4) for n in sizes]
    exact = [round(exact_agreement(scenario, n), 4) for n in sizes]
    rate = scenario.passes / scenario.runs
    return {
        "source": scenario.source,
        "scenario": scenario.name,
        "runs": scenario.runs,
        "passes": scenario.passes,
        "pass_rate": round(rate, 4),
        "threshold": scenario.threshold,
        "distance_from_threshold": round(rate - scenario.threshold, 4),
        "verdict": "pass" if all_runs else "fail",
        "single_run_false_pass": 0.0 if all_runs else round(rate, 4),
        "single_run_false_fail": round(1 - rate, 4) if all_runs else 0.0,
        f"agreement_at_{HYPOTHESIS_N}": (
            sampled[HYPOTHESIS_N - 1] if scenario.runs > HYPOTHESIS_N else None
        ),
        "stable_from": stable_from(sampled, TARGET),
        "sampled": sampled,
        "exact": exact,
    }


def curve(scenarios: Sequence[Scenario], draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    rows = [scenario_row(s, draws, seed) for s in scenarios if s.runs >= 2]
    held = [r for r in rows if r[f"agreement_at_{HYPOTHESIS_N}"] is not None]
    return {
        "command": "uv run python -m eval.verdict_curve",
        "method": (
            f"per scenario, {draws} seeded subsets of n runs drawn without replacement for "
            "each n in 1..runs-1; agreement is the share whose fixed-threshold verdict "
            "(pass rate >= (p_fail + p_pass) / 2) equals the verdict on all runs; 'exact' "
            "counts every subset (hypergeometric); stable_from is the smallest n from which "
            f"every larger n agrees in at least {TARGET:.0%} of draws"
        ),
        "draws": draws,
        "seed": seed,
        "target": TARGET,
        "hypothesis": (
            f"{HYPOTHESIS_N} runs agree with the all-runs verdict in at least {TARGET:.0%} of "
            "draws on every scenario"
        ),
        "hypothesis_holds": all(r[f"agreement_at_{HYPOTHESIS_N}"] >= TARGET for r in held),
        "scenarios_at_0_or_100": sum(1 for r in rows if r["passes"] in (0, r["runs"])),
        "scenarios": rows,
    }


def load_all(results: Path = RESULTS) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for label, name in FLAKE_REPORTS:
        scenarios += load_flake(label, results / name)
    for label, name in ARMS_REPORTS:
        scenarios += load_arms(label, results / name)
    return scenarios


def to_markdown(result: dict[str, Any]) -> str:
    n = HYPOTHESIS_N
    lines = [
        f"| source | scenario | passes | pass rate - threshold | verdict | agree at n={n} "
        "| stable from n | single-run false pass | single-run false fail |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["scenarios"]:
        at = r[f"agreement_at_{n}"]
        lines.append(
            f"| {r['source']} | `{r['scenario']}` | {r['passes']}/{r['runs']} "
            f"| {r['distance_from_threshold']:+.2f} | {r['verdict']} "
            f"| {'—' if at is None else f'{at:.1%}'} "
            f"| {r['stable_from'] if r['stable_from'] is not None else 'never'} "
            f"| {r['single_run_false_pass']:.0%} | {r['single_run_false_fail']:.0%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    result = curve(load_all(), draws=args.draws)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(to_markdown(result))
    print(
        f"\nhypothesis ({result['hypothesis']}): "
        f"{'holds' if result['hypothesis_holds'] else 'refuted'}; "
        f"{result['scenarios_at_0_or_100']} of {len(result['scenarios'])} scenarios pass 0% "
        f"or 100% of their runs; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
