"""Simulate the sequential CI gate over a grid of true pass rates.

For every true pass rate ``p`` in 0.05, 0.10, ..., 1.00 (plus ``p_fail`` and
``p_pass`` if they are off the grid) the script feeds ``--trials`` independent
sequences of Bernoulli(``p``) run outcomes to a fresh ``SequentialGate`` per
sequence and records how many runs the gate used and whether it said "pass".
Each requested implementation (``baseline``, ``core``) sees the same outcome
sequences (common random numbers), so differences between them are not noise
from the simulation.

Reported per point: mean runs used and P(gate says pass). Summary: expected runs
at ``p_fail`` and at ``p_pass``, the false-pass rate at ``p = p_fail`` and the
false-fail rate at ``p = p_pass``.

Usage::

    uv run python eval/sprt_sim.py --out eval/results/sprt_sim.json

Exit status: 0 when at least one implementation ran, 2 when none did (for
example only ``core`` was requested and it is not implemented yet).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_replay import stats_impl
from agent_replay.stats_impl import StatsImpl


@dataclass(frozen=True)
class GateParams:
    """The gate's configuration, shared by every simulated sequence."""

    p_fail: float = 0.6
    p_pass: float = 0.9
    alpha: float = 0.05
    beta: float = 0.05
    max_runs: int = 20


@dataclass(frozen=True)
class Point:
    """Simulation result at one true pass rate."""

    p: float
    mean_runs: float
    p_says_pass: float


@dataclass(frozen=True)
class Summary:
    """Headline numbers for one implementation."""

    expected_runs_at_p_fail: float
    expected_runs_at_p_pass: float
    false_pass_rate_at_p_fail: float
    false_fail_rate_at_p_pass: float


def default_grid(params: GateParams) -> list[float]:
    """0.05, 0.10, ..., 1.00, plus ``p_fail`` and ``p_pass``, sorted and de-duplicated."""
    grid = {round(0.05 * k, 2) for k in range(1, 21)}
    grid |= {round(params.p_fail, 4), round(params.p_pass, 4)}
    return sorted(grid)


def simulate_point(
    impl: StatsImpl, p: float, params: GateParams, *, trials: int, rng: random.Random
) -> Point:
    """Run ``trials`` gates on Bernoulli(``p``) outcomes drawn from ``rng``.

    Each trial draws all ``max_runs`` outcomes up front and feeds the gate as
    many as it asks for, so the stream a trial sees does not depend on how
    early an earlier trial's gate stopped: every implementation gets the same
    outcomes at the same trial.
    """
    total_runs = 0
    passes = 0
    for _ in range(trials):
        outcomes = [rng.random() < p for _ in range(params.max_runs)]
        gate = impl.SequentialGate(
            params.p_fail,
            params.p_pass,
            alpha=params.alpha,
            beta=params.beta,
            max_runs=params.max_runs,
        )
        decision = "continue"
        for outcome in outcomes:
            decision = gate.update(outcome)
            if decision != "continue":
                break
        if decision == "continue":
            raise RuntimeError(
                f"the gate did not decide after max_runs={params.max_runs} "
                "updates; the gate must truncate at max_runs"
            )
        total_runs += gate.runs
        passes += decision == "pass"
    return Point(p=p, mean_runs=total_runs / trials, p_says_pass=passes / trials)


def simulate(
    impl: StatsImpl,
    params: GateParams,
    *,
    grid: Sequence[float],
    trials: int,
    seed: int,
) -> list[Point]:
    """One :class:`Point` per grid value; the outcome stream at each point depends only on
    ``seed`` and the point's index, so every implementation sees the same sequences."""
    return [
        simulate_point(
            impl,
            p,
            params,
            trials=trials,
            rng=random.Random(f"{seed}:{i}"),  # noqa: S311 - simulation, not cryptography
        )
        for i, p in enumerate(grid)
    ]


def summarize(points: Sequence[Point], params: GateParams) -> Summary:
    """Expected runs and error rates at ``p_fail`` and ``p_pass`` (both must be on the grid)."""
    by_p = {round(pt.p, 4): pt for pt in points}
    at_fail = by_p[round(params.p_fail, 4)]
    at_pass = by_p[round(params.p_pass, 4)]
    return Summary(
        expected_runs_at_p_fail=at_fail.mean_runs,
        expected_runs_at_p_pass=at_pass.mean_runs,
        false_pass_rate_at_p_fail=at_fail.p_says_pass,
        false_fail_rate_at_p_pass=1 - at_pass.p_says_pass,
    )


def markdown(results: dict[str, list[Point]], params: GateParams, trials: int, seed: int) -> str:
    """The per-point table and the summary lines as Markdown."""
    names = list(results)
    lines = [
        f"SPRT gate simulation: p_fail={params.p_fail}, p_pass={params.p_pass}, "
        f"alpha={params.alpha}, beta={params.beta}, max_runs={params.max_runs}, "
        f"trials={trials}, seed={seed}",
        "",
        "| true p | " + " | ".join(f"{n} mean runs | {n} P(pass)" for n in names) + " |",
        "|---:|" + "---:|---:|" * len(names),
    ]
    for i, p in enumerate(results[names[0]]):
        cells = []
        for name in names:
            pt = results[name][i]
            cells += [f"{pt.mean_runs:.2f}", f"{pt.p_says_pass:.3f}"]
        lines.append(f"| {p.p:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    for name in names:
        s = summarize(results[name], params)
        lines.append(
            f"- {name}: expected runs {s.expected_runs_at_p_fail:.2f} at p_fail, "
            f"{s.expected_runs_at_p_pass:.2f} at p_pass; "
            f"false-pass rate at p_fail {s.false_pass_rate_at_p_fail:.3f}, "
            f"false-fail rate at p_pass {s.false_fail_rate_at_p_pass:.3f}"
        )
    return "\n".join(lines)


def to_json(results: dict[str, list[Point]], params: GateParams, trials: int, seed: int) -> str:
    """Parameters, per-point results and summaries as JSON (values rounded to 6 places)."""

    def rounded(d: dict[str, float]) -> dict[str, float]:
        return {k: round(v, 6) for k, v in d.items()}

    payload = {
        "parameters": {**asdict(params), "trials": trials, "seed": seed},
        "results": {
            name: {
                "summary": rounded(asdict(summarize(points, params))),
                "points": [rounded(asdict(pt)) for pt in points],
            }
            for name, points in results.items()
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--impl",
        action="append",
        choices=stats_impl.IMPLS,
        help="implementation to simulate (repeatable; default: all)",
    )
    parser.add_argument("--trials", type=int, default=2000, help="sequences per true p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-fail", type=float, default=0.6)
    parser.add_argument("--p-pass", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--out", type=Path, help="also write the results as JSON here")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    params = GateParams(
        p_fail=args.p_fail,
        p_pass=args.p_pass,
        alpha=args.alpha,
        beta=args.beta,
        max_runs=args.max_runs,
    )
    grid = default_grid(params)
    results: dict[str, list[Point]] = {}
    for name in args.impl or stats_impl.IMPLS:
        impl = stats_impl.load(name)
        try:
            results[name] = simulate(impl, params, grid=grid, trials=args.trials, seed=args.seed)
        except NotImplementedError:
            print(f"{name}: agent_replay/stats.py is not implemented yet; skipped", file=sys.stderr)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if not results:
        print("error: no statistics implementation ran; nothing to report", file=sys.stderr)
        return 2

    print(markdown(results, params, args.trials, args.seed))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_json(results, params, args.trials, args.seed), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
