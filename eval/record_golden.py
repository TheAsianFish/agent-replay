"""Record D1, the larger golden set: repeated live runs of the retail tasks, ready to label.

Each of the six retail tasks is replayed in ``model-live`` (the model answers for
real, tools are served from the Opus recording, as in ``replay flake``) ``--runs``
times per agent, by default the shipped agent (``claude-opus-5-5``) and its
Haiku variant. A cheaper model fails more often, which gives the golden set
runs of both outcomes, and runs by a model other than the judge's are what the
self-preference probe compares against. Each run is saved as a cassette, then
replayed in ``full`` mode through the retail suite's checks (without a judge) to
become a golden item with its objective label; ``replay label`` adds the human
ones.

Cassettes already on disk are kept, so an interrupted run resumes where it
stopped and costs nothing for the runs it already has. The estimate printed
first uses the measured cost per ``model-live`` run from the M4 flake reports.

Usage::

    uv run python eval/record_golden.py --runs 10 --out eval/golden/retail-d1 [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_replay._logging import configure_logging
from agent_replay.calibrate.build import golden_from_suite
from agent_replay.calibrate.golden import GoldenSet
from agent_replay.cassette import Cassette
from agent_replay.ci import load_suite
from agent_replay.errors import ReplayError
from agent_replay.runner import load_agent, make_client, replay
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay.toml"
RECORDINGS = ROOT / "examples" / "retail_agent" / "cassettes" / "live"
AGENTS = {
    "opus": "examples.retail_agent.agent:run",
    "haiku": "examples.retail_agent.variants:run_haiku",
}
COST_PER_RUN = {"opus": 1.0696 / 30, "haiku": 0.1045 / 120}
"""Measured: total spend over runs of the M4 live flake reports (RESULTS.md)."""


def plan(out: Path, runs: int, agents: Sequence[str]) -> list[tuple[str, str, Path]]:
    """(scenario, agent name, cassette path) for every run the set should hold."""
    suite = load_suite(SUITE)
    return [
        (scenario.name, name, out / "cassettes" / f"{scenario.name}__{name}__{k:02d}.jsonl")
        for scenario in suite.scenario
        for name in agents
        for k in range(runs)
    ]


def record(jobs: list[tuple[str, str, Path]], client: str) -> int:
    """Record every missing cassette; returns how many runs failed to record."""
    suite = {s.name: s for s in load_suite(SUITE).scenario}
    failed = 0
    for scenario, name, path in jobs:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        source = Cassette.load(RECORDINGS / Path(suite[scenario].cassette).name)
        retail_tools.reset_db()
        try:
            result = replay(
                source,
                load_agent(AGENTS[name]),
                mode="model-live",
                client=make_client(client),
                save_to=path,
            )
        except ReplayError as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"{path.name}: {'ok' if result.ok else 'agent error'} ${result.cost_usd:.4f}")
    return failed


def build(jobs: list[tuple[str, str, Path]], out: Path) -> None:
    """Golden items and objective labels from the recorded cassettes, one variant per agent."""
    suite = {s.name: s for s in load_suite(SUITE).scenario}
    golden = GoldenSet.load(out, must_exist=False)
    for name in sorted({n for _, n, _ in jobs}):
        scenarios = [
            suite[scenario].model_copy(
                update={"name": path.stem, "cassette": str(path.resolve()), "agent": AGENTS[name]}
            )
            for scenario, n, path in jobs
            if n == name and path.exists()
        ]
        retail_tools.reset_db()
        made = golden_from_suite(SUITE, variant=name, scenarios=scenarios)
        golden, added = golden.add_items(made.items)
        current = golden.current(source="objective")
        golden = golden.append_labels(
            [
                lb
                for lb in made.labels
                if (lb.item, lb.criterion, "objective", lb.labeller) not in current
            ]
        )
        print(f"{name}: {len(scenarios)} runs, {added} new items, {len(made.skipped)} skipped")
    retail_tools.reset_db()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10, help="runs per task and agent")
    parser.add_argument("--agent", action="append", choices=sorted(AGENTS), dest="agents")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "golden" / "retail-d1")
    parser.add_argument("--client", default="anthropic", help="anthropic or module:factory")
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    args = parser.parse_args(argv)
    configure_logging()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    agents = args.agents or sorted(AGENTS)
    jobs = plan(args.out, args.runs, agents)
    missing = [j for j in jobs if not j[2].exists()]
    cost = sum(COST_PER_RUN[name] for _, name, _ in missing) if args.client == "anthropic" else 0
    print(f"{len(missing)} of {len(jobs)} runs to record; estimated cost ${cost:.2f}")
    if args.dry_run:
        return 0
    failed = record(jobs, args.client)
    build(jobs, args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
