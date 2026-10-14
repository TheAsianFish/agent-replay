"""Fill a run store with the dashboard's demo data, using the repo's own commands.

Every run in it is a real ``replay ci`` or ``replay flake`` invocation on the
retail example, run in-process exactly as on the command line:

- the retail suite in each replay mode (``full``, ``tools-live``, and
  ``model-live`` against the deterministic fake model): passing runs;
- the six live ``claude-opus-5-5`` recordings (``replay-live.toml``), judged
  from the committed verdict cache: real costs and latencies;
- each seeded regression (``examples/retail_agent/mutants.py``): failing runs to
  diff against the passing ones;
- ``replay flake`` once with the fake model, then with the seeded flaky model
  at several seeds: a pass-rate trend per scenario.

The calibration page gets two ``replay calibrate`` reports on the retail golden set
(``eval/golden/retail``, built by ``make golden``):

- the real judge (``claude-opus-5-5``) from the committed verdict cache, offline;
- the deterministic fake judge with three repeats and every bias probe, so each
  section of the page has data.

Both compare the judge with the suite's deterministic verdicts (``--reference
objective``), not human labels: the golden set has none yet. Such reports never
gate, and the page says so. Their statistics come from ``REPLAY_STATS_IMPL``.

Usage::

    uv run python -m examples.demo_data --db sqlite:///.replay/demo.db
    uv run python -m examples.demo_data --if-empty   # $REPLAY_DB_URL; skip a filled store
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import typer

from agent_replay.cli import app
from agent_replay.store import Store

ROOT = Path(__file__).resolve().parents[1]
RETAIL = ROOT / "examples" / "retail_agent"


def _shown(path: Path) -> str:
    """``path`` relative to the working directory when it is under it, so stored suites read
    ``examples/retail_agent/replay.toml`` wherever the repo is checked out."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


SUITE = _shown(RETAIL / "replay.toml")
LIVE_SUITE = _shown(RETAIL / "replay-live.toml")
FAKE = "examples.retail_agent.fake_model:make_client"
FLAKY = "examples.retail_agent.flaky_model:make_client"
MUTANTS = ("wrong_write_argument", "skips_order_lookup", "first_tool_only")
FLAKE_SEEDS = (0, 1, 2, 3, 4)
GOLDEN = _shown(ROOT / "eval" / "golden" / "retail")
RUBRIC = _shown(RETAIL / "rubric.yaml")
JUDGE_CACHE = _shown(RETAIL / "judge-cache")
FAKE_JUDGE = "examples.retail_agent.fake_judge:make_judge"
# The fake judge's verdicts are free and deterministic, so a scratch cache shared between
# seeds is harmless; the committed cache is never written to.
FAKE_JUDGE_CACHE = str(Path(tempfile.gettempdir()) / "replay-demo-fake-judge-cache")

Invocation = tuple[Sequence[str], frozenset[int]]
"""Arguments to ``replay`` and the exit codes that mean it did what the demo wants."""

PASS, FAIL, EITHER = frozenset({0}), frozenset({1}), frozenset({0, 1})


def invocations(db: str) -> list[Invocation]:
    store = ["--db", db]
    plan: list[Invocation] = [
        (["ci", SUITE, *store], PASS),
        (["ci", SUITE, *store, "--mode", "tools-live"], PASS),
        (["ci", SUITE, *store, "--mode", "model-live", "--client", FAKE], PASS),
        (["ci", LIVE_SUITE, *store], PASS),
    ]
    for mutant in MUTANTS:
        agent = f"examples.retail_agent.mutants:{mutant}"
        plan.append((["ci", SUITE, *store, "--agent", agent], FAIL))
    skips = "examples.retail_agent.mutants:skips_order_lookup"
    plan.append((["ci", LIVE_SUITE, *store, "--agent", skips], FAIL))
    flake = ["flake", SUITE, *store, "--max-runs", "12"]
    plan.append(([*flake, "--client", FAKE], EITHER))
    for seed in FLAKE_SEEDS:
        plan.append(([*flake, "--client", FLAKY, "--seed", str(seed)], EITHER))
    calibrate = ["calibrate", GOLDEN, "--rubric", RUBRIC, "--reference", "objective", *store]
    plan.append(
        (
            [*calibrate, "--repeats", "1", "--no-probes", "--cache-dir", JUDGE_CACHE, "--offline"],
            PASS,
        )
    )
    plan.append(
        (
            [
                *calibrate,
                "--provider",
                FAKE_JUDGE,
                "--repeats",
                "3",
                "--cache-dir",
                FAKE_JUDGE_CACHE,
                "--online",
            ],
            PASS,
        )
    )
    return plan


@contextlib.contextmanager
def _env(**values: str) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def run_replay(args: Sequence[str]) -> int:
    """Run one ``replay`` command in this process and return its exit code."""
    try:
        code = app(list(args), standalone_mode=False)
    except typer.exceptions.TyperException as exc:
        # Usage errors from the command-line parser (typer's bundled click) carry
        # their exit code; `typer.Exit` is returned by the app itself.
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        exit_code = getattr(exc, "exit_code", 1)
        return exit_code if isinstance(exit_code, int) else 1
    return code if isinstance(code, int) else 0


def seed(db: str, *, if_empty: bool = False) -> int:
    with Store.open(db) as store:
        if if_empty and store.count_runs() > 0:
            print(f"the store already holds {store.count_runs()} runs; left as it is")
            return 0
    with _env(REPLAY_JUDGE_OFFLINE="1", COLUMNS="160"):
        for args, expected in invocations(db):
            shown = " ".join(a if a != db else "$DB" for a in args)
            print(f"$ replay {shown}", flush=True)
            code = run_replay(args)
            if code not in expected:
                print(f"replay exited {code}, expected {sorted(expected)}", file=sys.stderr)
                return 1
    with Store.open(db) as store:
        print(
            f"demo store ready: {store.count_runs()} runs, {len(store.list_suites(limit=500))} "
            f"suites, {len(store.list_flake_reports(limit=500))} flake reports, "
            f"{len(store.list_calibrations())} calibration reports"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        default=os.environ.get("REPLAY_DB_URL", "sqlite:///.replay/demo.db"),
        help="run store URL (default: $REPLAY_DB_URL, then sqlite:///.replay/demo.db)",
    )
    parser.add_argument(
        "--if-empty", action="store_true", help="do nothing when the store already has runs"
    )
    args = parser.parse_args(argv)
    return seed(args.db, if_empty=args.if_empty)


if __name__ == "__main__":
    raise SystemExit(main())
