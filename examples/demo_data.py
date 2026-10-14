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

The calibration page gets one report computed from ``eval/results/judge_smoke.json``:
the judge's per-criterion verdicts against the suite's deterministic verdict on 24
runs. That reference is the suite's own rules, not human labels, and the report's
``method`` says so; real calibration against human labels arrives with ``replay
calibrate``. Its statistics come from ``REPLAY_STATS_IMPL``.

Usage::

    uv run python -m examples.demo_data --db sqlite:///.replay/demo.db
    uv run python -m examples.demo_data --if-empty   # $REPLAY_DB_URL; skip a filled store
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import typer

from agent_replay import stats_impl
from agent_replay.cli import app
from agent_replay.judge.calibration import CalibrationReport, CriterionCalibration
from agent_replay.judge.providers import DEFAULT_ANTHROPIC_MODEL, DEFAULT_EFFORT
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.verdict import PROMPT_VERSION
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
JUDGE_SMOKE = ROOT / "eval" / "results" / "judge_smoke.json"

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


def calibration_from_judge_smoke(path: Path = JUDGE_SMOKE) -> CalibrationReport:
    """Per criterion, the judge's verdict against the deterministic verdict on the same run."""
    rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if r["judge_answered"]]
    rubric = Rubric.load(RETAIL / "rubric.yaml")
    impl = stats_impl.load()
    criteria: dict[str, CriterionCalibration] = {}
    for criterion in rubric.ids:
        judged = [r for r in rows if r["criteria"].get(criterion) in ("pass", "fail")]
        judge = [int(r["criteria"][criterion] == "pass") for r in judged]
        reference = [int(r["deterministic_pass"]) for r in judged]
        kappa = impl.cohens_kappa(judge, reference)
        low, high = impl.bootstrap_ci(impl.cohens_kappa, judge, reference)
        agreement = sum(j == d for j, d in zip(judge, reference, strict=True)) / len(judged)
        criteria[criterion] = CriterionCalibration(
            kappa=_clamp(kappa),
            ci_low=_clamp(low),
            ci_high=_clamp(high),
            agreement=agreement,
            n=len(judged),
        )
    stats_name = getattr(impl, "__name__", "?").rsplit(".", 1)[-1]
    measure = (
        "Cohen's kappa with a 95% bootstrap CI (2000 resamples)"
        if stats_name == "stats"
        else "raw agreement in the kappa column, with a normal-approximation bootstrap CI "
        "(REPLAY_STATS_IMPL=baseline; set it to core for Cohen's kappa)"
    )
    return CalibrationReport(
        rubric=rubric.name,
        rubric_fingerprint=rubric.fingerprint,
        judge_id=f"anthropic:{DEFAULT_ANTHROPIC_MODEL}:effort={DEFAULT_EFFORT}",
        prompt_version=PROMPT_VERSION,
        created_at=_newest_verdict(),
        n_items=len(rows),
        method=(
            f"Demo, not a calibration: the judge against the suite's deterministic verdict "
            f"(not human labels) on {len(rows)} retail runs from eval/judge_smoke.py; {measure}."
        ),
        criteria=criteria,
    )


def _newest_verdict() -> datetime:
    """When the newest committed judge verdict was made: the report's data is that old."""
    stamps = [
        datetime.fromisoformat(json.loads(p.read_text(encoding="utf-8"))["created_at"])
        for p in (RETAIL / "judge-cache").glob("*/*.json")
    ]
    return max(stamps).astimezone(UTC).replace(microsecond=0)


def _clamp(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(-1.0, min(1.0, value))


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
        store.save_calibration(calibration_from_judge_smoke())
        print(
            f"demo store ready: {store.count_runs()} runs, {len(store.list_suites(limit=500))} "
            f"suites, {len(store.list_flake_reports(limit=500))} flake reports, "
            f"{len(store.list_calibrations())} calibration report"
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
