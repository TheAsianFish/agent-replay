"""M9: the pydantic-ai travel desk, replayed offline and gated live on qwen2.5:7b.

1. ``replay ci`` in ``full`` over both suites (the fake-model recordings and
   the qwen2.5:7b ones): every scenario must pass with every step served
   ``exact``, 0 approximate. Offline; this part runs in CI.
2. With Ollama serving qwen2.5:7b, ``replay flake`` in ``model-live`` over the
   qwen suite: the model live, the tools served from its recordings, ``--runs``
   runs per scenario, one at a time (Ollama on CPU answers one request at a
   time, so concurrency only queues). The raw report goes to ``--out`` and a
   summary beside it (``*_summary.json``): passes per scenario, approximate
   steps per run, wall time, and the CPU, Ollama version and model digest.
   Without Ollama this part is skipped with a message, not failed.
3. M12, ``--arms strict,fallback``: instead of step 2, the two scenarios recorded
   on qwen2.5:7b (:data:`ARM_SCENARIOS`) in ``model-live`` once per unrecorded-call
   policy, ``--runs`` runs each, with the suite's ``identify`` keys on. ``strict``
   ends a run on a call the recording does not have; ``fallback`` runs the real
   tool instead. Every run is classified by how it ended (:func:`cause`), and one
   raw report (``--arms-out``) holds both arms' runs, causes and wall times.

Usage::

    uv run python -m eval.pydantic_ai [--runs 10] [--out eval/results/m9_flake_qwen.json]
    uv run python -m eval.pydantic_ai --arms strict,fallback --runs 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

from agent_replay.ci import Overrides, run_suite
from agent_replay.flake import FlakeReport, FlakeSettings, RunOutcome, flake_suite
from agent_replay.session import Policy
from examples.pydantic_ai_agent.agent import MODEL, ollama_url
from examples.pydantic_ai_agent.environment import describe, ollama_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_agent"
SUITES = (EXAMPLE / "replay.toml", EXAMPLE / "replay-qwen.toml")
QWEN_SUITE = SUITES[1]
CLIENT = "examples.pydantic_ai_agent.agent:make_client"
OUT = ROOT / "eval" / "results" / "m9_flake_qwen.json"
ARMS_OUT = ROOT / "eval" / "results" / "m12_arms_qwen.json"
ARM_SCENARIOS = ("cheapest_fare_in_euros", "book_earliest_flight")
"""The scenarios whose cassettes qwen2.5:7b recorded (the third serves the fake model's)."""
CAUSES = ("passed", "unrecorded", "approximate", "model")
CAUSE_MEANING = {
    "passed": "every check passed",
    "unrecorded": "a call no unused recording matched (by tool, or by the identify keys) "
    "ended the run under strict: the frozen world's limit",
    "approximate": "a step was served approximately (another request's recording) and a "
    "check then failed: possibly the frozen world's fault",
    "model": "a check failed (or the agent raised) with every served step exact: the "
    "model's own failure",
}


def replay_full() -> bool:
    """Both suites in ``full``: all pass, and no step is served approximately."""
    ok = True
    for suite in SUITES:
        report = run_suite(suite, overrides=Overrides(mode="full"))
        for s in report.scenarios:
            approximate = s.steps.by_status.get("approximate", 0)
            verdict = "PASS" if s.passed and not approximate else "FAIL"
            ok &= verdict == "PASS"
            print(
                f"{verdict} {suite.name} {s.name}: {s.steps.total} steps, {approximate} approximate"
            )
    return ok


def summarise(report: FlakeReport, wall_s: float, command: list[str]) -> dict[str, Any]:
    scenarios = []
    for s in report.scenarios:
        counted = [o for o in s.outcomes if o.counted]
        approximate = [o.steps.get("approximate", 0) for o in counted]
        scenarios.append(
            {
                "name": s.name,
                "classification": s.classification,
                "passes": s.passes,
                "runs": s.runs,
                "approximate_steps_per_run": approximate,
                "passes_with_an_approximate_step": sum(
                    1 for o, a in zip(counted, approximate, strict=True) if o.passed and a
                ),
                "failures": s.failures,
                "run_seconds": [round(o.duration_ms / 1000, 1) for o in counted],
            }
        )
    return {
        "command": " ".join(command),
        "wall_s": round(wall_s, 1),
        "environment": describe(),
        "stats_impl": report.stats_impl,
        "scenarios": scenarios,
    }


def flake(runs: int, out: Path) -> int:
    command = [
        "uv", "run", "replay", "flake", str(QWEN_SUITE.relative_to(ROOT)), "--no-store",
        "--mode", "model-live", "--client", CLIENT, "--max-runs", str(runs),
        "--concurrency", "1", "--json", str(out.relative_to(ROOT)),
    ]  # fmt: skip
    print("$", " ".join(command), flush=True)
    started = time.perf_counter()
    code = subprocess.call(command, cwd=ROOT, env=os.environ.copy())  # noqa: S603 - fixed argv
    wall_s = time.perf_counter() - started
    if not out.exists():
        print(f"replay flake wrote no report (exit {code})", file=sys.stderr)
        return 2
    report = FlakeReport.model_validate_json(out.read_text(encoding="utf-8"))
    summary = summarise(report, wall_s, command)
    path = out.with_name(out.stem + "_summary.json")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for s in summary["scenarios"]:
        print(
            f"{s['name']}: {s['passes']}/{s['runs']} passed ({s['classification']}), "
            f"approximate steps per run {s['approximate_steps_per_run']}"
        )
    print(f"wall time {summary['wall_s']} s on {summary['environment']['cpu']}; wrote {path}")
    return 0 if code in (0, 1) else code  # a failing gate is a result, not an error


def cause(outcome: RunOutcome) -> str:
    """How a run ended, one of :data:`CAUSES`, in that order of precedence: a pass is a
    pass even with an approximate step; an unrecorded call ends a ``strict`` run before
    any check runs; a failure after an approximate step may be the world's; the rest is
    the model's."""
    if outcome.passed:
        return "passed"
    if outcome.error is not None and outcome.error.type == "UnrecordedCallError":
        return "unrecorded"
    if outcome.steps.get("approximate", 0):
        return "approximate"
    return "model"


def _run_row(outcome: RunOutcome) -> dict[str, Any]:
    error = outcome.error
    return {
        "index": outcome.index,
        "seed": outcome.seed,
        "passed": outcome.passed,
        "cause": cause(outcome),
        "identity": error is not None and "identify keys" in error.message,
        "error": None if error is None else f"{error.type}: {error.message}",
        "failed_checks": list(outcome.failed_checks),
        "steps": dict(sorted(outcome.steps.items())),
        "seconds": round(outcome.duration_ms / 1000, 1),
    }


def arm_summary(report: FlakeReport, wall_s: float) -> dict[str, Any]:
    """One arm: per scenario its pass and cause counts, the gate's verdict and every run."""
    scenarios = []
    for s in report.scenarios:
        runs = [_run_row(o) for o in s.outcomes if o.counted]
        causes = {c: sum(1 for r in runs if r["cause"] == c) for c in CAUSES}
        scenarios.append(
            {
                "name": s.name,
                "runs": s.runs,
                "passes": s.passes,
                "decision": s.decision,
                "classification": s.classification,
                "causes": causes,
                "unrecorded_by_identity": sum(1 for r in runs if r["identity"]),
                "passes_with_an_approximate_step": sum(
                    1 for r in runs if r["passed"] and r["steps"].get("approximate", 0)
                ),
                "runs_with_a_fallback_step": sum(1 for r in runs if r["steps"].get("fallback", 0)),
                "failures": s.failures,
                "run_seconds_total": round(sum(r["seconds"] for r in runs), 1),
                "outcomes": runs,
            }
        )
    return {
        "policy": report.scenarios[0].policy if report.scenarios else None,
        "wall_s": round(wall_s, 1),
        "stats_impl": report.stats_impl,
        "scenarios": scenarios,
    }


def run_arms(
    arms: Sequence[Policy],
    runs: int,
    out: Path,
    *,
    client: str = CLIENT,
    suite: Path = QWEN_SUITE,
    scenarios: Sequence[str] = ARM_SCENARIOS,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Each arm in ``model-live`` under its policy, ``runs`` runs per scenario, one at a
    time; writes the raw report to ``out`` and returns it."""
    settings = FlakeSettings(max_runs=runs, concurrency=1)
    result: dict[str, Any] = {
        "command": f"uv run python -m eval.pydantic_ai --arms {','.join(arms)} --runs {runs}",
        "suite": str(suite.relative_to(ROOT)) if suite.is_relative_to(ROOT) else str(suite),
        "client": client,
        "mode": "model-live",
        "identify": "the suite's [defaults.identify] keys",
        "settings": settings.model_dump(mode="json"),
        "environment": environment if environment is not None else describe(),
        "causes": CAUSE_MEANING,
        "arms": {},
    }
    for arm in arms:
        print(f"arm {arm}: {runs} model-live runs of {', '.join(scenarios)}", flush=True)
        started = time.perf_counter()
        report = flake_suite(
            suite,
            settings=settings,
            overrides=Overrides(mode="model-live", policy=arm, client=client),
            scenarios=list(scenarios),
        )
        summary = arm_summary(report, time.perf_counter() - started)
        result["arms"][arm] = summary
        for s in summary["scenarios"]:
            counts = ", ".join(f"{n} {c}" for c, n in s["causes"].items() if n)
            print(f"  {s['name']}: {s['passes']}/{s['runs']} passed ({counts})", flush=True)
        print(f"  wall time {summary['wall_s']} s", flush=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return result


def parse_arms(text: str) -> tuple[Policy, ...]:
    arms = tuple(a.strip() for a in text.split(",") if a.strip())
    known = get_args(Policy.__value__)
    unknown = [a for a in arms if a not in known]
    if unknown or not arms:
        raise argparse.ArgumentTypeError(
            f"--arms takes a comma-separated list of policies ({', '.join(known)}); got {text!r}"
        )
    return arms  # type: ignore[return-value]  # checked against Policy above


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=10, help="model-live runs per scenario")
    parser.add_argument("--out", type=Path, default=OUT, help="raw flake report")
    parser.add_argument("--offline", action="store_true", help="only the full-mode replays")
    parser.add_argument(
        "--arms",
        type=parse_arms,
        help="M12: run the qwen-recorded scenarios once per policy (e.g. strict,fallback) "
        "instead of the flake",
    )
    parser.add_argument("--arms-out", type=Path, default=ARMS_OUT, help="raw report of --arms")
    args = parser.parse_args(argv)
    if not replay_full():
        return 1
    if args.offline:
        return 0
    if ollama_model() is None:
        print(
            f"skipped the model-live runs: Ollama at {ollama_url()} is not serving {MODEL} "
            f"(ollama serve; ollama pull {MODEL}; or set REPLAY_OLLAMA_URL)"
        )
        return 0
    if args.arms:
        run_arms(args.arms, args.runs, args.arms_out.resolve())
        return 0
    return flake(args.runs, args.out.resolve())


if __name__ == "__main__":
    sys.exit(main())
