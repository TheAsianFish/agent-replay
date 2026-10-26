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

Usage::

    uv run python -m eval.pydantic_ai [--runs 10] [--out eval/results/m9_flake_qwen.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agent_replay.ci import Overrides, run_suite
from agent_replay.flake import FlakeReport
from examples.pydantic_ai_agent.agent import MODEL, ollama_url
from examples.pydantic_ai_agent.environment import describe, ollama_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_agent"
SUITES = (EXAMPLE / "replay.toml", EXAMPLE / "replay-qwen.toml")
QWEN_SUITE = SUITES[1]
CLIENT = "examples.pydantic_ai_agent.agent:make_client"
OUT = ROOT / "eval" / "results" / "m9_flake_qwen.json"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=10, help="model-live runs per scenario")
    parser.add_argument("--out", type=Path, default=OUT, help="raw flake report")
    parser.add_argument("--offline", action="store_true", help="only the full-mode replays")
    args = parser.parse_args(argv)
    if not replay_full():
        return 1
    if args.offline:
        return 0
    if ollama_model() is None:
        print(
            f"skipped the model-live flake: Ollama at {ollama_url()} is not serving {MODEL} "
            f"(ollama serve; ollama pull {MODEL}; or set REPLAY_OLLAMA_URL)"
        )
        return 0
    return flake(args.runs, args.out.resolve())


if __name__ == "__main__":
    sys.exit(main())
