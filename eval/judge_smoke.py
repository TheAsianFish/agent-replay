"""Does the (uncalibrated) judge agree with the deterministic checks on the retail suite?

Replays the six live ``claude-opus-5-5`` retail recordings
(``examples/retail_agent/replay-live.toml``) with the real agent and with each
seeded mutant, and compares, per run, the judge's verdict with the suite's
deterministic verdict (every non-judge check passed and the agent finished).
This is a smoke test of the judge wiring on 24 runs, not a calibration: the
reference is the suite's own rules and τ-bench ground truth, not human labels,
and one judge sample per run says nothing about its consistency.

The judge's verdicts are cached in ``examples/retail_agent/judge-cache/``
(committed), so a re-run is offline and free; ``--offline`` makes a cache miss
an error instead of a model call.

Usage::

    uv run python eval/judge_smoke.py --offline --out eval/results/judge_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import JsonValue

from agent_replay.ci import Overrides, ScenarioReport, run_suite
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay-live.toml"
AGENTS: dict[str, str | None] = {
    "agent": None,
    "wrong_write_argument": "examples.retail_agent.mutants:wrong_write_argument",
    "skips_order_lookup": "examples.retail_agent.mutants:skips_order_lookup",
    "first_tool_only": "examples.retail_agent.mutants:first_tool_only",
}


def _row(variant: str, scenario: ScenarioReport) -> dict[str, JsonValue]:
    judge = next(c for c in scenario.checks if c.check.startswith("judge:"))
    others = [c for c in scenario.checks if not c.check.startswith("judge:")]
    deterministic_pass = scenario.error is None and all(c.passed for c in others)
    criteria = judge.details.get("criteria")
    verdicts: dict[str, JsonValue] = {}
    if isinstance(criteria, dict):
        for criterion, value in criteria.items():
            verdicts[criterion] = value.get("verdict") if isinstance(value, dict) else None
    return {
        "variant": variant,
        "scenario": scenario.name,
        "deterministic_pass": deterministic_pass,
        "failed_checks": [c.check for c in others if not c.passed],
        "judge_pass": judge.passed,
        "judge_answered": bool(verdicts),
        "criteria": verdicts,
        "cached": judge.details.get("cached"),
        "original_cost_usd": judge.details.get("original_cost_usd"),
    }


def run() -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for variant, agent in AGENTS.items():
        retail_tools.reset_db()
        report = run_suite(SUITE, overrides=Overrides(agent=agent))
        rows += [_row(variant, s) for s in report.scenarios]
    retail_tools.reset_db()
    return rows


def _markdown(rows: list[dict[str, JsonValue]]) -> str:
    lines = [
        "| variant | scenario | deterministic | judge | resolved | policy_followed "
        "| accurate_report |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        criteria = r["criteria"] if isinstance(r["criteria"], dict) else {}
        lines.append(
            f"| {r['variant']} | {r['scenario']} | {'pass' if r['deterministic_pass'] else 'fail'} "
            f"| {'pass' if r['judge_pass'] else 'fail'} | {criteria.get('resolved', '—')} "
            f"| {criteria.get('policy_followed', '—')} | {criteria.get('accurate_report', '—')} |"
        )
    answered = [r for r in rows if r["judge_answered"]]
    agree = sum(1 for r in answered if r["judge_pass"] == r["deterministic_pass"])
    regressions = [r for r in answered if not r["deterministic_pass"]]
    flagged = sum(1 for r in regressions if not r["judge_pass"])
    clean = [r for r in answered if r["deterministic_pass"]]
    passed = sum(1 for r in clean if r["judge_pass"])
    lines += [
        "",
        f"- judge answered {len(answered)}/{len(rows)} runs",
        f"- agreement with the deterministic verdict: {agree}/{len(answered)}",
        f"- runs the deterministic checks failed that the judge also failed: "
        f"{flagged}/{len(regressions)}",
        f"- runs the deterministic checks passed that the judge also passed: {passed}/{len(clean)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--offline", action="store_true", help="judge from the cache only")
    parser.add_argument("--out", type=Path, help="write the per-run rows as JSON here")
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["REPLAY_JUDGE_OFFLINE"] = "1"
    rows = run()
    print(_markdown(rows))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0 if all(r["judge_answered"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
