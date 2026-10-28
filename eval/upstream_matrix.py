"""The pydantic-ai cassettes across upstream releases (PLAN.md, evolution round 2; M13 row).

The committed pydantic-ai cassettes were recorded on pydantic-ai-slim 2.61.0. For
each release in :data:`VERSIONS`, a throwaway virtual environment under
``.replay/upstream/`` gets this checkout (editable) and that release of
``pydantic-ai-slim[anthropic]``, then replays both pydantic-ai suites in ``full``
(:func:`eval.pydantic_ai.replay_full`'s check, in-process in that environment):
per suite, the scenarios that pass and the approximate steps. A release that
changed what the framework sends for this agent shows up as approximate steps
or failures; the same requests replay ``exact``.

Needs network access (``uv pip install`` from PyPI), so it is a manual target
(``make bench-upstream``), not part of ``make bench``. The output is byte-stable
for a given set of resolved packages.

Usage::

    uv run python -m eval.upstream_matrix --out eval/results/upstream_pydantic_ai.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("2.58.0", "2.59.0", "2.60.0", "2.61.0", "2.62.0")
RECORDED_ON = "2.61.0"
OUT = ROOT / "eval" / "results" / "upstream_pydantic_ai.json"
COMMAND = "make bench-upstream"

# Runs inside each release's environment; prints one JSON object.
_PROBE = """
import importlib.metadata as m, json
from agent_replay.ci import Overrides, run_suite
from eval.pydantic_ai import SUITES
suites = []
for path in SUITES:
    report = run_suite(path, overrides=Overrides(mode="full"))
    suites.append({
        "suite": path.name,
        "scenarios": len(report.scenarios),
        "passed": sum(1 for s in report.scenarios if s.passed),
        "approximate_steps": sum(s.steps.by_status.get("approximate", 0) for s in report.scenarios),
        "steps": sum(s.steps.total for s in report.scenarios),
    })
print(json.dumps({"installed": m.version("pydantic-ai-slim"), "suites": suites}))
"""


def probe(version: str, venvs: Path) -> dict[str, Any]:
    venv = venvs / f"pydantic-ai-{version}"
    python = venv / "bin" / "python"
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH; install it (https://docs.astral.sh/uv/)")
    subprocess.run(  # noqa: S603 - argv built here, never a shell
        [uv, "venv", "-q", "--clear", "--python", "3.12", str(venv)], check=True
    )
    subprocess.run(  # noqa: S603 - argv built here, never a shell
        [
            uv,
            "pip",
            "install",
            "-q",
            "--python",
            str(python),
            "-e",
            str(ROOT),
            f"pydantic-ai-slim[anthropic]=={version}",
        ],
        check=True,
    )
    done = subprocess.run(  # noqa: S603 - argv built here, never a shell
        [str(python), "-c", _PROBE],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": f"{venv / 'bin'}", "REPLAY_LOG_LEVEL": "error"},
    )
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    if result["installed"] != version:
        raise RuntimeError(f"asked for pydantic-ai-slim {version}, got {result['installed']}")
    return {"version": version, **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--venvs", type=Path, default=ROOT / ".replay" / "upstream")
    parser.add_argument("--versions", default=",".join(VERSIONS))
    args = parser.parse_args(argv)
    rows = []
    for version in args.versions.split(","):
        row = probe(version.strip(), args.venvs)
        rows.append(row)
        print(
            f"{row['version']}: "
            + "; ".join(
                f"{s['suite']} {s['passed']}/{s['scenarios']} pass, "
                f"{s['approximate_steps']} approximate of {s['steps']} steps"
                for s in row["suites"]
            )
        )
    report = {"command": COMMAND, "recorded_on": RECORDED_ON, "releases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ok = all(s["passed"] == s["scenarios"] for r in rows for s in r["suites"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
