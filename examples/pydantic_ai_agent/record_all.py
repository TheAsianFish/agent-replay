"""Re-record the travel desk's cassettes: the fake model's, or qwen2.5:7b's through Ollama.

``python -m examples.pydantic_ai_agent.record_all`` (``make pydantic-ai-cassettes``)
records each task with the deterministic fake model into ``cassettes/``.

``--live`` records on the local model instead, into ``cassettes/qwen/``. A
cassette is a recording someone has checked, so each attempt is replayed
through the suite's checks and the first that passes is kept, up to
``--attempts`` per task. Every attempt, kept or not, is logged with its failed
checks, tool calls, answer and wall time to ``--log`` (committed under
``eval/results/``), so the recordings that were thrown away are on file too.
Needs Ollama serving qwen2.5:7b (``ollama pull qwen2.5:7b``); runs only
locally or from the manual workflow, never on a pull request.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agent_replay.ci import Overrides, load_suite, run_scenario
from agent_replay.runner import record
from examples.pydantic_ai_agent import agent, fake_model
from examples.pydantic_ai_agent.environment import describe, ollama_model
from examples.pydantic_ai_agent.tasks import TASKS

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
AGENT_REF = "examples.pydantic_ai_agent.agent:run"
SUITES = {"fake": HERE / "replay.toml", "live": HERE / "replay-qwen.toml"}
LOG = ROOT / "eval" / "results" / "m9_qwen_recording.json"


def _attempt(name: str, live: bool, scratch: Path) -> tuple[dict[str, Any], Path]:
    """Record task ``name`` once into ``scratch`` and check it with the suite's checks."""
    task = TASKS[name]
    client = agent.make_client() if live else fake_model.make_client()
    metadata: dict[str, Any] = {"model": agent.MODEL, "provider": "ollama" if live else "fake"}
    if live:
        metadata.update(ollama_model() or {})
    started = time.perf_counter()
    result, cassette = record(
        agent.run, task.text, client=client, agent_ref=AGENT_REF, metadata=metadata
    )
    wall_s = round(time.perf_counter() - started, 1)
    path = scratch / f"{name}.jsonl"
    cassette.dump(path)
    suite_path = SUITES["live" if live else "fake"]
    suite = load_suite(suite_path)
    scenario = next(s for s in suite.scenario if s.name == name)
    report, _ = run_scenario(
        scenario,
        suite.defaults,
        Overrides(cassette_dir=scratch),
        suite_path.parent,
        time.perf_counter,
    )
    entry = {
        "task": name,
        "passed": report.passed,
        "failed_checks": [c.check for c in report.checks if not c.passed],
        "error": result.error.message if result.error else None,
        "tool_calls": [
            {"name": s.name, "arguments": s.request} for s in result.trajectory.tool_steps()
        ],
        "model_calls": sum(1 for e in cassette.envelopes if e.kind.value == "model"),
        "tokens": result.tokens,
        "output": result.output,
        "wall_s": wall_s,
    }
    return entry, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true", help="record on qwen2.5:7b via Ollama")
    parser.add_argument("--attempts", type=int, default=5, help="live attempts per task")
    parser.add_argument("--log", type=Path, default=LOG, help="where --live logs every attempt")
    parser.add_argument("--task", action="append", choices=sorted(TASKS), help="only these")
    args = parser.parse_args(argv)
    if args.live and ollama_model() is None:
        print(
            f"Ollama at {agent.ollama_url()} is not serving {agent.MODEL}; start it "
            f"(ollama serve) and pull the model (ollama pull {agent.MODEL}), or set "
            "REPLAY_OLLAMA_URL",
            file=sys.stderr,
        )
        return 2
    out = HERE / "cassettes" / ("qwen" if args.live else "")
    names = args.task or list(TASKS)
    attempts: list[dict[str, Any]] = []
    missing = 0
    for name in names:
        kept = False
        for k in range(1, (args.attempts if args.live else 1) + 1):
            with tempfile.TemporaryDirectory() as tmp:
                entry, path = _attempt(name, args.live, Path(tmp))
                entry["attempt"] = k
                attempts.append(entry)
                verdict = "passed" if entry["passed"] else f"failed {entry['failed_checks']}"
                print(f"{name} attempt {k}: {verdict} in {entry['wall_s']} s", flush=True)
                if entry["passed"]:
                    shutil.copyfile(path, out / f"{name}.jsonl")
                    kept = True
                    break
        missing += not kept
        if not kept:
            print(f"{name}: no attempt passed; its cassette was left as it was", file=sys.stderr)
    if args.live:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = {"environment": describe(), "max_attempts": args.attempts, "attempts": attempts}
        args.log.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"logged {len(attempts)} attempt(s) to {args.log}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
