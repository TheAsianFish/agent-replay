"""Re-record every retail task's cassette with the deterministic fake model.

``python -m examples.retail_agent.record_all`` (``make retail-cassettes``) does
what ``replay record examples.retail_agent.agent:run --client
examples.retail_agent.fake_model:make_client -t <instruction> -o
examples/retail_agent/cassettes/<task id>.jsonl --force`` does for each task,
resetting the database in between.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_replay.runner import record
from examples.retail_agent import agent
from examples.retail_agent.fake_model import make_client
from examples.retail_agent.tasks import TASKS
from examples.retail_agent.tools import reset_db

CASSETTES = Path(__file__).parent / "cassettes"
AGENT_REF = "examples.retail_agent.agent:run"


def main() -> int:
    failed = 0
    for task in TASKS.values():
        reset_db()
        result, cassette = record(
            agent.run, task.instruction, client=make_client(), agent_ref=AGENT_REF
        )
        path = CASSETTES / f"{task.id}.jsonl"
        cassette.dump(path)
        status = "ok" if result.ok else f"agent raised {result.error}"
        print(f"{path}: {len(cassette.envelopes)} envelope(s), {status}")
        failed += not result.ok
    reset_db()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
