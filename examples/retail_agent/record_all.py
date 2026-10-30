"""Re-record every retail task's cassette, resetting the database in between.

``python -m examples.retail_agent.record_all`` (``make retail-cassettes``) does
what ``replay record examples.retail_agent.agent:run --client
examples.retail_agent.fake_model:make_client -t <instruction> -o
examples/retail_agent/cassettes/<task id>.jsonl --force`` does for each task (the
hand-written M18 task goes to ``cassettes/error/``).
``--client anthropic --out DIR`` records against the real API instead and
reports, per task, whether the final database equals τ-bench's expected one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_replay.runner import make_client, record
from examples.retail_agent import agent
from examples.retail_agent.tasks import ALL_TASKS, HANDWRITTEN, expected_db_hash
from examples.retail_agent.tools import db_hash, reset_db

CASSETTES = Path(__file__).parent / "cassettes"
AGENT_REF = "examples.retail_agent.agent:run"


FAKE = "examples.retail_agent.fake_model:make_client"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client", default=FAKE, help="'anthropic' or module:factory")
    parser.add_argument("--out", type=Path, default=CASSETTES, help="cassette directory")
    args = parser.parse_args(argv)
    failed = 0
    for task in ALL_TASKS.values():
        expected = expected_db_hash(task)
        reset_db()
        result, cassette = record(
            agent.run, task.instruction, client=make_client(args.client), agent_ref=AGENT_REF
        )
        path = (args.out / "error" if task.id in HANDWRITTEN else args.out) / f"{task.id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        cassette.dump(path)
        reward = "db matches" if db_hash() == expected else "db differs"
        status = "ok" if result.ok else f"agent raised {result.error}"
        cost = sum(e.cost_usd or 0.0 for e in cassette.envelopes)
        print(f"{path}: {len(cassette.envelopes)} envelope(s), {status}, {reward}, ${cost:.4f}")
        failed += not result.ok
    reset_db()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
