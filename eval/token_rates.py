"""Where ``replay calibrate``'s cost-estimate constants come from.

The estimate in :mod:`agent_replay.calibrate.calibrate` prices a judge call from
two measured numbers: characters per input token (:data:`CHARS_PER_TOKEN`) and
output tokens per call (:data:`OUTPUT_TOKENS_PER_CALL`). This script re-derives
both from the 21 retail verdicts committed in
``examples/retail_agent/judge-cache/`` (the ones ``eval/golden/retail`` holds
transcripts for) and checks the estimator against what those verdicts really
cost. Offline; exits 1 if a constant no longer matches its measurement.

Usage::

    uv run python eval/token_rates.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from agent_replay.calibrate.calibrate import CHARS_PER_TOKEN, OUTPUT_TOKENS_PER_CALL
from agent_replay.calibrate.golden import GoldenSet
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.verdict import system_prompt
from agent_replay.pricing import prices

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden" / "retail"
CACHE = ROOT / "examples" / "retail_agent" / "judge-cache"
RUBRIC = ROOT / "examples" / "retail_agent" / "rubric.yaml"
MODEL = "claude-opus-5-5"


def main() -> int:
    cache: dict[str, dict[str, object]] = {}
    for path in sorted(CACHE.glob("*/*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        cache[str(entry["input_fingerprint"])] = entry
    price = prices()[MODEL]
    rubric = Rubric.load(RUBRIC)
    system_tokens = len(system_prompt(rubric)) / CHARS_PER_TOKEN

    ratios: list[float] = []
    outputs: list[int] = []
    real = estimated = 0.0
    first = True
    for item in GoldenSet.load(GOLDEN).items:
        entry = cache.get(item.transcript.fingerprint)
        if entry is None:
            continue
        usage = entry["usage"]
        assert isinstance(usage, dict)
        chars = len(item.transcript.render())
        ratios.append(chars / int(usage["input_tokens"]))
        outputs.append(int(usage["output_tokens"]))
        real += float(str(entry["cost_usd"]))
        estimated += system_tokens * (price.cache_write if first else price.cache_read)
        estimated += chars / CHARS_PER_TOKEN * price.input
        estimated += OUTPUT_TOKENS_PER_CALL * price.output
        first = False
    estimated /= 1_000_000

    median_ratio = statistics.median(ratios)
    median_output = int(statistics.median(outputs))
    print(f"{len(ratios)} committed verdicts matched to golden transcripts")
    print(
        f"characters per input token: min {min(ratios):.2f} median {median_ratio:.2f} "
        f"max {max(ratios):.2f} (CHARS_PER_TOKEN = {CHARS_PER_TOKEN})"
    )
    print(
        f"output tokens per call: min {min(outputs)} median {median_output} max {max(outputs)} "
        f"(OUTPUT_TOKENS_PER_CALL = {OUTPUT_TOKENS_PER_CALL})"
    )
    print(f"real spend ${real:.4f}; the estimator says ${estimated:.4f}")
    ok = round(median_ratio, 1) == CHARS_PER_TOKEN and median_output == OUTPUT_TOKENS_PER_CALL
    if not ok:
        print(
            "the constants in agent_replay/calibrate/calibrate.py no longer match", file=sys.stderr
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
