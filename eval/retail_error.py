"""M18: does a lost ``is_error`` flag matter to a real model that reads the error text?

The retail task ``account_not_found_wrong_zip`` (hand-written, not from τ-bench: a
wrong zip code, so ``find_user_id_by_name_zip`` fails and the right reply is that the
account was not found, with nothing written) on qwen2.5:7b through Ollama:

1. ``--record``: record the task on the local model, up to ``--attempts`` (3) times;
   the first attempt that passes the suite's checks is kept as
   ``examples/retail_agent/cassettes/qwen/<task>.jsonl``. Every attempt is logged to
   ``eval/results/m18_qwen_recording.json`` with each model call's
   ``usage.input_tokens``, so a prompt Ollama cut to its context window is seen
   rather than trusted. Ollama counts only the tokens its prompt cache did not
   already hold, so each run also gets an upper bound on its largest prompt
   (:func:`prompt_bound`).
2. ``--probe``: what Ollama's ``/v1/messages`` does with the flag. The same tool
   result is sent with and without ``is_error: true`` (and once with a word added to
   its text, as a control that the count moves), each request opened by a different
   nonce so the prompt cache reuses the same prefix for all three; equal
   ``input_tokens`` with and without the flag means the flag never reached the model.
3. The arms (the default): ``model-live`` under ``fallback`` on the qwen recording (the
   fake-model one if no qwen attempt passed, and the report says so), ``--runs`` (10)
   runs with the unmutated agent and as many with ``drops_error_flag``, both on the
   local model. Every run records its tool calls, its answer, whether it replied "not
   found" with no write (:func:`examples.retail_agent.rules.reports_account_not_found`),
   how many ``is_error`` flags its model requests carried, and each model call's
   ``input_tokens`` and ``output_tokens``, with the bound on its largest prompt. The
   report answers the milestone's hypothesis (:func:`hypothesis`).

Without Ollama serving the model every live part exits 2 with the fix named. The
harness is tested offline against the fake model (``tests/test_retail_error.py``).

Usage::

    uv run python -m eval.retail_error --record      # make record-retail-error-qwen
    uv run python -m eval.retail_error               # make eval-retail-error-qwen
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import ToolUseBlockParam

from agent_replay.ci import Overrides, ScenarioReport, load_suite, run_scenario
from agent_replay.flake import FlakeSettings, flake_suite
from agent_replay.runner import RunResult, load_agent, make_client, record
from eval.pydantic_ai import Capture, arm_summary
from examples.pydantic_ai_agent.environment import describe, ollama_model
from examples.retail_agent import ollama
from examples.retail_agent.rules import reports_account_not_found
from examples.retail_agent.tasks import ACCOUNT_NOT_FOUND, HANDWRITTEN
from examples.retail_agent.tools import reset_db

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "retail_agent"
SUITE = EXAMPLE / "replay-error.toml"
FAKE_CASSETTES = EXAMPLE / "cassettes" / "error"
QWEN_CASSETTES = EXAMPLE / "cassettes" / "qwen"
OUT = ROOT / "eval" / "results" / "m18_error_qwen.json"
RECORDING_LOG = ROOT / "eval" / "results" / "m18_qwen_recording.json"
CLIENT = "examples.retail_agent.ollama:make_client"
TASK = HANDWRITTEN[ACCOUNT_NOT_FOUND]
CONTEXT_WINDOW = 4096
"""The context Ollama 0.40 gives qwen2.5:7b by default (``llama-server -c 4096``, with
context shift on); a prompt that reaches it may have been cut."""


@dataclass(frozen=True)
class Arm:
    label: str
    agent: str


ARMS = (
    Arm("unmutated", "examples.retail_agent.ollama:run"),
    Arm("drops_error_flag", "eval.mutations.agents:drops_error_flag_ollama"),
)

# --- what a run sent and said ---------------------------------------------------------


def _usage(response: Any, key: str = "input_tokens") -> int | None:
    usage = response.get("usage") if isinstance(response, dict) else None
    tokens = usage.get(key) if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) else None


def prompt_bound(
    input_tokens: Sequence[int | None], output_tokens: Sequence[int | None], prefix: int | None
) -> int | None:
    """An upper bound on the run's largest prompt, in tokens.

    Ollama reports ``input_tokens`` net of its prompt cache: a call after the first
    counts only what the cache did not hold (the new tool results), and a run's first
    call counts only where it leaves the previous run's conversation. Every call's
    prompt is the previous call's prompt and reply plus what it reports, or less when
    the cache held more, so the last call's prompt is at most the first call's full
    prompt (``prefix``: the system prompt, tools and task, counted on a cold cache;
    else the run's own first count), plus every later call's input, plus every earlier
    call's output."""
    if not input_tokens or any(t is None for t in [*input_tokens, *output_tokens[:-1]]):
        return None
    first = input_tokens[0]
    assert first is not None
    start = max(first, prefix or 0)
    later = sum(t for t in input_tokens[1:] if t is not None)
    replies = sum(t for t in output_tokens[:-1] if t is not None)
    return start + later + replies


def error_flags_sent(result: RunResult) -> int:
    """How many tool results the run's last model request marked ``is_error: true`` (the
    last request carries the whole conversation)."""
    requests = [s.request for s in result.trajectory.model_steps() if isinstance(s.request, dict)]
    if not requests:
        return 0
    messages = requests[-1].get("messages")
    flagged = 0
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            flagged += sum(
                1
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("is_error") is True
            )
    return flagged


def run_facts(result: RunResult | None, prefix: int | None = None) -> dict[str, Any]:
    """One run's tool calls (with each result, or the error a failed call raised), its
    answer, the not-found verdict,
    the ``is_error`` flags its requests carried, each model call's ``input_tokens`` and
    ``output_tokens``, and the bound on its largest prompt (:func:`prompt_bound`)."""
    if result is None:
        return {}
    verdict, reason = reports_account_not_found(result)
    steps = result.trajectory.model_steps()
    inputs = [_usage(s.response) for s in steps]
    outputs = [_usage(s.response, "output_tokens") for s in steps]
    return {
        "tool_calls": [
            {
                "name": s.name,
                "input": s.request,
                "status": s.status.value,
                "response": s.response,
                "error": None if s.error is None else f"{s.error.type}: {s.error.message}",
            }
            for s in result.trajectory.tool_steps()
        ],
        "output": result.output,
        "not_found_no_write": verdict,
        "not_found_reason": reason,
        "error_flags_sent": error_flags_sent(result),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "prompt_tokens_bound": prompt_bound(inputs, outputs, prefix),
    }


@dataclass
class RetailCapture(Capture):
    """:class:`Capture` with the retail run's facts (:func:`run_facts`) in place of the
    travel desk's wire facts. ``prefix``: the cold count of the first prompt."""

    prefix: int | None = None

    def save(self, report: ScenarioReport, result: RunResult | None, started: datetime) -> str:
        run_id = f"{report.name}/{len(self.runs)}"
        self.runs[run_id] = run_facts(result, self.prefix)
        if result is not None:
            self.results[run_id] = result
        return run_id


# --- 1. recording ---------------------------------------------------------------------


def _check(cassette_dir: Path) -> ScenarioReport:
    suite = load_suite(SUITE)
    report, _ = run_scenario(
        suite.scenario[0],
        suite.defaults,
        Overrides(cassette_dir=cassette_dir),
        SUITE.parent,
        time.perf_counter,
    )
    return report


def ollama_metadata() -> dict[str, Any]:
    """The cassette header's metadata for a recording on Ollama: model, server, digest."""
    metadata: dict[str, Any] = {"model": ollama.ollama_model(), "provider": "ollama"}
    metadata.update(ollama_model(ollama.ollama_model()) or {})
    return metadata


def record_attempt(
    agent: str, client: str, scratch: Path, metadata: dict[str, Any], prefix: int | None = None
) -> tuple[dict[str, Any], Path]:
    """Record the task once into ``scratch`` and check it in ``full`` with the suite.
    ``prefix``: the first attempt's cold count of the first prompt (:func:`prompt_bound`)."""
    reset_db()
    started = time.perf_counter()
    result, cassette = record(
        load_agent(agent),
        TASK.instruction,
        client=make_client(client),
        agent_ref=agent,
        metadata=metadata,
    )
    wall_s = round(time.perf_counter() - started, 1)
    path = scratch / f"{TASK.id}.jsonl"
    cassette.dump(path)
    report = _check(scratch)
    reset_db()
    entry = {
        "passed": report.passed,
        "failed_checks": [c.check for c in report.checks if not c.passed],
        "error": result.error.message if result.error else None,
        **run_facts(result, prefix),
        "model_calls": len(result.trajectory.model_steps()),
        "wall_s": wall_s,
    }
    return entry, path


def record_task(
    attempts: int,
    log: Path = RECORDING_LOG,
    out: Path = QWEN_CASSETTES,
    *,
    agent: str = ARMS[0].agent,
    client: str = CLIENT,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Up to ``attempts`` recordings; the first that passes is copied into ``out``.
    ``metadata`` goes into each cassette's header (default :func:`ollama_metadata`)."""
    header = metadata if metadata is not None else ollama_metadata()
    entries: list[dict[str, Any]] = []
    kept = False
    for k in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            # Later attempts find the prompt in Ollama's cache; the first one's count is cold.
            prefix = (
                entries[0]["input_tokens"][0] if entries and entries[0]["input_tokens"] else None
            )
            entry, path = record_attempt(agent, client, Path(tmp), header, prefix)
            entry["attempt"] = k
            entries.append(entry)
            verdict = "passed" if entry["passed"] else f"failed {entry['failed_checks']}"
            print(
                f"attempt {k}: {verdict} in {entry['wall_s']} s, input tokens "
                f"{entry['input_tokens']}, largest prompt at most {entry['prompt_tokens_bound']}",
                flush=True,
            )
            if entry["passed"]:
                out.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out / path.name)
                kept = True
                break
    result = {
        "command": "uv run python -m eval.retail_error --record",
        "task": TASK.id,
        "instruction": TASK.instruction,
        "environment": environment if environment is not None else describe(ollama.ollama_model()),
        "context_window": CONTEXT_WINDOW,
        "max_attempts": attempts,
        "kept": _relative(out / f"{TASK.id}.jsonl") if kept else None,
        "attempts": entries,
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"logged {len(entries)} attempt(s) to {log}")
    return result


# --- 2. the flag on the wire ----------------------------------------------------------

_PROBE_TOOL = {
    "name": "find_user_id_by_name_zip",
    "description": "Find user id by first name, last name, and zip code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["first_name", "last_name", "zip"],
    },
}
PROBES = (
    ("W", "Error: user not found", False),
    ("A", "Error: user not found", True),
    ("B", "Error: user not found", False),
    ("C", "Error: user not found (lookup failed)", False),
)
"""(nonce, tool result text, ``is_error``): W warms the cache with the shared prefix and is
not compared; A and B differ only in the flag; C is the control, B's text with three
more words."""


def probe_flag(client: Anthropic, model: str) -> dict[str, Any]:
    """Send :data:`PROBES` and report each one's ``usage.input_tokens``. The nonce opens
    the user message, so after the warm-up every probe finds the same prefix (the chat
    template and the tool list) in Ollama's prompt cache and counts only the tokens from
    its nonce on. Without the warm-up the first probe would also pay for that prefix."""
    rows = []
    for nonce, text, flag in PROBES:
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": "call_1", "content": text}
        if flag:
            result["is_error"] = True
        tool_use: ToolUseBlockParam = {
            "type": "tool_use",
            "id": "call_1",
            "name": "find_user_id_by_name_zip",
            "input": {"first_name": "Noah", "last_name": "Ito", "zip": "98178"},
        }
        response = client.messages.create(
            model=model,
            max_tokens=1,
            tools=[_PROBE_TOOL],  # type: ignore[list-item]
            messages=[
                {"role": "user", "content": f"{nonce}. I'm Noah Ito, zip code 98178."},
                {"role": "assistant", "content": [tool_use]},
                {"role": "user", "content": [result]},  # type: ignore[list-item]
            ],
        )
        rows.append(
            {
                "nonce": nonce,
                "content": text,
                "is_error": flag,
                "input_tokens": response.usage.input_tokens,
            }
        )
    _, flagged, unflagged, control = (r["input_tokens"] for r in rows)
    return {
        "probes": rows,
        "flag_changes_prompt": flagged != unflagged,
        "control_changes_prompt": control != unflagged,
        "reading": "the flag never reaches the model"
        if flagged == unflagged and control != unflagged
        else "the flag reaches the model"
        if flagged != unflagged
        else "inconclusive: the control did not move the count",
    }


# --- 3. the arms ----------------------------------------------------------------------


def run_arm(
    arm: Arm,
    runs: int,
    *,
    client: str,
    cassette_dir: Path,
    suite: Path = SUITE,
    prefix: int | None = None,
) -> tuple[dict[str, Any], RetailCapture]:
    """``runs`` model-live runs of the task under ``fallback`` with ``arm``'s agent."""
    capture = RetailCapture(prefix=prefix)
    settings = FlakeSettings(gate="wilson", max_runs=runs, concurrency=1)
    print(f"arm {arm.label}: {runs} model-live runs under fallback", flush=True)
    started = time.perf_counter()
    report = flake_suite(
        suite,
        settings=settings,
        overrides=Overrides(
            agent=arm.agent,
            mode="model-live",
            policy="fallback",
            client=client,
            cassette_dir=cassette_dir,
        ),
        sink=capture.sink(),
    )
    summary = arm_summary(report, time.perf_counter() - started, capture)
    summary["label"] = arm.label
    summary["agent"] = arm.agent
    for s in summary["scenarios"]:
        rows = s["outcomes"]
        s["not_found_no_write"] = sum(1 for r in rows if r.get("not_found_no_write"))
        s["runs_with_an_error_flag_sent"] = sum(1 for r in rows if r.get("error_flags_sent"))
        bounds = [r.get("prompt_tokens_bound") for r in rows]
        known = [b for b in bounds if isinstance(b, int)]
        s["max_prompt_tokens_bound"] = max(known, default=None)
        s["runs_without_a_bound"] = len(bounds) - len(known)
        s["runs_that_may_reach_the_context_window"] = sum(1 for b in known if b >= CONTEXT_WINDOW)
        print(
            f"  {s['name']}: {s['passes']}/{s['runs']} passed, {s['not_found_no_write']} "
            f"replied not found with no write; {summary['wall_s']} s",
            flush=True,
        )
    return summary, capture


def hypothesis(arms: dict[str, Any]) -> dict[str, Any]:
    """The milestone's rule, for 10 runs an arm (scaled to others): the mutant replies "not
    found" with no write in at least 8 of 10 and within 2 of the unmutated arm, so the
    flag is near-equivalent; fewer than 6, with the unmutated arm more than 2 above it,
    means the flag matters to a real model; a mutant under 6 that the unmutated arm does
    not beat by more than 2 says the task fails either way, not that the flag matters;
    anything else is partial."""
    counts = {label: arm["scenarios"][0]["not_found_no_write"] for label, arm in arms.items()}
    runs = {label: arm["scenarios"][0]["runs"] for label, arm in arms.items()}
    mutant, unmutated = counts.get("drops_error_flag"), counts.get("unmutated")
    if mutant is None or unmutated is None:
        return {"not_found_no_write": counts, "runs": runs, "verdict": "incomplete"}
    n = runs["drops_error_flag"]
    difference = unmutated - mutant
    gap = 0.2 * n  # "within 2" of 10
    if mutant >= 0.8 * n and abs(difference) <= gap:
        verdict = "near-equivalent: the mutant replies not found as the unmutated agent does"
    elif mutant < 0.6 * n and difference > gap:
        verdict = "the flag matters to the real model"
    elif mutant < 0.6 * n:
        verdict = "inconclusive: the unmutated arm fails the task about as often"
    else:
        verdict = "partial: between the two thresholds, or more than 2 from the unmutated arm"
    return {
        "not_found_no_write": counts,
        "runs": runs,
        "difference": difference,
        "rule": "mutant >= 8/10 and within 2 of unmutated: near-equivalent; mutant < 6/10 "
        "and unmutated more than 2 above it: the flag matters; mutant < 6/10 otherwise: "
        "inconclusive; else partial",
        "verdict": verdict,
    }


def run_arms(
    runs: int,
    out: Path = OUT,
    *,
    arms: Sequence[Arm] = ARMS,
    client: str = CLIENT,
    environment: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    recording_log: Path = RECORDING_LOG,
    qwen_cassettes: Path = QWEN_CASSETTES,
) -> dict[str, Any]:
    """Every arm on the qwen recording (else the fake-model one); the report is rewritten
    after each arm so a stopped run keeps the arm it finished."""
    qwen = qwen_cassettes / f"{TASK.id}.jsonl"
    cassette_dir = qwen_cassettes if qwen.exists() else FAKE_CASSETTES
    log = json.loads(recording_log.read_text(encoding="utf-8")) if recording_log.exists() else None
    # The recording's first call ran on a cold cache: the full system prompt, tools and task.
    prefix = log["attempts"][0]["input_tokens"][0] if log and log["attempts"] else None
    result: dict[str, Any] = {
        "command": f"uv run python -m eval.retail_error --runs {runs}",
        "suite": _relative(SUITE),
        "task": TASK.id,
        "instruction": TASK.instruction,
        "mode": "model-live",
        "policy": "fallback",
        "identify": "the suite's [defaults.identify] keys",
        "client": client,
        "cassette": _relative(cassette_dir / f"{TASK.id}.jsonl"),
        "cassette_source": "qwen recording"
        if qwen.exists()
        else "fake-model recording (no qwen attempt passed)",
        "recording_attempts": None if log is None else len(log["attempts"]),
        "environment": environment if environment is not None else describe(ollama.ollama_model()),
        "context_window": CONTEXT_WINDOW,
        "prompt_prefix_tokens": prefix,
        "flag_probe": probe,
        "arms": {},
        "hypothesis": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        summary, _ = run_arm(arm, runs, client=client, cassette_dir=cassette_dir, prefix=prefix)
        result["arms"][arm.label] = summary
        result["hypothesis"] = hypothesis(result["arms"])
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['hypothesis']['verdict']}; wrote {out}")
    return result


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _serving(check: Callable[[str], dict[str, str] | None] = ollama_model) -> bool:
    if check(ollama.ollama_model()) is not None:
        return True
    print(
        f"Ollama at {ollama.ollama_url()} is not serving {ollama.ollama_model()}; start it "
        f"(ollama serve) and pull the model (ollama pull {ollama.ollama_model()}), or set "
        "REPLAY_OLLAMA_URL and REPLAY_OLLAMA_MODEL",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--record", action="store_true", help="record the task (step 1)")
    parser.add_argument("--attempts", type=int, default=3, help="recording attempts")
    parser.add_argument("--runs", type=int, default=10, help="model-live runs per arm")
    parser.add_argument("--out", type=Path, default=OUT, help="raw report of the arms")
    parser.add_argument("--log", type=Path, default=RECORDING_LOG, help="recording log")
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.runs < 1:
        parser.error("--attempts and --runs must be at least 1")
    if not _serving():
        return 2
    if args.record:
        recorded = record_task(args.attempts, args.log.resolve())
        return 0 if recorded["kept"] else 1
    probe = probe_flag(make_client(CLIENT), ollama.ollama_model())  # type: ignore[arg-type]
    print(f"flag probe: {probe['reading']} ({[p['input_tokens'] for p in probe['probes']]})")
    run_arms(args.runs, args.out.resolve(), probe=probe, recording_log=args.log.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
