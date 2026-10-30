"""M18 and M19: retail tasks and agent mutations on a real model, through Ollama.

M18 asked whether a lost ``is_error`` flag matters to a model that reads the error text,
on the hand-written task ``account_not_found_wrong_zip`` (a wrong zip code, so
``find_user_id_by_name_zip`` fails and the right reply is that the account was not
found, with nothing written). M19 runs the agent mutations the scripted model catches on
``lookup_latest_order_total`` (``--task``, ``--mutations``) and asks which of them are
regressions for qwen2.5:7b, and by which check. Each task has its suite, its fake-model
cassettes, its reports and its default mutations (:data:`SETUPS`); without ``--task``
and ``--mutations`` the command is M18's.

1. ``--record``: record the task on the local model, up to ``--attempts`` (3) times;
   the first attempt that passes the suite's checks is kept as
   ``examples/retail_agent/cassettes/qwen/<task>.jsonl``. Every attempt is logged (for
   M18 ``eval/results/m18_qwen_recording.json``) with each model call's
   ``usage.input_tokens``, so a prompt Ollama cut to its context window is seen
   rather than trusted. Ollama counts only the tokens its prompt cache did not
   already hold, so each run also gets an upper bound on its largest prompt
   (:func:`prompt_bound`); with ``--server-log`` the server's own count of every
   prompt (``task.n_tokens``) is read instead.
2. The probe, run first by the default command: what Ollama's ``/v1/messages`` passes
   on to the model. M18's (:func:`probe_flag`) sends the same tool result with and
   without ``is_error: true``; M19's (:func:`probe_ids`) the same tool result with its
   matching ``tool_use_id`` and with a mismatched one. Each has a warm-up first and a
   three-word control last; equal ``input_tokens`` with and without the change means it
   never reached the model.
3. The arms (the default, after the probe): ``model-live`` under ``fallback`` on the
   qwen recording (the fake-model one if no qwen attempt passed, and the report says
   so): ``--runs`` (10) runs of the unmutated agent, then ``--mutant-runs`` (default
   ``--runs``) of each mutation, every one on the local model. Every run records its
   failed checks and its cause, its tool calls, the most ``tool_use`` blocks in one
   reply, its answer and each model call's tokens (and with ``--server-log`` each
   prompt's size and whether the server cut it). M18's report answers its flag
   hypothesis (:func:`hypothesis`), M19's the detection rule fixed in PLAN.md beside
   the scripted model's verdict on the same task (:func:`detection`).

Without Ollama serving the model every live part exits 2 with the fix named. The
harness is tested offline against the fake model (``tests/test_retail_error.py``,
``tests/test_retail_mutations_qwen.py``).

Usage::

    uv run python -m eval.retail_error --record      # make record-retail-error-qwen
    uv run python -m eval.retail_error               # make eval-retail-error-qwen
    uv run python -m eval.retail_error --task lookup_latest_order_total --record
    uv run python -m eval.retail_error --task lookup_latest_order_total \
        --runs 10 --mutant-runs 5 --server-log /tmp/ollama.log   # make eval-retail-mutations-qwen
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
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
from eval.mutation_study import scenario_catches
from eval.mutations import MUTATIONS
from eval.pydantic_ai import Capture, arm_summary
from examples.pydantic_ai_agent.environment import describe, ollama_model
from examples.retail_agent import ollama
from examples.retail_agent.rules import reports_account_not_found
from examples.retail_agent.tasks import ACCOUNT_NOT_FOUND, ALL_TASKS, HANDWRITTEN, Task
from examples.retail_agent.tools import reset_db

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "retail_agent"
SUITE = EXAMPLE / "replay-error.toml"
QWEN_SUITE = EXAMPLE / "replay-qwen.toml"
SCRIPTED_SUITE = EXAMPLE / "replay.toml"
FAKE_CASSETTES = EXAMPLE / "cassettes" / "error"
QWEN_CASSETTES = EXAMPLE / "cassettes" / "qwen"
OUT = ROOT / "eval" / "results" / "m18_error_qwen.json"
RECORDING_LOG = ROOT / "eval" / "results" / "m18_qwen_recording.json"
CLIENT = "examples.retail_agent.ollama:make_client"
FAKE_CLIENT = "examples.retail_agent.fake_model:make_client"
TASK = HANDWRITTEN[ACCOUNT_NOT_FOUND]
LOOKUP = "lookup_latest_order_total"
CONTEXT_WINDOW = 4096
"""The context Ollama 0.40 gives qwen2.5:7b by default (``llama-server -c 4096``, with
context shift on); a prompt that reaches it may have been cut. M19 starts the server
with ``OLLAMA_CONTEXT_LENGTH=8192`` and still counts the runs that pass 4,096."""

ACTS_ON_PARALLEL = frozenset({"first_tool_only", "mismatches_tool_use_ids"})
"""Mutations that change a run only when one reply holds two or more ``tool_use``
blocks; with fewer they never acted, a pass is not a survival and a failure is not a catch."""

M19_MUTATIONS = (
    "skips_order_lookup",
    "first_tool_only",
    "truncates_tool_results",
    "max_turns_too_low",
    "returns_empty_answer",
    "mismatches_tool_use_ids",
    "strips_policy",
    "truncates_history",
)
"""The eight agent mutations that can act on a read-only task."""

UNABLE_TO_ACT = {
    "wrong_write_argument": "changes the arguments of writes; the task makes none",
    "repeats_write_calls": "repeats writes; the task makes none",
    "drops_error_flag": "acts on a failed tool call; the task's tools all succeed",
    "unregisters_product_tool": "removes get_product_details, which the task never needs",
}
"""The other four agent mutations, and why none can change a run of the lookup task."""

PREDICTED = {
    "skips_order_lookup": ("caught", "the order read answers 'lookup skipped', so no total"),
    "truncates_tool_results": ("caught", "120 characters cut the order JSON before its total"),
    "truncates_history": ("caught", "the last four messages no longer hold the request"),
    "max_turns_too_low": ("caught", "the lookup needs four turns: run_error:RuntimeError"),
    "returns_empty_answer": ("caught", "the control: an empty answer has no total"),
    "first_tool_only": ("survives", "acts only on a reply with two tool_use blocks"),
    "mismatches_tool_use_ids": ("survives", "acts only on a reply with two tool_use blocks"),
    "strips_policy": ("survives", "a read-only task's ground truth does not need the policy"),
}
"""M19's hypothesis per mutation, with its mechanism, fixed in PLAN.md before the run."""


@dataclass(frozen=True)
class Arm:
    label: str
    agent: str


ARMS = (
    Arm("unmutated", "examples.retail_agent.ollama:run"),
    Arm("drops_error_flag", "eval.mutations.agents:drops_error_flag_ollama"),
)


def arms_for(mutations: Sequence[str]) -> tuple[Arm, ...]:
    """The unmutated agent, then each mutation's agent on the local model."""
    return (ARMS[0], *(Arm(m, f"eval.mutations.agents:{m}_ollama") for m in mutations))


@dataclass(frozen=True)
class Setup:
    """One task's suite, fake-model cassettes, reports and default mutations."""

    task: Task
    suite: Path
    fake_cassettes: Path
    out: Path
    recording_log: Path
    mutations: tuple[str, ...]


SETUPS = {
    ACCOUNT_NOT_FOUND: Setup(
        TASK, SUITE, FAKE_CASSETTES, OUT, RECORDING_LOG, ("drops_error_flag",)
    ),
    LOOKUP: Setup(
        ALL_TASKS[LOOKUP],
        QWEN_SUITE,
        EXAMPLE / "cassettes",
        ROOT / "eval" / "results" / "m19_retail_mutations_qwen.json",
        ROOT / "eval" / "results" / "m19_qwen_recording.json",
        M19_MUTATIONS,
    ),
}
M18 = SETUPS[ACCOUNT_NOT_FOUND]

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


def most_tool_uses(result: RunResult) -> int:
    """The most ``tool_use`` blocks in one of the run's model replies: a mutation that
    acts only on parallel calls (:data:`ACTS_ON_PARALLEL`) never acted below 2."""
    most = 0
    for step in result.trajectory.model_steps():
        content = step.response.get("content") if isinstance(step.response, dict) else None
        if isinstance(content, list):
            uses = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
            most = max(most, uses)
    return most


def run_facts(
    result: RunResult | None, prefix: int | None = None, task: str = ACCOUNT_NOT_FOUND
) -> dict[str, Any]:
    """One run's tool calls (with each result, or the error a failed call raised), its
    answer, the most ``tool_use`` blocks in one reply, each model call's
    ``input_tokens`` and ``output_tokens``, and the bound on its largest prompt
    (:func:`prompt_bound`); on M18's task also the not-found verdict and the
    ``is_error`` flags its requests carried."""
    if result is None:
        return {}
    steps = result.trajectory.model_steps()
    inputs = [_usage(s.response) for s in steps]
    outputs = [_usage(s.response, "output_tokens") for s in steps]
    facts: dict[str, Any] = {
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
        "most_tool_uses_in_a_reply": most_tool_uses(result),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "prompt_tokens_bound": prompt_bound(inputs, outputs, prefix),
    }
    if task == ACCOUNT_NOT_FOUND:
        verdict, reason = reports_account_not_found(result)
        facts["not_found_no_write"] = verdict
        facts["not_found_reason"] = reason
        facts["error_flags_sent"] = error_flags_sent(result)
    return facts


# --- the server's own count of every prompt --------------------------------------------

_NEW_PROMPT = re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = \d+, task\.n_tokens = (\d+)")
_RELEASE = re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d+)")
_SLOT = re.compile(r"n_ctx_slot = \d+")


@dataclass
class ServerLog:
    """The Ollama server's log, read as it grows. llama-server logs one ``new prompt``
    line per request with the whole prompt's size (``task.n_tokens``, whatever the cache
    held) and the slot's context (``n_ctx_slot``), then one ``stop processing`` line
    with ``truncated``; runs are sequential, so the lines added during a run are its
    requests."""

    path: Path
    offset: int = 0

    def skip(self) -> None:
        """Start reading from the end: what was logged before belongs to no run."""
        self.offset = self.path.stat().st_size if self.path.exists() else 0

    def read(self) -> list[dict[str, int]]:
        """The requests logged since the last read: prompt tokens, slot context, truncated."""
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            text = f.read().decode("utf-8", errors="replace")
        # Only whole lines; a request still being logged is read next time.
        complete = text[: text.rfind("\n") + 1]
        self.offset += len(complete.encode("utf-8", errors="replace"))
        requests: list[dict[str, int]] = []
        for line in complete.splitlines():
            if m := _NEW_PROMPT.search(line):
                requests.append({"n_ctx_slot": int(m[1]), "prompt_tokens": int(m[2])})
            elif (m := _RELEASE.search(line)) and requests and "truncated" not in requests[-1]:
                requests[-1]["truncated"] = int(m[2])
        return requests

    def slot_line(self) -> str | None:
        """The first line that names the slot's context (the model load's ``n_ctx_slot``)."""
        if not self.path.exists():
            return None
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _SLOT.search(line):
                return line.strip()
        return None


def server_facts(requests: list[dict[str, int]]) -> dict[str, Any]:
    prompts = [r["prompt_tokens"] for r in requests]
    return {
        "prompt_tokens": prompts,
        "largest_prompt": max(prompts, default=None),
        "truncated_requests": sum(1 for r in requests if r.get("truncated")),
        "n_ctx_slot": sorted({r["n_ctx_slot"] for r in requests}),
    }


@dataclass
class RetailCapture(Capture):
    """:class:`Capture` with the retail run's facts (:func:`run_facts`) in place of the
    travel desk's wire facts. ``prefix``: the cold count of the first prompt;
    ``server``: the server log, whose new requests each run takes."""

    prefix: int | None = None
    task: str = ACCOUNT_NOT_FOUND
    server: ServerLog | None = None

    def save(self, report: ScenarioReport, result: RunResult | None, started: datetime) -> str:
        run_id = f"{report.name}/{len(self.runs)}"
        facts = run_facts(result, self.prefix, self.task)
        if self.server is not None:
            facts["server"] = server_facts(self.server.read())
        self.runs[run_id] = facts
        if result is not None:
            self.results[run_id] = result
        return run_id


# --- 1. recording ---------------------------------------------------------------------


def _check(cassette_dir: Path, setup: Setup = M18, agent: str | None = None) -> ScenarioReport:
    suite = load_suite(setup.suite)
    scenario = next(s for s in suite.scenario if s.name == setup.task.id)
    report, _ = run_scenario(
        scenario,
        suite.defaults,
        Overrides(cassette_dir=cassette_dir, agent=agent),
        setup.suite.parent,
        time.perf_counter,
    )
    return report


def ollama_metadata() -> dict[str, Any]:
    """The cassette header's metadata for a recording on Ollama: model, server, digest."""
    metadata: dict[str, Any] = {"model": ollama.ollama_model(), "provider": "ollama"}
    metadata.update(ollama_model(ollama.ollama_model()) or {})
    return metadata


def record_attempt(
    agent: str,
    client: str,
    scratch: Path,
    metadata: dict[str, Any],
    prefix: int | None = None,
    setup: Setup = M18,
) -> tuple[dict[str, Any], Path]:
    """Record the task once into ``scratch`` and check it in ``full`` with the suite.
    ``prefix``: the first attempt's cold count of the first prompt (:func:`prompt_bound`)."""
    reset_db()
    started = time.perf_counter()
    result, cassette = record(
        load_agent(agent),
        setup.task.instruction,
        client=make_client(client),
        agent_ref=agent,
        metadata=metadata,
    )
    wall_s = round(time.perf_counter() - started, 1)
    path = scratch / f"{setup.task.id}.jsonl"
    cassette.dump(path)
    report = _check(scratch, setup)
    reset_db()
    entry = {
        "passed": report.passed,
        "failed_checks": [c.check for c in report.checks if not c.passed],
        "error": result.error.message if result.error else None,
        **run_facts(result, prefix, setup.task.id),
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
    setup: Setup = M18,
) -> dict[str, Any]:
    """Up to ``attempts`` recordings; the first that passes is copied into ``out``.
    ``metadata`` goes into each cassette's header (default :func:`ollama_metadata`)."""
    header = metadata if metadata is not None else ollama_metadata()
    task = setup.task
    entries: list[dict[str, Any]] = []
    kept = False
    for k in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            # Later attempts find the prompt in Ollama's cache; the first one's count is cold.
            prefix = (
                entries[0]["input_tokens"][0] if entries and entries[0]["input_tokens"] else None
            )
            entry, path = record_attempt(agent, client, Path(tmp), header, prefix, setup)
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
    command = "uv run python -m eval.retail_error --record"
    result = {
        "command": command if setup is M18 else f"{command} --task {task.id}",
        "task": task.id,
        "instruction": task.instruction,
        "environment": environment if environment is not None else describe(ollama.ollama_model()),
        "context_window": CONTEXT_WINDOW,
        "max_attempts": attempts,
        "kept": _relative(out / f"{task.id}.jsonl") if kept else None,
        "attempts": entries,
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"logged {len(entries)} attempt(s) to {log}")
    return result


# --- 2. what reaches the model ----------------------------------------------------------

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

ID_PROBES = (
    ("W", "call_1", "noah_ito_3850"),
    ("A", "call_1", "noah_ito_3850"),
    ("B", "call_2", "noah_ito_3850"),
    ("C", "call_1", "noah_ito_3850 (one exact match)"),
)
"""(nonce, the result's ``tool_use_id``, tool result text), the tool call's id always
``call_1``: W warms the cache; A's id matches the call, B's does not (same length, so a
template that printed ids would count the same); C is the control, A with three more
words in the result."""


def _probe_request(client: Anthropic, model: str, nonce: str, result: dict[str, Any]) -> int:
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
    return response.usage.input_tokens


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
        tokens = _probe_request(client, model, nonce, result)
        rows.append({"nonce": nonce, "content": text, "is_error": flag, "input_tokens": tokens})
    _, flagged, unflagged, control = (r["input_tokens"] for r in rows)
    return {
        "probes": rows,
        "flag_changes_prompt": flagged != unflagged,
        "control_changes_prompt": control != unflagged,
        "reading": _reading("flag", changed=flagged != unflagged, control=control != unflagged),
    }


def probe_ids(client: Anthropic, model: str) -> dict[str, Any]:
    """Send :data:`ID_PROBES` as :func:`probe_flag` sends its probes: does a mismatched
    ``tool_use_id`` reach the model (so ``mismatches_tool_use_ids`` could act if a reply
    ever held two calls), or does Ollama drop the id before the prompt?"""
    rows = []
    for nonce, tool_use_id, text in ID_PROBES:
        result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
        tokens = _probe_request(client, model, nonce, result)
        rows.append(
            {"nonce": nonce, "tool_use_id": tool_use_id, "content": text, "input_tokens": tokens}
        )
    _, matching, mismatched, control = (r["input_tokens"] for r in rows)
    return {
        "probes": rows,
        "id_changes_prompt": matching != mismatched,
        "control_changes_prompt": control != matching,
        "reading": _reading("id", changed=matching != mismatched, control=control != matching),
    }


def _reading(what: str, *, changed: bool, control: bool) -> str:
    if changed:
        return f"the {what} reaches the model"
    if control:
        return f"the {what} never reaches the model"
    return "inconclusive: the control did not move the count"


# --- 3. the arms ----------------------------------------------------------------------


def run_arm(
    arm: Arm,
    runs: int,
    *,
    client: str,
    cassette_dir: Path,
    suite: Path = SUITE,
    prefix: int | None = None,
    task: str = ACCOUNT_NOT_FOUND,
    server: ServerLog | None = None,
) -> tuple[dict[str, Any], RetailCapture]:
    """``runs`` model-live runs of the task under ``fallback`` with ``arm``'s agent."""
    capture = RetailCapture(prefix=prefix, task=task, server=server)
    settings = FlakeSettings(gate="wilson", max_runs=runs, concurrency=1)
    print(f"arm {arm.label}: {runs} model-live runs under fallback", flush=True)
    if server is not None:
        server.skip()
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
        if task == ACCOUNT_NOT_FOUND:
            s["not_found_no_write"] = sum(1 for r in rows if r.get("not_found_no_write"))
            s["runs_with_an_error_flag_sent"] = sum(1 for r in rows if r.get("error_flags_sent"))
        else:
            s["failed_checks"] = dict(
                sorted(Counter(c for r in rows for c in r["failed_checks"]).items())
            )
            s["run_errors"] = dict(
                sorted(Counter(r["error"].split(":")[0] for r in rows if r["error"]).items())
            )
            s["most_tool_uses_in_a_reply"] = max(
                (r.get("most_tool_uses_in_a_reply", 0) for r in rows), default=0
            )
        bounds = [r.get("prompt_tokens_bound") for r in rows]
        known = [b for b in bounds if isinstance(b, int)]
        s["max_prompt_tokens_bound"] = max(known, default=None)
        s["runs_without_a_bound"] = len(bounds) - len(known)
        s["runs_that_may_reach_the_context_window"] = sum(1 for b in known if b >= CONTEXT_WINDOW)
        if server is not None:
            largest = [r["server"]["largest_prompt"] for r in rows if r.get("server")]
            sized = [p for p in largest if isinstance(p, int)]
            s["largest_prompt"] = max(sized, default=None)
            s["runs_over_the_default_window"] = sum(1 for p in sized if p > CONTEXT_WINDOW)
            s["truncated_requests"] = sum(
                r["server"]["truncated_requests"] for r in rows if r.get("server")
            )
        found = (
            f", {s['not_found_no_write']} replied not found with no write"
            if task == ACCOUNT_NOT_FOUND
            else ""
        )
        print(
            f"  {s['name']}: {s['passes']}/{s['runs']} passed{found}; {summary['wall_s']} s",
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


# --- M19: the scripted model's verdict on the same task, and the detection rule ---------


def scripted_verdicts(mutations: Sequence[str], task: str = LOOKUP) -> dict[str, Any]:
    """The scripted model's ``model-live`` verdict on ``task`` alone, per mutation: the
    mutation study's configuration (``strict``, no identify keys) on replay.toml's
    scenario, with the fake model. Offline and deterministic."""
    suite = load_suite(SCRIPTED_SUITE)
    scenario = next(s for s in suite.scenario if s.name == task)
    agents = {m.id: m.agent for m in MUTATIONS if m.target == "agent"}
    verdicts: dict[str, Any] = {}
    for mutation in mutations:
        reset_db()
        try:
            report, _ = run_scenario(
                scenario,
                suite.defaults,
                Overrides(
                    agent=agents[mutation],
                    mode="model-live",
                    policy="strict",
                    client=FAKE_CLIENT,
                    identify={},
                ),
                SCRIPTED_SUITE.parent,
                time.perf_counter,
            )
        finally:
            reset_db()
        caught_by = sorted(scenario_catches(report))
        verdicts[mutation] = {
            "verdict": "survives" if report.passed else "caught",
            "caught_by": caught_by,
            "by_a_crash": _crashes(caught_by),
        }
    return verdicts


def _crashes(caught_by: Sequence[str]) -> list[str]:
    """Run errors other than the turn limit's ``RuntimeError``: a parser or the replay
    failing, not a check reading the run."""
    return [c for c in caught_by if c.startswith("run_error:") and c != "run_error:RuntimeError"]


def detection(arms: dict[str, Any], scripted: dict[str, Any] | None = None) -> dict[str, Any]:
    """M19's rule, fixed in PLAN.md before the run: resolvable only if the unmutated agent
    passes at least 8 of 10 (80%); then a mutation is caught at 1 of 5 passes or fewer
    (20%), survives at 4 of 5 or more (80%), and is unresolved between. A mutation that
    acts only on parallel calls (:data:`ACTS_ON_PARALLEL`) and never saw a reply with two
    ``tool_use`` blocks never acted. Each verdict is set beside the prediction and the
    scripted model's verdict on the same task, with the checks and run errors that
    caught it."""
    unmutated = arms.get("unmutated", {}).get("scenarios", [{}])[0]
    runs, passes = unmutated.get("runs", 0), unmutated.get("passes", 0)
    resolvable = runs > 0 and passes >= 0.8 * runs
    rows: dict[str, Any] = {}
    for label, arm in arms.items():
        if label == "unmutated":
            continue
        s = arm["scenarios"][0]
        n, k = s["runs"], s["passes"]
        verdict = "caught" if k <= 0.2 * n else "survives" if k >= 0.8 * n else "unresolved"
        most = s.get("most_tool_uses_in_a_reply", 0)
        acted = label not in ACTS_ON_PARALLEL or most >= 2
        caught_by = sorted(
            {c for r in s["outcomes"] for c in r["failed_checks"]}
            | {f"run_error:{r['error'].split(':')[0]}" for r in s["outcomes"] if r["error"]}
        )
        # A mutation that never acted cannot have caused a failure: its passes are the
        # baseline's, so it is never acted whatever the rule reads (``rule_verdict``).
        reported = verdict if acted else "never acted"
        predicted = PREDICTED.get(label, (None, None))
        row: dict[str, Any] = {
            "passes": k,
            "runs": n,
            "verdict": reported if resolvable else "not resolvable",
            "rule_verdict": verdict,
            "acted": acted,
            "most_tool_uses_in_a_reply": most,
            "caught_by": caught_by,
            "by_a_crash": _crashes(caught_by),
            "predicted": predicted[0],
            "mechanism": predicted[1],
        }
        if resolvable and acted and predicted[0] is not None:
            row["as_predicted"] = (verdict == "survives") == (predicted[0] == "survives") and (
                verdict != "unresolved"
            )
        if scripted is not None and label in scripted:
            row["scripted"] = scripted[label]
            if resolvable and acted and verdict != "unresolved":
                row["agrees_with_scripted"] = verdict == scripted[label]["verdict"]
        rows[label] = row
    return {
        "rule": "resolvable only if the unmutated agent passes >= 8/10; a mutation is caught "
        "at <= 1/5 passes, survives at >= 4/5, unresolved between; a mutation that acts only "
        "on parallel calls and never saw two tool_use blocks in one reply never acted",
        "unmutated": {"passes": passes, "runs": runs},
        "resolvable": resolvable,
        "verdict": "resolvable"
        if resolvable
        else f"not resolvable on {ollama.ollama_model()} for this task",
        "mutations": rows,
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
    setup: Setup = M18,
    mutant_runs: int | None = None,
    server: ServerLog | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Every arm on the qwen recording (else the fake-model one); the report is rewritten
    after each arm so a stopped run keeps the arm it finished. ``mutant_runs``: runs of
    each arm after the first (default ``runs``)."""
    task = setup.task
    qwen = qwen_cassettes / f"{task.id}.jsonl"
    cassette_dir = qwen_cassettes if qwen.exists() else setup.fake_cassettes
    log = json.loads(recording_log.read_text(encoding="utf-8")) if recording_log.exists() else None
    # The recording's first call ran on a cold cache: the full system prompt, tools and task.
    prefix = log["attempts"][0]["input_tokens"][0] if log and log["attempts"] else None
    m18 = setup is M18
    result: dict[str, Any] = {
        "command": command or f"uv run python -m eval.retail_error --runs {runs}",
        "suite": _relative(setup.suite),
        "task": task.id,
        "instruction": task.instruction,
        "mode": "model-live",
        "policy": "fallback",
        "identify": "the suite's [defaults.identify] keys",
        "client": client,
        "cassette": _relative(cassette_dir / f"{task.id}.jsonl"),
        "cassette_source": "qwen recording"
        if qwen.exists()
        else "fake-model recording (no qwen attempt passed)",
        "recording_attempts": None if log is None else len(log["attempts"]),
        "environment": environment if environment is not None else describe(ollama.ollama_model()),
        "context_window": CONTEXT_WINDOW,
        "prompt_prefix_tokens": prefix,
        ("flag_probe" if m18 else "id_probe"): probe,
        "arms": {},
        ("hypothesis" if m18 else "detection"): {},
    }
    if not m18:
        result["unable_to_act"] = UNABLE_TO_ACT
        result["predicted"] = {
            m: {"verdict": v, "mechanism": why} for m, (v, why) in PREDICTED.items()
        }
        result["scripted"] = scripted_verdicts([a.label for a in arms[1:]], task.id)
        if server is not None:
            result["server_log"] = {"path": str(server.path), "n_ctx_slot_line": server.slot_line()}
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for i, arm in enumerate(arms):
        n = runs if i == 0 or mutant_runs is None else mutant_runs
        summary, _ = run_arm(
            arm,
            n,
            client=client,
            cassette_dir=cassette_dir,
            suite=setup.suite,
            prefix=prefix,
            task=task.id,
            server=server,
        )
        result["arms"][arm.label] = summary
        if m18:
            result["hypothesis"] = hypothesis(result["arms"])
        else:
            result["detection"] = detection(result["arms"], result["scripted"])
            result["arms_wall_s"] = round(time.perf_counter() - started, 1)
            result["window"] = window(result["arms"])
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = result["hypothesis" if m18 else "detection"]["verdict"]
    print(f"{verdict}; wrote {out}")
    return result


def window(arms: dict[str, Any]) -> dict[str, Any]:
    """Across every arm: the largest prompt the server saw, the runs whose largest prompt
    passed 4,096 tokens (Ollama's default window, cut there before M19), and any request
    the server truncated."""
    rows = [r for arm in arms.values() for s in arm["scenarios"] for r in s["outcomes"]]
    largest = [r["server"]["largest_prompt"] for r in rows if r.get("server")]
    sized = [p for p in largest if isinstance(p, int)]
    return {
        "runs": len(rows),
        "runs_with_server_counts": len(sized),
        "largest_prompt": max(sized, default=None),
        "runs_over_4096": sum(1 for p in sized if p > CONTEXT_WINDOW),
        "truncated_requests": sum(
            r["server"]["truncated_requests"] for r in rows if r.get("server")
        ),
        "largest_prompt_bound": max(
            (
                r["prompt_tokens_bound"]
                for r in rows
                if isinstance(r.get("prompt_tokens_bound"), int)
            ),
            default=None,
        ),
    }


def replay_qwen_recordings(qwen_cassettes: Path = QWEN_CASSETTES) -> bool:
    """Every task's qwen2.5:7b recording replayed in ``full`` by its suite, with the agent
    that recorded it: each must pass with no step served approximately. Offline (part of
    ``make eval``); a task without a qwen recording is named and skipped."""
    ok = True
    for setup in SETUPS.values():
        if not (qwen_cassettes / f"{setup.task.id}.jsonl").exists():
            print(f"SKIP {setup.task.id}: no qwen recording in {_relative(qwen_cassettes)}")
            continue
        reset_db()
        try:
            report = _check(qwen_cassettes, setup, agent=ARMS[0].agent)
        finally:
            reset_db()
        approximate = report.steps.by_status.get("approximate", 0)
        passed = report.passed and approximate == 0
        ok &= passed
        print(
            f"{'PASS' if passed else 'FAIL'} {_relative(setup.suite)} {setup.task.id}: "
            f"{report.steps.total} steps, {approximate} approximate"
        )
    return ok


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


def parse_mutations(text: str) -> tuple[str, ...]:
    """``--mutations``: a comma-separated list of agent mutation ids."""
    known = [m.id for m in MUTATIONS if m.target == "agent"]
    chosen = tuple(m.strip() for m in text.split(",") if m.strip())
    unknown = [m for m in chosen if m not in known]
    if unknown or not chosen:
        raise argparse.ArgumentTypeError(
            f"--mutations takes agent mutation ids ({', '.join(known)}); got {text!r}"
        )
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--task", choices=sorted(SETUPS), default=ACCOUNT_NOT_FOUND)
    parser.add_argument(
        "--mutations", type=parse_mutations, help="agent mutations (default: the task's)"
    )
    parser.add_argument("--record", action="store_true", help="record the task (step 1)")
    parser.add_argument("--attempts", type=int, default=3, help="recording attempts")
    parser.add_argument("--runs", type=int, default=10, help="model-live runs of the first arm")
    parser.add_argument("--mutant-runs", type=int, help="runs of each mutation (default --runs)")
    parser.add_argument("--out", type=Path, help="raw report of the arms (default: the task's)")
    parser.add_argument("--log", type=Path, help="recording log (default: the task's)")
    parser.add_argument(
        "--server-log", type=Path, help="the Ollama server's log, for every prompt's size"
    )
    parser.add_argument(
        "--offline", action="store_true", help="only replay the qwen recordings in full"
    )
    args = parser.parse_args(argv)
    if args.offline:
        return 0 if replay_qwen_recordings() else 1
    if (
        args.attempts < 1
        or args.runs < 1
        or (args.mutant_runs is not None and args.mutant_runs < 1)
    ):
        parser.error("--attempts, --runs and --mutant-runs must be at least 1")
    setup = SETUPS[args.task]
    log = (args.log or setup.recording_log).resolve()
    if not _serving():
        return 2
    if args.record:
        recorded = record_task(args.attempts, log, setup=setup)
        return 0 if recorded["kept"] else 1
    client = make_client(CLIENT)
    server = ServerLog(args.server_log) if args.server_log else None
    if setup is M18:
        probe = probe_flag(client, ollama.ollama_model())  # type: ignore[arg-type]
        print(f"flag probe: {probe['reading']} ({[p['input_tokens'] for p in probe['probes']]})")
    else:
        if server is not None:
            server.skip()
        probe = probe_ids(client, ollama.ollama_model())  # type: ignore[arg-type]
        if server is not None:
            probe["server"] = server_facts(server.read())
        print(f"id probe: {probe['reading']} ({[p['input_tokens'] for p in probe['probes']]})")
    mutations = args.mutations or setup.mutations
    command = None
    if setup is not M18:
        command = " ".join(["uv run python -m eval.retail_error", *(argv or sys.argv[1:])])
    run_arms(
        args.runs,
        (args.out or setup.out).resolve(),
        arms=arms_for(mutations),
        probe=probe,
        recording_log=log,
        setup=setup,
        mutant_runs=args.mutant_runs,
        server=server,
        command=command,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
