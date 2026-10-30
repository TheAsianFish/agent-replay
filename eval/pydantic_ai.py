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
4. M15, ``--swap``: the change ``model-live`` exists to test. The same two scenarios
   with the live side changed, one arm each (:data:`SWAP_ARMS`): qwen2.5:3b under
   ``strict`` and ``fallback``, qwen2.5:7b without the prompt's guard line, the
   qwen2.5:7b control, and the booking at ``temperature=0``, after the probe calls
   (:func:`probe_temperature`) that show Ollama's ``/v1/messages`` takes the
   parameter. ``--model``, ``--prompt-variant`` and ``--temperature`` change the
   live side of an ``--arms`` run the same way. Every run row also carries its tool
   calls and what reached the wire (model, temperature, whether the guard was in the
   system prompt), and ``--swap`` writes the aligned diff of the first failed
   qwen2.5:3b ``strict`` run against its cassette beside the report.
5. M16, ``--long``: the rebooking chain (:data:`LONG_SCENARIO`, at least 9 served tool
   calls, each argument read from the previous result) under ``strict`` and
   ``fallback``, 20 runs per arm (15 when its recording took over 150 s). Each run's
   tool steps are kept as a status sequence, and each arm gets the survival curve (the
   share of runs with no world step by served-call index, against the independence
   prediction from M12's short runs), the rate of a world step after an earlier one in
   the same run against the arm's base rate, and the identity keys' firings
   (:func:`long_analysis`).
6. M17, ``--no-guard``: M15's no-guard arm (qwen2.5:7b without the prompt's guard line)
   again, under ``fallback`` and ``synth`` (:data:`NO_GUARD_ARMS`), 20 runs per arm of
   the two qwen-recorded scenarios. ``synth`` answers a call the recording lacks with a
   schema-minimal stub (``0.0`` for a conversion), which is a world step like any other
   step off the recording. Each run gains the figures its answer states and whether one
   of them is a stub's (:func:`figures`), and the report answers the milestone's
   hypotheses beside M15's ``strict`` arm (:func:`no_guard_hypotheses`).

Usage::

    uv run python -m eval.pydantic_ai [--runs 10] [--out eval/results/m9_flake_qwen.json]
    uv run python -m eval.pydantic_ai --arms strict,fallback --runs 20
    uv run python -m eval.pydantic_ai --swap
    uv run python -m eval.pydantic_ai --long
    uv run python -m eval.pydantic_ai --no-guard
    uv run python -m eval.pydantic_ai --arms strict --model qwen2.5:3b --prompt-variant no_guard
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from agent_replay.cassette import Cassette, Kind
from agent_replay.ci import Overrides, RunSink, ScenarioReport, load_suite, run_suite
from agent_replay.diff import diff_steps, steps_from_cassette, steps_from_trajectory
from agent_replay.diff import summary as diff_summary
from agent_replay.diff import to_markdown as diff_markdown
from agent_replay.flake import FlakeReport, FlakeSettings, RunOutcome, flake_suite
from agent_replay.runner import RunResult
from agent_replay.session import Policy
from examples.pydantic_ai_agent.agent import GUARD, MODEL, PromptVariant, ollama_url
from examples.pydantic_ai_agent.environment import describe, ollama_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pydantic_ai_agent"
SUITES = (EXAMPLE / "replay.toml", EXAMPLE / "replay-qwen.toml")
QWEN_SUITE = SUITES[1]
CLIENT = "examples.pydantic_ai_agent.agent:make_client"
OUT = ROOT / "eval" / "results" / "m9_flake_qwen.json"
ARMS_OUT = ROOT / "eval" / "results" / "m12_arms_qwen.json"
SWAP_OUT = ROOT / "eval" / "results" / "m15_swap_qwen.json"
LONG_OUT = ROOT / "eval" / "results" / "m16_long_qwen.json"
LONG_RECORDING_LOG = ROOT / "eval" / "results" / "m16_qwen_recording.json"
NO_GUARD_OUT = ROOT / "eval" / "results" / "m17_no_guard_qwen.json"
SMALL_MODEL = "qwen2.5:3b"
AGENTS: dict[tuple[PromptVariant, float | None], str] = {
    ("recording", None): "examples.pydantic_ai_agent.agent:run",
    ("no_guard", None): "examples.pydantic_ai_agent.agent:run_no_guard",
    ("recording", 0.0): "examples.pydantic_ai_agent.agent:run_temperature_0",
}
"""The suite's ``agent`` for each prompt variant and temperature an arm can ask for."""
ARM_SCENARIOS = ("cheapest_fare_in_euros", "book_earliest_flight")
"""The scenarios whose cassettes qwen2.5:7b recorded (the third serves the fake model's)."""
CAUSES = ("passed", "unrecorded", "approximate", "synth", "model")
CAUSE_MEANING = {
    "passed": "every check passed",
    "unrecorded": "a call no unused recording matched (by tool, or by the identify keys) "
    "ended the run under strict: the frozen world's limit",
    "approximate": "a step was served approximately (another request's recording) and a "
    "check then failed: possibly the frozen world's fault",
    "synth": "a call the recording lacks was answered with a stub (the synth policy) and a "
    "check then failed or the agent raised: possibly the stub's fault",
    "model": "a check failed (or the agent raised) with every served step exact: the "
    "model's own failure",
}


def replay_full() -> bool:
    """Both suites in ``full``: all pass, and no step is served approximately. The
    committed cassettes were recorded on :data:`MODEL`, so an exported
    ``$REPLAY_OLLAMA_MODEL`` is set aside here: it would change every model request."""
    ok = True
    for suite in SUITES:
        with live_model(MODEL):
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
    pass even with an approximate or stubbed step; an unrecorded call ends a ``strict``
    run before any check runs; a failure after an approximate step may be the world's,
    and so may one after a stub; the rest is the model's."""
    if outcome.passed:
        return "passed"
    if outcome.error is not None and outcome.error.type == "UnrecordedCallError":
        return "unrecorded"
    if outcome.steps.get("approximate", 0):
        return "approximate"
    if outcome.steps.get("synth", 0):
        return "synth"
    return "model"


@dataclass
class Capture:
    """A :class:`RunSink` that keeps what the flake report leaves out of each run: its
    tool calls, its answer and what its model requests carried, by run id. With
    ``inner`` (a store's sink) the run is stored too and keeps the store's id."""

    inner: RunSink | None = None
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, RunResult] = field(default_factory=dict)

    def save(self, report: ScenarioReport, result: RunResult | None, started: datetime) -> str:
        stored = self.inner.save(report, result, started) if self.inner is not None else None
        run_id = stored or f"{report.name}/{len(self.runs)}"
        self.runs[run_id] = wire(result)
        if result is not None:
            self.results[run_id] = result
        return run_id

    def sink(self) -> RunSink:
        if self.inner is None:
            return RunSink(save=self.save)
        return RunSink(start=self.inner.start, save=self.save, finish=self.inner.finish)


def _temperature(request: dict[str, Any]) -> Any:
    """The request's temperature: pydantic-ai sends it in ``extra_body`` (``anthropic`` 1.x
    has no ``temperature`` argument), which the SDK merges into the JSON body."""
    extra = request.get("extra_body")
    if isinstance(extra, dict) and "temperature" in extra:
        return extra["temperature"]
    return request.get("temperature")


def wire(result: RunResult | None) -> dict[str, Any]:
    """One run's tool calls (name, input, status, and for a call off the recording that
    still returned, the result it got), its answer, the model, temperature and guard line
    its model requests carried (``None`` when the request had none), and how many of its
    model replies asked for more than one tool at once."""
    if result is None:
        return {}
    steps = result.trajectory.model_steps()
    requests = [s.request for s in steps if isinstance(s.request, dict)]
    models = sorted({str(r.get("model")) for r in requests})
    temperatures = sorted({json.dumps(_temperature(r)) for r in requests})
    guard = GUARD.split("\n", 1)[0]
    return {
        "tool_calls": [
            {
                "name": s.name,
                "input": s.request,
                "status": s.status.value,
                **({"response": s.response} if s.status.value in OFF_RECORDING else {}),
            }
            for s in result.trajectory.tool_steps()
        ],
        "output": result.output,
        "request_models": models,
        "request_temperatures": [json.loads(t) for t in temperatures],
        "guard_in_system": sorted({guard in json.dumps(r.get("system")) for r in requests}),
        "parallel_tool_replies": sum(1 for s in steps if _tool_uses(s.response) > 1),
    }


OFF_RECORDING = ("fallback", "synth")
"""Statuses of a tool step that ran off the recording and still returned a result."""


def _tool_uses(response: Any) -> int:
    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, list):
        return 0
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")


def _run_row(outcome: RunOutcome, captured: dict[str, Any] | None = None) -> dict[str, Any]:
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
        **({} if captured is None else {"run_id": outcome.run_id, **captured}),
    }


def tool_path(row: dict[str, Any]) -> list[list[Any]]:
    """A run's tool calls as comparable ``[name, input]`` pairs."""
    return [[c["name"], c["input"]] for c in row.get("tool_calls", [])]


def arm_summary(
    report: FlakeReport, wall_s: float, capture: Capture | None = None
) -> dict[str, Any]:
    """One arm: per scenario its pass and cause counts, the gate's verdict and every run.
    With ``capture``, each run row gains its tool calls and wire facts, and each scenario
    the number of distinct tool paths and answers among its runs."""
    scenarios = []
    for s in report.scenarios:
        runs = [
            _run_row(o, None if capture is None else capture.runs.get(o.run_id or "", {}))
            for o in s.outcomes
            if o.counted
        ]
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
                **(
                    {}
                    if capture is None
                    else {
                        "distinct_tool_paths": len({json.dumps(tool_path(r)) for r in runs}),
                        "distinct_outputs": len({json.dumps(r.get("output")) for r in runs}),
                    }
                ),
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


@dataclass(frozen=True)
class Arm:
    """One M15 arm: a policy and the live side's model, prompt and temperature."""

    label: str
    policy: Policy
    model: str = MODEL
    prompt_variant: PromptVariant = "recording"
    temperature: float | None = None
    runs: int = 20
    scenarios: tuple[str, ...] = ARM_SCENARIOS

    @property
    def agent(self) -> str:
        try:
            return AGENTS[(self.prompt_variant, self.temperature)]
        except KeyError:
            raise ValueError(
                f"arm {self.label}: no agent for prompt variant {self.prompt_variant!r} at "
                f"temperature {self.temperature}; the agent has {sorted(AGENTS, key=str)}"
            ) from None


SWAP_ARMS = (
    Arm("3b_strict", "strict", model=SMALL_MODEL),
    Arm("3b_fallback", "fallback", model=SMALL_MODEL),
    Arm("7b_no_guard_strict", "strict", prompt_variant="no_guard"),
    Arm("7b_control_strict", "strict"),
    Arm(
        "7b_temperature_0_strict",
        "strict",
        temperature=0.0,
        runs=10,
        scenarios=("book_earliest_flight",),
    ),
)
"""M15's arms, in the order they run."""

PROBE_PROMPT = "Invent a name for a new travel agency and give it a one-line slogan."
PROBE_TEMPERATURES = (0.0, 0.0, 1.0, 1.0)
"""Two calls at temperature 0, which should agree, then two at 1, which should not."""


@contextlib.contextmanager
def live_model(model: str) -> Iterator[None]:
    """``$REPLAY_OLLAMA_MODEL`` set to ``model`` for the block (the agent reads it per run)."""
    before = os.environ.get("REPLAY_OLLAMA_MODEL")
    os.environ["REPLAY_OLLAMA_MODEL"] = model
    try:
        yield
    finally:
        if before is None:
            del os.environ["REPLAY_OLLAMA_MODEL"]
        else:
            os.environ["REPLAY_OLLAMA_MODEL"] = before


def _post(path: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(  # noqa: S310 - local server URL from config
        f"{ollama_url()}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        return json.loads(response.read())


def probe_temperature(model: str = MODEL, post: Any = _post) -> dict[str, Any]:
    """Ollama's ``/v1/messages`` at :data:`PROBE_TEMPERATURES` on one open-ended prompt:
    the answers at temperature 0 should be identical and those at 1 should differ, or
    the server is ignoring the parameter."""
    calls: list[dict[str, Any]] = []
    for temperature in PROBE_TEMPERATURES:
        body = {
            "model": model,
            "max_tokens": 60,
            "temperature": temperature,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
        }
        reply = post("/v1/messages", body)
        text = "".join(b.get("text", "") for b in reply.get("content", []))
        calls.append({"request": body, "text": text})
    by_t: dict[float, set[str]] = {}
    for c in calls:
        by_t.setdefault(c["request"]["temperature"], set()).add(c["text"])
    return {
        "endpoint": "/v1/messages",
        "calls": calls,
        "identical_at_0": len(by_t.get(0.0, set())) == 1,
        "identical_at_1": len(by_t.get(1.0, set())) == 1,
    }


def run_arm(
    arm: Arm,
    *,
    client: str = CLIENT,
    suite: Path = QWEN_SUITE,
    sink: RunSink | None = None,
) -> tuple[dict[str, Any], Capture]:
    """One arm in ``model-live``; returns its summary and the captured runs."""
    capture = Capture(inner=sink)
    settings = FlakeSettings(max_runs=arm.runs, concurrency=1)
    print(
        f"arm {arm.label}: {arm.runs} model-live runs of {', '.join(arm.scenarios)} "
        f"({arm.model}, {arm.prompt_variant}, temperature {arm.temperature}, {arm.policy})",
        flush=True,
    )
    started = time.perf_counter()
    with live_model(arm.model):
        report = flake_suite(
            suite,
            settings=settings,
            overrides=Overrides(
                mode="model-live", policy=arm.policy, client=client, agent=arm.agent
            ),
            scenarios=list(arm.scenarios),
            sink=capture.sink(),
        )
    summary = arm_summary(report, time.perf_counter() - started, capture)
    summary.update(
        model=arm.model,
        prompt_variant=arm.prompt_variant,
        temperature=arm.temperature,
        agent=arm.agent,
        settings=settings.model_dump(mode="json"),
    )
    for s in summary["scenarios"]:
        counts = ", ".join(f"{n} {c}" for c, n in s["causes"].items() if n)
        print(
            f"  {s['name']}: {s['passes']}/{s['runs']} passed ({counts}); "
            f"{s['unrecorded_by_identity']} by identity, {s['distinct_tool_paths']} tool paths",
            flush=True,
        )
    print(f"  wall time {summary['wall_s']} s", flush=True)
    return summary, capture


def first_failed_diff(
    summary: dict[str, Any], capture: Capture, suite: Path = QWEN_SUITE
) -> dict[str, Any] | None:
    """The aligned diff (``replay diff``'s table) of the arm's first failed run against
    the cassette it replayed, or ``None`` when every run passed."""
    cassettes = {s.name: s.cassette for s in load_suite(suite).scenario}
    for s in summary["scenarios"]:
        for row in s["outcomes"]:
            result = capture.results.get(row.get("run_id") or "")
            if row["passed"] or result is None:
                continue
            path = suite.parent / cassettes[s["name"]]
            label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            diff = diff_steps(
                steps_from_cassette(Cassette.load(path)),
                steps_from_trajectory(result.trajectory),
                base_label=label,
                head_label=f"{summary.get('model')} run {row['index']}",
            )
            return {
                "scenario": s["name"],
                "run_index": row["index"],
                "cause": row["cause"],
                "summary": diff_summary(diff),
                "markdown": diff_markdown(diff),
            }
    return None


def run_swap(
    arms: Sequence[Arm],
    out: Path,
    *,
    client: str = CLIENT,
    suite: Path = QWEN_SUITE,
    environment: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    sink: RunSink | None = None,
    command: str = "uv run python -m eval.pydantic_ai --swap",
) -> dict[str, Any]:
    """M15: each arm in turn, the report rewritten after every arm so a stopped run keeps
    the arms it finished; with an arm on :data:`SMALL_MODEL` under ``strict`` that failed
    a run, the diff of that run against its cassette goes beside the report (``.md``)."""
    result: dict[str, Any] = {
        "command": command,
        "suite": str(suite.relative_to(ROOT)) if suite.is_relative_to(ROOT) else str(suite),
        "client": client,
        "mode": "model-live",
        "identify": "the suite's [defaults.identify] keys",
        "recorded_on": MODEL,
        "environment": environment
        if environment is not None
        else {m: describe(m) for m in sorted({a.model for a in arms})},
        "temperature_probe": probe,
        "causes": CAUSE_MEANING,
        "arms": {},
        "diff": None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        summary, capture = run_arm(arm, client=client, suite=suite, sink=sink)
        result["arms"][arm.label] = summary
        if result["diff"] is None and arm.model == SMALL_MODEL and arm.policy == "strict":
            diff = first_failed_diff(summary, capture, suite)
            if diff is not None:
                markdown = diff.pop("markdown")
                md_path = out.with_suffix(".diff.md")
                md_path.write_text(markdown, encoding="utf-8")
                result["diff"] = {**diff, "markdown": md_path.name}
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return result


LONG_SCENARIO = "trace_rebooking_chain"
LONG_ARMS = (
    Arm("strict", "strict", scenarios=(LONG_SCENARIO,)),
    Arm("fallback", "fallback", scenarios=(LONG_SCENARIO,)),
)
"""M16's arms: the recording model and prompt, under each policy."""
LONG_RUNS, LONG_RUNS_SLOW, SLOW_RUN_S = 20, 15, 150.0
SHORT_RUN_RATE = (3, 143)
"""World steps per served tool call on M12's short runs (``m12_arms_qwen.json``: 3
unrecorded of 143 served tool steps), the rate an independent per-call model predicts
from."""
SHORTEST_CHAIN = 9
"""Served tool calls on the chain's shortest path: 8 ``get_booking`` and 1 conversion."""
LEVEL = 0.95


def is_world(status: str) -> bool:
    """A tool step off the recorded path: served from another request's recording
    (``approximate``) or not in the cassette (``unrecorded``, ``fallback``, ``synth``)."""
    return status != "exact"


def identity_firings(calls: Sequence[dict[str, Any]], recorded: Counter[str]) -> int:
    """Calls the identify keys turned away: off the recording while unused recordings of
    the same tool remained (under ``strict`` the run's error names the keys; under
    ``fallback`` the real tool ran). A call of a tool the cassette has no more of is
    off the path by count, not by identity."""
    served: Counter[str] = Counter()
    firings = 0
    for call in calls:
        if is_world(call["status"]) and call["status"] != "approximate":
            firings += served[call["name"]] < recorded[call["name"]]
        else:
            served[call["name"]] += 1
    return firings


def survival_curve(sequences: Sequence[Sequence[str]], per_call: float) -> list[dict[str, Any]]:
    """By served-call index n (from 1): the runs still at risk (n calls reached, no world
    step before), the world steps at n, the product-limit share with no world step
    through n (a run that ended earlier leaves the risk set, it does not count as a
    survivor), the plain share of all runs with no world step through n, and the
    independence prediction ``(1 - per_call)^n``."""
    runs = len(sequences)
    first = [next((i for i, s in enumerate(q, 1) if is_world(s)), None) for q in sequences]
    curve, survival = [], 1.0
    for n in range(1, max((len(q) for q in sequences), default=0) + 1):
        at_risk = sum(
            1
            for q, f in zip(sequences, first, strict=True)
            if len(q) >= n and (f is None or f >= n)
        )
        world = sum(1 for f in first if f == n)
        survival *= 1 - world / at_risk if at_risk else 1.0
        curve.append(
            {
                "index": n,
                "at_risk": at_risk,
                "world_steps": world,
                "survival": round(survival, 4),
                "share_without_world_step": round(
                    sum(1 for f in first if f is None or f > n) / runs, 4
                ),
                "independent": round((1 - per_call) ** n, 4),
            }
        )
    return curve


def clustering(sequences: Sequence[Sequence[str]], minimum: int = 5) -> dict[str, Any]:
    """Whether world steps cluster within runs: the share of steps that are world steps
    among those with an earlier world step in the same run, against the share among all
    steps. Independence predicts a ratio near 1; the hypothesis is at least 3. With fewer
    than ``minimum`` world steps the ratio is reported as not measurable."""
    steps = sum(len(q) for q in sequences)
    world = sum(1 for q in sequences for s in q if is_world(s))
    after = after_world = 0
    for q in sequences:
        seen = False
        for status in q:
            if seen:
                after += 1
                after_world += is_world(status)
            seen |= is_world(status)
    base = world / steps if steps else None
    conditional = after_world / after if after else None
    measurable = world >= minimum and bool(base) and conditional is not None
    ratio = conditional / base if measurable and base and conditional is not None else None
    return {
        "tool_steps": steps,
        "world_steps": world,
        "base_rate": None if base is None else round(base, 4),
        "steps_after_a_world_step": after,
        "world_steps_after_a_world_step": after_world,
        "conditional_rate": None if conditional is None else round(conditional, 4),
        "ratio": None if ratio is None else round(ratio, 2),
        "measurable": measurable,
        "minimum_world_steps": minimum,
        "clusters": None if ratio is None else ratio >= 3,
    }


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def exact_interval(k: int, n: int, level: float = LEVEL) -> tuple[float, float]:
    """The two-sided Clopper-Pearson interval for ``k`` successes in ``n`` trials, by
    bisection on the binomial tail (stdlib only)."""
    tail = (1 - level) / 2

    def solve(target: float, k_: int) -> float:  # the p at which P(X <= k_) == target
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _binomial_cdf(k_, n, mid) > target else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(1 - tail, k - 1)
    upper = 1.0 if k == n else solve(tail, k)
    return round(lower, 4), round(upper, 4)


def binomial_range(n: int, p: float, level: float = LEVEL) -> tuple[int, int]:
    """The central range of counts a Binomial(n, p) falls in with at least ``level``."""
    tail = (1 - level) / 2
    lower = next(k for k in range(n + 1) if _binomial_cdf(k, n, p) > tail)
    upper = next(k for k in range(n + 1) if _binomial_cdf(k, n, p) >= 1 - tail)
    return lower, upper


def long_analysis(scenario: dict[str, Any], recorded: Counter[str]) -> dict[str, Any]:
    """M16's reading of one arm's runs of the chain (see the module docstring)."""
    per_call = SHORT_RUN_RATE[0] / SHORT_RUN_RATE[1]
    rows = scenario["outcomes"]
    sequences = [[c["status"] for c in r.get("tool_calls", [])] for r in rows]
    firings = [identity_firings(r.get("tool_calls", []), recorded) for r in rows]
    for row, seq, fired in zip(rows, sequences, firings, strict=True):
        row["statuses"] = seq
        row["first_world_step"] = next((i for i, s in enumerate(seq, 1) if is_world(s)), None)
        row["identity_firings"] = fired
    runs = len(rows)
    ended = sum(1 for r in rows if r["cause"] == "unrecorded")
    with_world = sum(1 for r in rows if r["first_world_step"] is not None)
    lengths = [len(q) for q in sequences]
    predicted = 1 - (1 - per_call) ** SHORTEST_CHAIN
    return {
        "served_calls_per_run": lengths,
        "mean_served_calls": round(sum(lengths) / runs, 2) if runs else None,
        "runs_with_a_world_step": with_world,
        "world_endings": {
            "runs": ended,
            "of": runs,
            "interval": list(exact_interval(ended, runs)) if runs else None,
            "independent_share_at_shortest_chain": round(predicted, 4),
            "independent_count": round(predicted * runs, 2),
            "independent_range": list(binomial_range(runs, predicted)) if runs else None,
        },
        "identity_firings": sum(firings),
        "runs_with_an_identity_firing": sum(1 for f in firings if f),
        "clustering": clustering(sequences),
        "survival": survival_curve(sequences, per_call),
    }


def long_runs(log: Path = LONG_RECORDING_LOG) -> int:
    """Runs per arm: :data:`LONG_RUNS`, or :data:`LONG_RUNS_SLOW` when the chain's live
    recording took over :data:`SLOW_RUN_S` (its kept attempt, else its slowest)."""
    if not log.exists():
        return LONG_RUNS
    attempts = json.loads(log.read_text(encoding="utf-8")).get("attempts", [])
    kept = [a["wall_s"] for a in attempts if a.get("passed")]
    walls = kept or [a["wall_s"] for a in attempts]
    return LONG_RUNS_SLOW if walls and max(walls) > SLOW_RUN_S else LONG_RUNS


def run_long(
    runs: int,
    out: Path,
    *,
    client: str = CLIENT,
    suite: Path = QWEN_SUITE,
    environment: dict[str, Any] | None = None,
    sink: RunSink | None = None,
) -> dict[str, Any]:
    """M16: :data:`LONG_ARMS` on the chain, ``runs`` runs each; the report is rewritten
    after every arm so a stopped run keeps the arm it finished."""
    path = suite.parent / next(
        s.cassette for s in load_suite(suite).scenario if s.name == LONG_SCENARIO
    )
    cassette = Cassette.load(path)
    recorded = Counter(e.name for e in cassette.envelopes if e.kind is Kind.TOOL)
    result: dict[str, Any] = {
        "command": "uv run python -m eval.pydantic_ai --long",
        "suite": str(suite.relative_to(ROOT)) if suite.is_relative_to(ROOT) else str(suite),
        "scenario": LONG_SCENARIO,
        "cassette": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "cassette_model": cassette.header.metadata.get("model"),
        "cassette_provider": cassette.header.metadata.get("provider"),
        "recorded_tool_calls": dict(sorted(recorded.items())),
        "client": client,
        "mode": "model-live",
        "identify": "the suite's [defaults.identify] keys",
        "short_run_rate": {"world_steps": SHORT_RUN_RATE[0], "served": SHORT_RUN_RATE[1]},
        "environment": environment if environment is not None else describe(),
        "causes": CAUSE_MEANING,
        "arms": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in LONG_ARMS:
        summary, _ = run_arm(
            dataclasses.replace(arm, runs=runs), client=client, suite=suite, sink=sink
        )
        for s in summary["scenarios"]:
            s["long"] = long_analysis(s, recorded)
            long = s["long"]
            print(
                f"  {arm.label}: {long['runs_with_a_world_step']}/{s['runs']} runs with a world "
                f"step, {long['identity_firings']} identity firings, clustering "
                f"{long['clustering']['ratio']}",
                flush=True,
            )
        result["arms"][arm.label] = summary
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return result


NO_GUARD_ARMS = (
    Arm("fallback", "fallback", prompt_variant="no_guard"),
    Arm("synth", "synth", prompt_variant="no_guard"),
)
"""M17's arms: M15's no-guard prompt on the recorder, under the two policies that keep a
diverged run going (``strict`` is M15's ``7b_no_guard_strict``)."""
NO_GUARD_RUNS = 20
M15_NO_GUARD_ARM = "7b_no_guard_strict"
FARE_SCENARIO = "cheapest_fare_in_euros"
FALLBACK_STANDS, FALLBACK_WITHDRAWN = 6, 15
"""M15's attribution stands if the ``fallback`` fare passes at most 6 of 20, is withdrawn
at 15 or more, and is partial in between."""
STUB_FIGURE_RUNS = 10
"""``synth`` fare runs that state a stub's figure, at least (the hypothesis)."""
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"
_CURRENCY = r"(?:€|\$|EUR|USD|euros?|dollars?)"
CURRENCY_FIGURE = re.compile(
    rf"{_CURRENCY}\s*({_AMOUNT})|(?<![\w.])({_AMOUNT})\s*{_CURRENCY}", re.IGNORECASE
)


def _numbers(value: Any) -> list[float]:
    """Every number in a JSON value, booleans aside."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    return []


def figures(row: dict[str, Any]) -> dict[str, Any]:
    """The money figures one run's answer states (a number next to a currency word or
    sign), the numbers its stubbed steps returned, and whether the answer states one of
    them: a figure the stub, not any recording or real tool, produced."""
    output = row.get("output")
    text = output if isinstance(output, str) else ""
    stated = [float((a or b).replace(",", "")) for a, b in CURRENCY_FIGURE.findall(text)]
    stubbed = [c for c in row.get("tool_calls", []) if c["status"] == "synth"]
    stub_numbers = sorted({n for c in stubbed for n in _numbers(c.get("response"))})
    return {
        "figures": stated,
        "stubbed_steps": len(stubbed),
        "stub_numbers": stub_numbers,
        "states_stub_figure": any(f in stub_numbers for f in stated),
        "world_steps": sum(1 for c in row.get("tool_calls", []) if is_world(c["status"])),
    }


def no_guard_analysis(scenario: dict[str, Any]) -> dict[str, Any]:
    """One arm's scenario: each run gains :func:`figures`; the counts the hypotheses read,
    and every pass with a stubbed step with its answer and tool calls (the manufactured
    world, if any)."""
    rows = scenario["outcomes"]
    for row in rows:
        row.update(figures(row))
    stubbed_passes = [r for r in rows if r["passed"] and r["stubbed_steps"]]
    return {
        "runs_with_a_world_step": sum(1 for r in rows if r["world_steps"]),
        "runs_with_a_stubbed_step": sum(1 for r in rows if r["stubbed_steps"]),
        "runs_stating_a_stub_figure": sum(1 for r in rows if r["states_stub_figure"]),
        "passes_with_a_stubbed_step": len(stubbed_passes),
        "stubbed_passes": [
            {"index": r["index"], "output": r.get("output"), "tool_calls": r.get("tool_calls")}
            for r in stubbed_passes
        ],
        "figures_by_run": [r["figures"] for r in rows],
    }


def m15_no_guard(path: Path = SWAP_OUT) -> dict[str, Any] | None:
    """M15's no-guard ``strict`` arm, per scenario: passes, runs, causes and identity
    endings, from its committed report (``None`` when it is absent)."""
    if not path.exists():
        return None
    arm = json.loads(path.read_text(encoding="utf-8"))["arms"].get(M15_NO_GUARD_ARM)
    if arm is None:
        return None
    return {
        s["name"]: {
            "passes": s["passes"],
            "runs": s["runs"],
            "causes": s["causes"],
            "unrecorded_by_identity": s["unrecorded_by_identity"],
        }
        for s in arm["scenarios"]
    }


def _scenario(arm: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if arm is None:
        return None
    return next((s for s in arm["scenarios"] if s["name"] == name), None)


def no_guard_hypotheses(arms: dict[str, Any]) -> dict[str, Any]:
    """The milestone's live hypotheses, read off the two arms' fare runs (M17 in PLAN.md):
    the ``fallback`` fare passes at most 6 of 20 (M15's attribution stands; 15 or more
    withdraws it; between is partial); ``synth`` passes at most as often as ``fallback``;
    at least 10 ``synth`` fare answers state a stub's figure; no run with a stubbed step
    passes, on either scenario."""
    fallback = _scenario(arms.get("fallback"), FARE_SCENARIO)
    synth = _scenario(arms.get("synth"), FARE_SCENARIO)
    result: dict[str, Any] = {}
    if fallback is not None:
        passes = fallback["passes"]
        verdict = (
            "stands"
            if passes <= FALLBACK_STANDS
            else "withdrawn"
            if passes >= FALLBACK_WITHDRAWN
            else "partial"
        )
        result["fallback_fare"] = {
            "passes": passes,
            "runs": fallback["runs"],
            "m15_attribution": verdict,
            "holds": verdict == "stands",
        }
    if synth is not None:
        stub_figures = synth["no_guard"]["runs_stating_a_stub_figure"]
        result["synth_fare_stub_figures"] = {
            "runs": stub_figures,
            "of": synth["runs"],
            "at_least": STUB_FIGURE_RUNS,
            "holds": stub_figures >= STUB_FIGURE_RUNS,
        }
        if fallback is not None:
            result["synth_at_most_fallback"] = {
                "synth": synth["passes"],
                "fallback": fallback["passes"],
                "holds": synth["passes"] <= fallback["passes"],
            }
        stubbed = sum(
            s["no_guard"]["passes_with_a_stubbed_step"] for s in arms["synth"]["scenarios"]
        )
        result["passes_with_a_stubbed_step"] = {"runs": stubbed, "holds": stubbed == 0}
    return result


def run_no_guard(
    runs: int,
    out: Path,
    *,
    client: str = CLIENT,
    suite: Path = QWEN_SUITE,
    environment: dict[str, Any] | None = None,
    sink: RunSink | None = None,
    m15: Path = SWAP_OUT,
) -> dict[str, Any]:
    """M17: :data:`NO_GUARD_ARMS`, ``runs`` runs per scenario each; the report is rewritten
    after every arm so a stopped run keeps the arm it finished."""
    result: dict[str, Any] = {
        "command": "uv run python -m eval.pydantic_ai --no-guard",
        "suite": str(suite.relative_to(ROOT)) if suite.is_relative_to(ROOT) else str(suite),
        "client": client,
        "mode": "model-live",
        "identify": "the suite's [defaults.identify] keys",
        "recorded_on": MODEL,
        "environment": environment if environment is not None else describe(),
        "causes": CAUSE_MEANING,
        "world_step": "a tool step not served exactly from the recording (approximate, "
        "unrecorded, fallback or synth), as in M16",
        "m15_strict": m15_no_guard(m15),
        "arms": {},
        "hypotheses": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in NO_GUARD_ARMS:
        summary, _ = run_arm(
            dataclasses.replace(arm, runs=runs), client=client, suite=suite, sink=sink
        )
        for s in summary["scenarios"]:
            s["no_guard"] = no_guard_analysis(s)
            ng = s["no_guard"]
            print(
                f"  {s['name']}: {ng['runs_with_a_stubbed_step']} runs with a stubbed step, "
                f"{ng['runs_stating_a_stub_figure']} stating a stub's figure, "
                f"{ng['passes_with_a_stubbed_step']} passes with one",
                flush=True,
            )
        result["arms"][arm.label] = summary
        result["hypotheses"] = no_guard_hypotheses(result["arms"])
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


def _custom_arms(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Arm, ...]:
    """``--arms`` with the live side changed: one :class:`Arm` per policy."""
    model = args.model or MODEL
    arms = tuple(
        Arm(
            f"{model}_{args.prompt_variant}_t{args.temperature}_{policy}",
            policy,
            model=model,
            prompt_variant=args.prompt_variant,
            temperature=args.temperature,
            runs=args.runs,
        )
        for policy in args.arms
    )
    for arm in arms:
        try:
            _ = arm.agent
        except ValueError as exc:
            parser.error(str(exc))
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs", type=int, default=None, help="model-live runs per scenario (default 10)"
    )
    parser.add_argument("--out", type=Path, default=OUT, help="raw flake report")
    parser.add_argument("--offline", action="store_true", help="only the full-mode replays")
    parser.add_argument(
        "--arms",
        type=parse_arms,
        help="M12: run the qwen-recorded scenarios once per policy (e.g. strict,fallback) "
        "instead of the flake",
    )
    parser.add_argument(
        "--arms-out",
        type=Path,
        default=None,
        help=f"raw report of --arms (default {ARMS_OUT.relative_to(ROOT)}; with --model, "
        "--prompt-variant or --temperature, .replay/arms_swapped.json)",
    )
    parser.add_argument(
        "--swap", action="store_true", help="M15: the model, prompt and temperature arms"
    )
    parser.add_argument("--swap-out", type=Path, default=SWAP_OUT, help="raw report of --swap")
    parser.add_argument(
        "--long",
        action="store_true",
        help="M16: the rebooking chain under strict and fallback (--runs: default 20, or "
        "15 when its recording took over 150 s)",
    )
    parser.add_argument("--long-out", type=Path, default=LONG_OUT, help="raw report of --long")
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="M17: the no-guard prompt under fallback and synth (--runs: default 20)",
    )
    parser.add_argument(
        "--no-guard-out", type=Path, default=NO_GUARD_OUT, help="raw report of --no-guard"
    )
    parser.add_argument(
        "--model", default=None, help=f"--arms: the live model (default {MODEL}, the recorder)"
    )
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(get_args(PromptVariant)),
        default="recording",
        help="--arms: the live agent's instructions",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="--arms: the live model's temperature"
    )
    args = parser.parse_args(argv)
    swapped = (args.model, args.prompt_variant, args.temperature) != (None, "recording", None)
    if args.long and (args.arms or args.swap or swapped):
        parser.error("--long runs its own arms; it takes no --arms, --swap or live-side flags")
    if args.no_guard and (args.arms or args.swap or args.long or swapped):
        parser.error(
            "--no-guard runs its own arms; it takes no --arms, --swap, --long or live-side flags"
        )
    if args.runs is not None and args.runs < 1:
        parser.error(f"--runs must be at least 1; got {args.runs}")
    if args.runs is None and not args.long:
        args.runs = NO_GUARD_RUNS if args.no_guard else 10
    if swapped and not args.arms:
        parser.error("--model, --prompt-variant and --temperature change an --arms run")
    arms = SWAP_ARMS if args.swap else _custom_arms(args, parser) if swapped else ()
    if not replay_full():
        return 1
    if args.offline:
        return 0
    needed = sorted({a.model for a in arms}) if arms else [MODEL]
    missing = [m for m in needed if ollama_model(m) is None]
    if missing:
        pulls = "; ".join(f"ollama pull {m}" for m in missing)
        print(
            f"skipped the model-live runs: Ollama at {ollama_url()} is not serving "
            f"{', '.join(missing)} (ollama serve; {pulls}; or set REPLAY_OLLAMA_URL)"
        )
        return 0
    return _live(args, arms, argv)


def _live(args: argparse.Namespace, arms: Sequence[Arm], argv: list[str] | None) -> int:
    """The model-live half :func:`main` asked for, once Ollama is known to serve it."""
    if args.no_guard:
        run_no_guard(args.runs, args.no_guard_out.resolve())
    elif args.long:
        run_long(long_runs() if args.runs is None else args.runs, args.long_out.resolve())
    elif args.swap:
        run_swap(arms, args.swap_out.resolve(), probe=probe_temperature())
    elif arms:
        # Never the committed M12 report: an ad hoc arm writes to .replay/ unless told.
        run_swap(
            arms,
            (args.arms_out or ROOT / ".replay" / "arms_swapped.json").resolve(),
            command=" ".join(["uv run python -m eval.pydantic_ai", *(argv or sys.argv[1:])]),
        )
    elif args.arms:
        run_arms(args.arms, args.runs, (args.arms_out or ARMS_OUT).resolve())
    else:
        return flake(args.runs, args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
