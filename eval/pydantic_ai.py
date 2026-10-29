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

Usage::

    uv run python -m eval.pydantic_ai [--runs 10] [--out eval/results/m9_flake_qwen.json]
    uv run python -m eval.pydantic_ai --arms strict,fallback --runs 20
    uv run python -m eval.pydantic_ai --swap
    uv run python -m eval.pydantic_ai --arms strict --model qwen2.5:3b --prompt-variant no_guard
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from agent_replay.cassette import Cassette
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
SMALL_MODEL = "qwen2.5:3b"
AGENTS: dict[tuple[PromptVariant, float | None], str] = {
    ("recording", None): "examples.pydantic_ai_agent.agent:run",
    ("no_guard", None): "examples.pydantic_ai_agent.agent:run_no_guard",
    ("recording", 0.0): "examples.pydantic_ai_agent.agent:run_temperature_0",
}
"""The suite's ``agent`` for each prompt variant and temperature an arm can ask for."""
ARM_SCENARIOS = ("cheapest_fare_in_euros", "book_earliest_flight")
"""The scenarios whose cassettes qwen2.5:7b recorded (the third serves the fake model's)."""
CAUSES = ("passed", "unrecorded", "approximate", "model")
CAUSE_MEANING = {
    "passed": "every check passed",
    "unrecorded": "a call no unused recording matched (by tool, or by the identify keys) "
    "ended the run under strict: the frozen world's limit",
    "approximate": "a step was served approximately (another request's recording) and a "
    "check then failed: possibly the frozen world's fault",
    "model": "a check failed (or the agent raised) with every served step exact: the "
    "model's own failure",
}


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


def cause(outcome: RunOutcome) -> str:
    """How a run ended, one of :data:`CAUSES`, in that order of precedence: a pass is a
    pass even with an approximate step; an unrecorded call ends a ``strict`` run before
    any check runs; a failure after an approximate step may be the world's; the rest is
    the model's."""
    if outcome.passed:
        return "passed"
    if outcome.error is not None and outcome.error.type == "UnrecordedCallError":
        return "unrecorded"
    if outcome.steps.get("approximate", 0):
        return "approximate"
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
    """One run's tool calls (name, input, status), its answer, the model, temperature and
    guard line its model requests carried (``None`` when the request had none), and how
    many of its model replies asked for more than one tool at once."""
    if result is None:
        return {}
    steps = result.trajectory.model_steps()
    requests = [s.request for s in steps if isinstance(s.request, dict)]
    models = sorted({str(r.get("model")) for r in requests})
    temperatures = sorted({json.dumps(_temperature(r)) for r in requests})
    guard = GUARD.split("\n", 1)[0]
    return {
        "tool_calls": [
            {"name": s.name, "input": s.request, "status": s.status.value}
            for s in result.trajectory.tool_steps()
        ],
        "output": result.output,
        "request_models": models,
        "request_temperatures": [json.loads(t) for t in temperatures],
        "guard_in_system": sorted({guard in json.dumps(r.get("system")) for r in requests}),
        "parallel_tool_replies": sum(1 for s in steps if _tool_uses(s.response) > 1),
    }


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
    parser.add_argument("--runs", type=int, default=10, help="model-live runs per scenario")
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
    if args.swap:
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
