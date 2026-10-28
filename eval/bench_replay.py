"""Recording is cheap, replay is fast: recorder overhead and replay wall-clock (PLAN.md).

Two measurements, both offline.

**Recorder overhead per boundary (D5).** For each kind of boundary the recorder
intercepts, the same call is timed bare (not intercepted) and recorded (inside
a :class:`~agent_replay.RecordingSession`), interleaved one pair at a time, with
the order within a pair alternating so neither side always runs first:

- ``model``: ``messages.create`` on :class:`agent_replay.fake.FakeAnthropic` with
  the retail agent's real first-turn request (policy, tool list, task); bare is
  the fake client itself, recorded is ``agent_replay.wrap`` of it.
- ``tool``: the retail ``get_order_details`` tool; bare is the undecorated
  function, recorded is the ``@tool``.
- ``clock``: ``datetime.now(UTC)`` vs ``agent_replay.now()``.
- ``random``: ``random.Random.random`` vs ``agent_replay.random()``.

The overhead of one pair is ``recorded - bare``; the median and p95 of those
paired differences are the reported overhead, next to the median and p95 of
each side. Pairing cancels drift (frequency scaling, cache state) that two
separate blocks of timings would mix into the difference. Calls run in sessions
of ``--batch`` calls, the size of a real agent run, so per-session bookkeeping
grows as it does in practice; each recorded session is then replayed in
``full`` mode and every served call is timed (``replay``: the cost of looking up
and decoding a recorded response). A warm-up of ``--warmup`` pairs is run and
discarded first; the garbage collector stays enabled (pauses are part of the
cost) and is run between boundaries.

**Appending to a partial cassette (M14).** The same recorded call, in sessions of
``--batch`` calls that keep their envelopes in memory and in sessions that also
append each envelope to ``<out>.partial`` (a temporary directory), the two kinds
of session alternating which goes first; call ``i`` of one is paired with call
``i`` of the other, and the append overhead is the median of those differences.
The cost of ``finish()`` (which, for an appending session, writes the footer and
renames the file) is timed for both. The hypothesis compares the largest
per-boundary append overhead with the median recorded latency of the model
calls in the six live cassettes.

**Replay vs live wall-clock (D1).** For every scenario of
``examples/retail_agent/replay-live.toml`` (six live ``claude-opus-5-5``
recordings), the recorded live duration is the sum of the ``latency_ms`` of
the cassette's envelopes. The agent calls the model and its tools one at a
time, so the sum is the time spent inside boundary calls; the cassette stores no
end-of-run timestamp, so the agent's own code between calls is not in it, and
the sum is a lower bound on the live run's wall-clock. The replay is the code
path ``replay ci`` runs per scenario (:func:`agent_replay.ci.run_scenario`:
load the cassette and the agent, replay in ``full`` mode), timed without checks
and with the suite's checks (the judge reads its committed verdict cache;
``REPLAY_JUDGE_OFFLINE=1``), ``--repeats`` times after one warm-up run; the
median is reported, and speedup = recorded live duration / median replay time.

Everything except the timings and the environment block is deterministic (fixed
seeds, fixed requests, committed cassettes). Timings depend on the machine, so
the JSON is not byte-stable and CI does not diff it.

Usage::

    uv run python eval/bench_replay.py --out eval/results/bench_replay.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random as _random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue

import agent_replay
from agent_replay._json import canonical_json, to_jsonable
from agent_replay._logging import configure_logging
from agent_replay.cassette import Cassette, Kind
from agent_replay.ci import Overrides, load_baselines, load_suite, run_scenario
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.session import RecordingSession, ReplaySession
from examples.retail_agent import tools as retail_tools
from examples.retail_agent.agent import MODEL, SYSTEM, TOOLS
from examples.retail_agent.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
LIVE_SUITE = ROOT / "examples" / "retail_agent" / "replay-live.toml"
COMMAND = "uv run python eval/bench_replay.py --out eval/results/bench_replay.json"
BOUNDARIES = ("model", "tool", "clock", "random")
SEED = 0

# The retail agent's first turn on cancel_pending_laptop, as it sends it.
TASK = TASKS["cancel_pending_laptop"].instruction
ORDER_ID = "#W2417020"


class HarnessError(RuntimeError):
    """The benchmark's own setup is broken (a scenario failed or did not run)."""


def _model_request() -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM,
        "tools": [t.to_anthropic() for t in TOOLS.values()],
        "messages": [{"role": "user", "content": TASK}],
        "output_config": {"effort": "low"},
        "cache_control": {"type": "ephemeral"},
    }


def _reply(_: FakeRequest) -> Reply:
    return Reply(
        tool_calls=[
            ToolCall(
                "find_user_id_by_name_zip",
                {"first_name": "Emma", "last_name": "Smith", "zip": "10192"},
            )
        ]
    )


@dataclass(frozen=True)
class Boundary:
    """One boundary kind: the bare call, the intercepted call, and what it sends."""

    name: str
    description: str
    bare: Callable[[], object]
    recorded: Callable[[], object]
    request_bytes: int
    """Length of the recorded request's canonical JSON (what is fingerprinted)."""


def boundaries() -> list[Boundary]:
    request = _model_request()
    bare_client = FakeAnthropic(_reply)
    wrapped_client = agent_replay.wrap(FakeAnthropic(_reply))
    order_tool = TOOLS["get_order_details"]
    bare_rng = _random.Random(SEED)  # noqa: S311 - a benchmark's draws, not cryptography
    agent_replay.capture.seed(SEED)
    model_json = to_jsonable(request)
    return [
        Boundary(
            "model",
            "messages.create on FakeAnthropic with the retail agent's first-turn request; "
            "bare: the fake client, recorded: agent_replay.wrap(fake client)",
            lambda: bare_client.messages.create(**request),
            lambda: wrapped_client.messages.create(**request),
            len(canonical_json(model_json)),
        ),
        Boundary(
            "tool",
            "retail get_order_details; bare: the undecorated function, recorded: the @tool",
            lambda: order_tool.fn(ORDER_ID),
            lambda: order_tool(ORDER_ID),
            len(canonical_json({"order_id": ORDER_ID})),
        ),
        Boundary(
            "clock",
            "bare: datetime.now(UTC), recorded: agent_replay.now()",
            lambda: datetime.now(UTC),
            agent_replay.now,
            len(canonical_json({})),
        ),
        Boundary(
            "random",
            "bare: random.Random.random, recorded: agent_replay.random()",
            bare_rng.random,
            agent_replay.random,
            len(canonical_json({})),
        ),
    ]


# --- statistics -------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100), linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _us(ns: float) -> float:
    return round(ns / 1000, 3)


def _summary_us(values_ns: Sequence[float]) -> dict[str, JsonValue]:
    return {
        "median_us": _us(statistics.median(values_ns)),
        "p95_us": _us(percentile(values_ns, 95)),
    }


# --- recorder overhead ------------------------------------------------------------


def _time_ns(fn: Callable[[], object]) -> int:
    started = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - started


@dataclass(frozen=True)
class Samples:
    bare: list[int]
    recorded: list[int]
    served: list[int]


def sample(boundary: Boundary, *, iterations: int, batch: int) -> Samples:
    """``iterations`` interleaved (bare, recorded) pairs, then each recorded session replayed."""
    bare: list[int] = []
    recorded: list[int] = []
    served: list[int] = []
    done = 0
    while done < iterations:
        size = min(batch, iterations - done)
        session = RecordingSession(agent="eval.bench_replay", task=boundary.name)
        with session:
            for i in range(done, done + size):
                if i % 2 == 0:
                    bare.append(_time_ns(boundary.bare))
                    recorded.append(_time_ns(boundary.recorded))
                else:
                    recorded.append(_time_ns(boundary.recorded))
                    bare.append(_time_ns(boundary.bare))
        cassette = session.finish()
        if len(cassette.envelopes) != size:
            raise HarnessError(
                f"{boundary.name}: recorded {len(cassette.envelopes)} envelopes for {size} calls"
            )
        with ReplaySession(cassette, mode="full") as replay:
            for _ in range(size):
                served.append(_time_ns(boundary.recorded))
        if replay.fatal is not None or replay.unconsumed():
            raise HarnessError(f"{boundary.name}: the replay did not serve every recorded call")
        done += size
    return Samples(bare, recorded, served)


@dataclass(frozen=True)
class AppendSamples:
    in_memory: list[int]
    appended: list[int]
    finish_in_memory: list[int]
    finish_appended: list[int]


def sample_append(
    boundary: Boundary, *, iterations: int, batch: int, out_dir: Path
) -> AppendSamples:
    """The recorded call in in-memory sessions and in sessions appending to a partial."""
    in_memory: list[int] = []
    appended: list[int] = []
    finish_in_memory: list[int] = []
    finish_appended: list[int] = []
    done = 0
    sessions = 0
    while done < iterations:
        size = min(batch, iterations - done)
        kinds = (False, True) if sessions % 2 == 0 else (True, False)
        for appending in kinds:
            out = out_dir / f"{boundary.name}-{sessions}.jsonl" if appending else None
            session = RecordingSession(agent="eval.bench_replay", task=boundary.name, out=out)
            times = appended if appending else in_memory
            with session:
                for _ in range(size):
                    times.append(_time_ns(boundary.recorded))
            (finish_appended if appending else finish_in_memory).append(_time_ns(session.finish))
            if out is not None:
                if len(Cassette.load(out).envelopes) != size:
                    raise HarnessError(f"{boundary.name}: the appended cassette lost envelopes")
                out.unlink()
        done += size
        sessions += 1
    return AppendSamples(in_memory, appended, finish_in_memory, finish_appended)


def measure_append(
    boundary: Boundary, *, iterations: int, warmup: int, batch: int
) -> dict[str, JsonValue]:
    with tempfile.TemporaryDirectory() as tmp:
        if warmup:
            sample_append(boundary, iterations=warmup, batch=batch, out_dir=Path(tmp))
        gc.collect()
        s = sample_append(boundary, iterations=iterations, batch=batch, out_dir=Path(tmp))
    overhead = [a - m for a, m in zip(s.appended, s.in_memory, strict=True)]
    return {
        "in_memory": _summary_us(s.in_memory),
        "appended": _summary_us(s.appended),
        "overhead": _summary_us(overhead),
        "finish_in_memory": _summary_us(s.finish_in_memory),
        "finish_appended": _summary_us(s.finish_appended),
    }


def measure_session(*, iterations: int, warmup: int) -> dict[str, JsonValue]:
    """The fixed cost of one recording session with no calls: create, enter, exit, finish."""

    def one() -> None:
        session = RecordingSession(agent="eval.bench_replay", task="session")
        with session:
            pass
        session.finish()

    for _ in range(warmup):
        one()
    gc.collect()
    return {"iterations": iterations, **_summary_us([_time_ns(one) for _ in range(iterations)])}


def measure_boundary(
    boundary: Boundary, *, iterations: int, warmup: int, batch: int
) -> dict[str, JsonValue]:
    if warmup:
        sample(boundary, iterations=warmup, batch=batch)
    gc.collect()
    s = sample(boundary, iterations=iterations, batch=batch)
    overhead = [r - b for r, b in zip(s.recorded, s.bare, strict=True)]
    return {
        "boundary": boundary.name,
        "description": boundary.description,
        "request_bytes": boundary.request_bytes,
        "iterations": iterations,
        "bare": _summary_us(s.bare),
        "recorded": _summary_us(s.recorded),
        "overhead": _summary_us(overhead),
        "replay_served": _summary_us(s.served),
        "append": measure_append(boundary, iterations=iterations, warmup=warmup, batch=batch),
    }


# --- replay vs live ---------------------------------------------------------------


@dataclass(frozen=True)
class RecordedDuration:
    """What a cassette says its boundary calls took, in milliseconds."""

    envelopes: int
    model_calls: int
    tool_calls: int
    model_ms: float
    tool_ms: float
    total_ms: float


def recorded_duration(cassette: Cassette) -> RecordedDuration:
    """Sums of the recorded ``latency_ms`` (every envelope, and per kind)."""
    model = [e.latency_ms for e in cassette.envelopes if e.kind is Kind.MODEL]
    tool = [e.latency_ms for e in cassette.envelopes if e.kind is Kind.TOOL]
    return RecordedDuration(
        envelopes=len(cassette.envelopes),
        model_calls=len(model),
        tool_calls=len(tool),
        model_ms=round(math.fsum(model), 3),
        tool_ms=round(math.fsum(tool), 3),
        total_ms=round(math.fsum(e.latency_ms for e in cassette.envelopes), 3),
    )


def recorded_model_latencies(suite_path: Path) -> list[float]:
    """``latency_ms`` of every model envelope in the suite's cassettes."""
    suite = load_suite(suite_path)
    return [
        e.latency_ms
        for scenario in suite.scenario
        for e in Cassette.load(suite_path.parent / scenario.cassette).envelopes
        if e.kind is Kind.MODEL
    ]


def measure_scenarios(suite_path: Path, *, repeats: int) -> list[dict[str, JsonValue]]:
    suite = load_suite(suite_path)
    overrides = Overrides()
    base_dir = suite_path.parent
    baselines = load_baselines(suite, overrides, base_dir)
    rows: list[dict[str, JsonValue]] = []
    for scenario in suite.scenario:
        cassette = Cassette.load(base_dir / scenario.cassette)
        recorded = recorded_duration(cassette)
        timings: dict[str, list[int]] = {"replay": [], "replay_and_checks": []}
        for i in range(repeats + 1):  # the first run of each variant is a warm-up
            for variant, with_checks in (("replay", False), ("replay_and_checks", True)):
                retail_tools.reset_db()
                started = time.perf_counter_ns()
                report, _ = run_scenario(
                    scenario,
                    suite.defaults,
                    overrides,
                    base_dir,
                    time.perf_counter,
                    baselines=baselines,
                    with_checks=with_checks,
                )
                elapsed = time.perf_counter_ns() - started
                if not report.passed:
                    problem = report.error.message if report.error else "a check failed"
                    raise HarnessError(f"{scenario.name} ({variant}) did not pass: {problem}")
                if i > 0:
                    timings[variant].append(elapsed)
        retail_tools.reset_db()
        row: dict[str, JsonValue] = {
            "scenario": scenario.name,
            "cassette": Path(scenario.cassette).as_posix(),
            "envelopes": recorded.envelopes,
            "model_calls": recorded.model_calls,
            "tool_calls": recorded.tool_calls,
            "recorded_model_ms": recorded.model_ms,
            "recorded_tool_ms": recorded.tool_ms,
            "recorded_live_ms": recorded.total_ms,
        }
        for variant, values in timings.items():
            median_ms = statistics.median(values) / 1e6
            row[f"{variant}_median_ms"] = round(median_ms, 3)
            row[f"{variant}_p95_ms"] = round(percentile(values, 95) / 1e6, 3)
            row[f"speedup_{variant}"] = round(recorded.total_ms / median_ms, 1)
        rows.append(row)
    return rows


# --- report -----------------------------------------------------------------------


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def environment() -> dict[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }


def _num(row: JsonValue, key: str) -> float:
    assert isinstance(row, dict)
    value = row[key]
    assert isinstance(value, int | float)
    return float(value)


def _stat(row: JsonValue, group: str, key: str) -> float:
    assert isinstance(row, dict)
    return _num(row[group], key)


def run(
    *, iterations: int = 2000, warmup: int = 200, batch: int = 20, repeats: int = 20
) -> dict[str, JsonValue]:
    """Both measurements, as JSON data."""
    boundary_rows: list[JsonValue] = []
    for boundary in boundaries():
        boundary_rows.append(
            measure_boundary(boundary, iterations=iterations, warmup=warmup, batch=batch)
        )
        gc.collect()
    sessions = max(1, iterations // batch)
    session_row = measure_session(iterations=sessions, warmup=max(1, warmup // batch))
    scenario_rows: list[JsonValue] = list(measure_scenarios(LIVE_SUITE, repeats=repeats))
    live_ms = math.fsum(_num(r, "recorded_live_ms") for r in scenario_rows)
    replay_ms = math.fsum(_num(r, "replay_median_ms") for r in scenario_rows)
    checked_ms = math.fsum(_num(r, "replay_and_checks_median_ms") for r in scenario_rows)
    speedups = [_num(r, "speedup_replay") for r in scenario_rows]
    model_latencies = recorded_model_latencies(LIVE_SUITE)
    model_median_ms = statistics.median(model_latencies)
    appends: dict[str, JsonValue] = {
        str(_field(r, "boundary")): _num(_field(_field(r, "append"), "overhead"), "median_us")
        for r in boundary_rows
    }
    worst_append_us = max(_num(appends, k) for k in appends)
    return {
        "method": {
            "overhead": "per boundary, interleaved pairs of the same call bare and recorded "
            "(order alternating within a pair), timed with time.perf_counter_ns; overhead = "
            "median and p95 of the paired differences (recorded - bare), next to the median "
            "and p95 of each side; calls run in sessions of `batch` calls after `warmup` "
            "discarded pairs; GC enabled",
            "session": "the fixed cost of one recording session with no calls (create, enter, "
            "exit, finish), median and p95 over iterations / batch sessions",
            "replay_served": "each recorded session replayed in full mode; the time of every "
            "served call (lookup, fingerprint, decode), median and p95",
            "append": "the recorded call in sessions of `batch` calls kept in memory and in "
            "sessions also appending each envelope to <out>.partial, alternating which goes "
            "first; overhead = median of (appended - in memory) paired by call index; finish() "
            "timed for both; the worst boundary's overhead over the median latency_ms of the "
            "live suite's model envelopes",
            "recorded_live_ms": "sum of the cassette's envelope latency_ms (the agent calls "
            "one boundary at a time); excludes the agent's own code between calls, so it is a "
            "lower bound on the live run's wall-clock",
            "replay_ms": "wall-clock of agent_replay.ci.run_scenario (load cassette and agent, "
            "full replay; with or without the suite's checks, judge from its committed cache), "
            "median of `repeats` runs after one warm-up; speedup = recorded_live_ms / median",
        },
        "command": COMMAND,
        "parameters": {
            "iterations": iterations,
            "warmup": warmup,
            "batch": batch,
            "repeats": repeats,
            "seed": SEED,
        },
        "datasets": {
            "overhead": "D5: synthetic calls through the fake model and the retail tools",
            "replay": LIVE_SUITE.relative_to(ROOT).as_posix()
            + ": six live claude-opus-5-5 recordings (D1-like)",
        },
        "environment": environment(),
        "boundaries": boundary_rows,
        "session": session_row,
        "scenarios": scenario_rows,
        "summary": {
            "overhead_median_us": {
                str(_field(r, "boundary")): _stat(r, "overhead", "median_us") for r in boundary_rows
            },
            "recorded_live_ms_total": round(live_ms, 3),
            "replay_median_ms_total": round(replay_ms, 3),
            "replay_and_checks_median_ms_total": round(checked_ms, 3),
            "speedup_suite": round(live_ms / replay_ms, 1),
            "speedup_suite_with_checks": round(live_ms / checked_ms, 1),
            "speedup_min": min(speedups),
            "speedup_max": max(speedups),
            "append_overhead_median_us": appends,
            "recorded_model_calls": len(model_latencies),
            "recorded_model_latency_median_ms": round(model_median_ms, 3),
            "append_worst_share_of_model_latency": round(
                worst_append_us / (model_median_ms * 1000), 8
            ),
        },
    }


def _worst(appends: JsonValue) -> float:
    assert isinstance(appends, dict)
    return max(_num(appends, k) for k in appends)


def _field(row: JsonValue, key: str) -> JsonValue:
    assert isinstance(row, dict)
    return row[key]


def to_text(report: dict[str, JsonValue]) -> str:
    params = report["parameters"]
    assert isinstance(params, dict)
    lines = [
        f"Recorder overhead per boundary ({params['iterations']} paired calls, "
        f"{params['warmup']} warm-up, sessions of {params['batch']}; microseconds)",
        "",
        "| boundary | request bytes | bare median | bare p95 | recorded median | recorded p95 "
        "| overhead median | overhead p95 | replay served median | replay served p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    boundaries_ = report["boundaries"]
    assert isinstance(boundaries_, list)
    for r in boundaries_:
        cells = [
            f"{_stat(r, group, key):.2f}"
            for group in ("bare", "recorded", "overhead", "replay_served")
            for key in ("median_us", "p95_us")
        ]
        row = f"| {_field(r, 'boundary')} | {_field(r, 'request_bytes')} | "
        lines.append(row + " | ".join(cells) + " |")
    lines += [
        "",
        "Appending each envelope to a partial cassette (microseconds)",
        "",
        "| boundary | in memory median | appending median | append overhead median "
        "| append overhead p95 | finish in memory | finish appending |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in boundaries_:
        a = _field(r, "append")
        cells = [
            f"{_stat(a, group, key):.2f}"
            for group, key in (
                ("in_memory", "median_us"),
                ("appended", "median_us"),
                ("overhead", "median_us"),
                ("overhead", "p95_us"),
                ("finish_in_memory", "median_us"),
                ("finish_appended", "median_us"),
            )
        ]
        lines.append(f"| {_field(r, 'boundary')} | " + " | ".join(cells) + " |")
    session = report["session"]
    lines += [
        "",
        f"- one recording session with no calls (create, enter, exit, finish; "
        f"{_field(session, 'iterations')} sessions): median {_num(session, 'median_us'):.2f}, "
        f"p95 {_num(session, 'p95_us'):.2f}",
    ]
    lines += [
        "",
        f"Full replay vs recorded live duration ({params['repeats']} runs per scenario after "
        "one warm-up; milliseconds)",
        "",
        "| scenario | envelopes | recorded live | replay median | replay p95 | speedup "
        "| replay+checks median | speedup with checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    for r in scenarios:
        lines.append(
            f"| {_field(r, 'scenario')} | {_field(r, 'envelopes')} "
            f"| {_num(r, 'recorded_live_ms'):.1f} | {_num(r, 'replay_median_ms'):.2f} "
            f"| {_num(r, 'replay_p95_ms'):.2f} | {_num(r, 'speedup_replay'):.0f}x "
            f"| {_num(r, 'replay_and_checks_median_ms'):.2f} "
            f"| {_num(r, 'speedup_replay_and_checks'):.0f}x |"
        )
    summary = report["summary"]
    assert isinstance(summary, dict)
    lines += [
        "",
        f"- suite: recorded live {_num(summary, 'recorded_live_ms_total'):.1f} ms, full replay "
        f"{_num(summary, 'replay_median_ms_total'):.2f} ms "
        f"({_num(summary, 'speedup_suite'):.0f}x), with checks "
        f"{_num(summary, 'replay_and_checks_median_ms_total'):.2f} ms "
        f"({_num(summary, 'speedup_suite_with_checks'):.0f}x); per scenario "
        f"{_num(summary, 'speedup_min'):.0f}x to {_num(summary, 'speedup_max'):.0f}x",
        f"- append: worst boundary {_worst(summary['append_overhead_median_us']):.2f} us "
        f"per envelope against a median recorded model latency of "
        f"{_num(summary, 'recorded_model_latency_median_ms'):.1f} ms "
        f"({_num(summary, 'recorded_model_calls'):.0f} calls): "
        f"{100 * _num(summary, 'append_worst_share_of_model_latency'):.4f}%",
    ]
    env = report["environment"]
    assert isinstance(env, dict)
    lines.append(
        f"- environment: Python {env['python']} ({env['implementation']}), {env['system']} "
        f"{env['machine']}, {env['cpu_count']} CPUs, {env['cpu_model']}"
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--iterations", type=int, default=2000, help="paired calls per boundary")
    parser.add_argument("--warmup", type=int, default=200, help="discarded pairs per boundary")
    parser.add_argument("--batch", type=int, default=20, help="calls per recording session")
    parser.add_argument("--repeats", type=int, default=20, help="replays per scenario")
    parser.add_argument("--out", type=Path, help="also write the results as JSON here")
    args = parser.parse_args(argv)
    for name in ("iterations", "batch", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    # The judge reads its committed cache only; a miss is a harness bug, never a network call.
    os.environ["REPLAY_JUDGE_OFFLINE"] = "1"
    os.environ.setdefault("REPLAY_LOG_LEVEL", "ERROR")
    configure_logging()
    try:
        report = run(
            iterations=args.iterations, warmup=args.warmup, batch=args.batch, repeats=args.repeats
        )
    except HarnessError as exc:
        print(f"error: harness bug: {exc}", file=sys.stderr)
        return 2
    print(to_text(report), end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
