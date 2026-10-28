"""Cassette bytes at scale (PLAN.md, M13): how big long runs get, and what recovers it.

**Synthetic runs.** A fake-model agent (:func:`run`) calls one tool a turn and
answers on its last turn, so a run of ``n`` turns is ``n`` model calls and
``n - 1`` tool calls. Every model request carries the whole conversation, as
the Messages API requires, so each envelope copies all the earlier ones and
cassette bytes grow with ``n**2``. The system prompt is the retail agent's
length (:data:`SYSTEM_CHARS`) and the tool results about 250 bytes; both are
seeded, non-repeating text (:func:`words`: a seeded vocabulary of 4,096
pseudo-words drawn uniformly, about 12 bits a word), because repeated filler
flatters every compressor. Runs of :data:`TURNS` turns are recorded with a
fixed clock and timer and a fixed cassette id, so their bytes are byte-stable.

Per run: the JSONL bytes, the largest line (one envelope), what the last turn
added, the model envelopes' share; the same bytes through gzip (``zlib``,
DEFLATE, a 32 KB window), bz2 (900 KB blocks) and xz (``lzma`` preset 6, an
8 MB dictionary), each as bytes and ratio; and the bytes of a shared-prefix
encoding (:func:`prefix_encode`: each model request stored as what it appends
to the previous model request), raw and through each codec. The exponent is
the least-squares slope of log bytes on log turns, with the slope between each
pair of neighbouring sizes beside it.

**Real recordings.** The same columns for every committed real recording (the
six live ``claude-opus-5-5`` retail cassettes and the two qwen2.5:7b
pydantic-ai ones), with the first model request's bytes and how much each
model request grew over the one before it (the conversation's growth per turn,
which sets the quadratic term), so the synthetic workload is checked against
real traces.

**The decision** (fixed in PLAN.md before the run): transparent compressed
cassettes with the best codec that recovers at least 10x at 100 turns; a
new cassette schema with the prefix encoding (PLAN.md's "v2"; cassettes are
already at schema 2, so it would be v3) only if it beats that codec by more than
3x (the prefix encoding under the same codec against the codec alone, and the
raw prefix encoding against the codec).

**Timings** (machine-dependent, like ``bench-replay``): recording, ``Cassette``
load from JSONL and from each compressed form (decompress, then parse), and
``full`` replay per size; and the run store (SQLite) on 100 replays of the
20-turn run and 10 of the 100-turn run: ``save_run`` time, database bytes, the
share of them that is step request and response JSON, and the ``get_run``
payload (the ``/api/runs/{id}`` body) and time.

Two outputs. ``--sizes`` (committed, ``eval/results/bench_scale_sizes.json``)
holds every size and the decision; raw and prefix-encoded sizes are byte-stable
and ``--check`` (run by the test suite) recomputes them against the committed
file; compressed sizes are recorded with the zlib, bz2 and liblzma versions,
because a different build can differ by a few bytes. ``--out`` holds the
timings (``.replay/`` by default; ``make bench-scale-record`` rewrites the
committed copy).

Usage::

    uv run python -m eval.bench_scale [--out .replay/bench_scale.json]
    uv run python -m eval.bench_scale --check
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import itertools
import json
import lzma
import math
import random
import statistics
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock
from pydantic import JsonValue

from agent_replay import tool
from agent_replay._logging import configure_logging
from agent_replay.cassette import Cassette, Kind
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.runner import replay
from agent_replay.session import RecordingSession
from agent_replay.store import Store
from agent_replay.trajectory import StepStatus
from eval.bench_replay import environment
from examples.retail_agent.agent import SYSTEM as RETAIL_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
SIZES = ROOT / "eval" / "results" / "bench_scale_sizes.json"
OUT = ROOT / ".replay" / "bench_scale.json"
COMMAND = "make bench-scale"
TURNS = (5, 20, 50, 100, 200)
DECISION_TURNS = 100
MIN_RATIO = 10
PREFIX_MARGIN = 3
STORE_LOADS = ((20, 100), (100, 10))
"""(turns, runs) for the run-store measurement."""
SEED = 13
SYSTEM_CHARS = len(RETAIL_SYSTEM)
RESULT_CHARS = 220
REPLY_CHARS = 80
VOCABULARY = 4096
FIXED_TIME = datetime(2026, 10, 28, tzinfo=UTC)
REAL = (
    *sorted((ROOT / "examples" / "retail_agent" / "cassettes" / "live").glob("*.jsonl")),
    *sorted((ROOT / "examples" / "pydantic_ai_agent" / "cassettes" / "qwen").glob("*.jsonl")),
)
TASK = "Read the records one at a time until told to stop, then summarise them."


# --- the workload -------------------------------------------------------------------------


def _vocabulary() -> list[str]:
    rng = random.Random(f"vocabulary-{SEED}")  # noqa: S311 - benchmark text, not secrets
    letters = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choices(letters, k=rng.randint(2, 9))) for _ in range(VOCABULARY)]


_WORDS = _vocabulary()


def words(key: str, chars: int) -> str:
    """Seeded text of about ``chars`` characters; a different ``key`` never repeats it."""
    rng = random.Random(f"{SEED}-{key}")  # noqa: S311 - benchmark text, not secrets
    out: list[str] = []
    length = 0
    while length < chars:
        word = rng.choice(_WORDS)
        out.append(word)
        length += len(word) + 1
    return " ".join(out)[:chars]


SYSTEM = words("system", SYSTEM_CHARS)


@tool
def read_record(record_id: int) -> dict[str, JsonValue]:
    """Read one record by id."""
    return {"record_id": record_id, "notes": words(f"record-{record_id}", RESULT_CHARS)}


def run(client: Anthropic, task: str) -> str:
    """A plain tool-use loop: call the model, run what it asks for, until it answers."""
    messages: list[MessageParam] = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=4096,
            system=SYSTEM,
            tools=[read_record.to_anthropic()],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if not uses:
            return " ".join(b.text for b in response.content if b.type == "text")
        messages.append(
            {
                "role": "user",
                "content": [
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=b.id,
                        content=json.dumps(read_record(**cast(dict[str, int], b.input))),
                    )
                    for b in uses
                ],
            }
        )


def fake_model(turns: int) -> FakeAnthropic:
    """Reads record 0, 1, ... one a turn, and answers on turn ``turns``."""

    def policy(request: FakeRequest) -> Reply:
        done = len(request.tool_results())
        text = words(f"reply-{done}", REPLY_CHARS)
        if done + 1 >= turns:
            return Reply(text=text)
        return Reply(text=text, tool_calls=[ToolCall("read_record", {"record_id": done})])

    return FakeAnthropic(policy)


def _ticks() -> Callable[[], float]:
    """A timer that advances one millisecond a call, so every latency_ms is the same."""
    state = {"t": 0.0}

    def timer() -> float:
        state["t"] += 0.001
        return state["t"]

    return timer


def record_run(turns: int) -> Cassette:
    """Record a ``turns``-turn run with the fake model, byte-stable."""
    from agent_replay import wrap  # noqa: PLC0415 - keeps the module's imports light

    session = RecordingSession(
        agent="eval.bench_scale:run", task=TASK, clock=lambda: FIXED_TIME, timer=_ticks()
    )
    client = wrap(fake_model(turns))
    with session:
        output = run(cast(Anthropic, client), TASK)
    cassette = session.finish(output=output)
    # The recorder's versions vary by machine (Python 3.12.16 against 3.14.4 is a byte), so
    # the benchmark's cassettes leave them out and their sizes do not depend on the runner.
    header = cassette.header.model_copy(
        update={"cassette_id": f"bench-scale-{turns:03d}", "recorder": {}}
    )
    cassette = cassette.model_copy(update={"header": header})
    model_calls = sum(1 for e in cassette.envelopes if e.kind is Kind.MODEL)
    if model_calls != turns or cassette.footer.status != "ok":
        raise RuntimeError(f"the {turns}-turn run made {model_calls} model calls")
    return cassette


# --- the prefix-encoding prototype --------------------------------------------------------


def prefix_encode(text: str) -> str:
    """Store each model request as what it appends to the previous model request.

    A prototype, not a format: a model envelope whose request has the same fields
    as the previous model envelope's and whose ``messages`` start with all of that
    request's messages keeps only ``{"$prefix": <seq>, "$append": [new messages],
    "$set": {changed fields}}``. Everything else is written as is.
    """
    out: list[str] = []
    previous: tuple[int, dict[str, Any]] | None = None
    for line in text.split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "envelope" and obj["kind"] == "model":
            request = obj["request"]
            if previous is not None and _extends(previous[1], request):
                seq, before = previous
                obj["request"] = {
                    "$prefix": seq,
                    "$append": request["messages"][len(before["messages"]) :],
                    "$set": {
                        k: v for k, v in request.items() if k != "messages" and before.get(k) != v
                    },
                }
            previous = (obj["seq"], request)
        out.append(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(out) + "\n"


def _extends(before: dict[str, Any], after: dict[str, Any]) -> bool:
    old, new = before.get("messages"), after.get("messages")
    if not isinstance(old, list) or not isinstance(new, list) or len(new) < len(old):
        return False
    return new[: len(old)] == old and set(before) <= set(after)


def prefix_decode(text: str) -> str:
    """Invert :func:`prefix_encode` (proves the encoding loses nothing)."""
    out: list[str] = []
    requests: dict[int, dict[str, Any]] = {}
    for line in text.split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "envelope" and obj["kind"] == "model":
            request = obj["request"]
            if isinstance(request, dict) and "$prefix" in request:
                base = requests[request["$prefix"]]
                rebuilt = {k: v for k, v in base.items() if k != "messages"}
                rebuilt.update(request["$set"])
                rebuilt["messages"] = [*base["messages"], *request["$append"]]
                obj["request"] = {k: rebuilt[k] for k in _field_order(base, rebuilt)}
            requests[obj["seq"]] = obj["request"]
        out.append(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(out) + "\n"


def _field_order(base: dict[str, Any], rebuilt: dict[str, Any]) -> list[str]:
    return [*base, *(k for k in rebuilt if k not in base)]


def _compact(value: Any) -> bytes:
    """``value`` as the cassette writes JSON: no spaces, UTF-8."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalised(text: str) -> str:
    """The JSONL re-serialised as :func:`prefix_encode` writes it, for the round trip."""
    lines = text.split("\n")
    out = (json.dumps(json.loads(x), separators=(",", ":"), ensure_ascii=False) for x in lines if x)
    return "\n".join(out) + "\n"


# --- measuring ----------------------------------------------------------------------------

CODECS: dict[str, Callable[[bytes], bytes]] = {
    "gzip": lambda b: gzip.compress(b, compresslevel=9, mtime=0),
    "bz2": lambda b: bz2.compress(b, compresslevel=9),
    "xz": lambda b: lzma.compress(b, preset=6),
}
DECOMPRESS: dict[str, Callable[[bytes], bytes]] = {
    "gzip": gzip.decompress,
    "bz2": bz2.decompress,
    "xz": lzma.decompress,
}


def codec_versions() -> dict[str, Any]:
    return {
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "bz2": "libbz2 (Python " + sys.version.split()[0] + ")",
        "liblzma": _liblzma_version(),
        "gzip_level": 9,
        "bz2_level": 9,
        "xz_preset": 6,
    }


def _liblzma_version() -> str:
    import ctypes  # noqa: PLC0415
    import ctypes.util  # noqa: PLC0415

    name = ctypes.util.find_library("lzma")
    if name is None:
        return "unknown"
    try:
        lib = ctypes.CDLL(name)
        lib.lzma_version_string.restype = ctypes.c_char_p
        version = lib.lzma_version_string()
    except (OSError, AttributeError):
        return "unknown"
    return version.decode() if isinstance(version, bytes) else "unknown"


def sizes(text: str, *, compress: bool = True) -> dict[str, Any]:
    """Every size of one cassette's JSONL: raw, per codec, prefix-encoded.

    ``compress=False`` leaves the codecs out (the byte-stable part, for ``--check``).
    """
    raw = text.encode("utf-8")
    lines = [line for line in text.split("\n") if line]
    envelopes = [json.loads(x) for x in lines[1:-1]]
    model_bytes = sum(
        len(x.encode()) for x, e in zip(lines[1:-1], envelopes, strict=True) if e["kind"] == "model"
    )
    prefix = prefix_encode(text)
    if prefix_decode(prefix) != _normalised(text):
        raise RuntimeError("the prefix encoding did not round-trip")
    prefix_raw = prefix.encode("utf-8")
    requests = [len(_compact(e["request"])) for e in envelopes if e["kind"] == "model"]
    growth = [b - a for a, b in itertools.pairwise(requests)]
    codecs: dict[str, Any] = {}
    for name, codec in CODECS.items() if compress else ():
        packed = len(codec(raw))
        prefixed = len(codec(prefix_raw))
        codecs[name] = {
            "bytes": packed,
            "ratio": _round(len(raw) / packed),
            "prefix_bytes": prefixed,
            "prefix_ratio": _round(len(raw) / prefixed),
        }
    return {
        "turns": len(requests),
        "envelopes": len(envelopes),
        "raw_bytes": len(raw),
        "bytes_per_turn": round(len(raw) / max(1, len(requests))),
        "largest_line_bytes": max(len(x.encode()) for x in lines),
        "model_share": _round(model_bytes / len(raw), 3),
        "first_request_bytes": requests[0] if requests else 0,
        "growth_per_turn_bytes": round(statistics.fmean(growth)) if growth else 0,
        "requests_at_100_turns_bytes": _projected(requests, DECISION_TURNS),
        "prefix_raw_bytes": len(prefix_raw),
        "prefix_raw_ratio": _round(len(raw) / len(prefix_raw)),
        "codecs": codecs,
    }


def _projected(requests: Sequence[int], turns: int) -> int:
    """Model request bytes alone at ``turns`` turns, growing as this run's requests grew."""
    if len(requests) < 2:
        return 0
    growth = (requests[-1] - requests[0]) / (len(requests) - 1)
    return round(turns * requests[0] + growth * turns * (turns - 1) / 2)


def _round(x: float, digits: int = 2) -> float:
    return round(x, digits)


def fit_exponent(points: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Least-squares slope of log bytes on log turns, and the slope between neighbours."""
    xs = [math.log(t) for t, _ in points]
    ys = [math.log(b) for _, b in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )
    local = [
        {
            "from": a[0],
            "to": b[0],
            "exponent": _round(math.log(b[1] / a[1]) / math.log(b[0] / a[0]), 3),
        }
        for a, b in itertools.pairwise(points)
    ]
    quadratic = _quadratic(points)
    return {
        "exponent": _round(slope, 3),
        "between_sizes": local,
        "quadratic": quadratic,
    }


def _quadratic(points: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Exact least squares for bytes = c0 + c1*n + c2*n**2 (rational arithmetic)."""
    rows = [(Fraction(1), Fraction(n), Fraction(n * n)) for n, _ in points]
    ys = [Fraction(b) for _, b in points]
    ata = [[sum((r[i] * r[j] for r in rows), Fraction(0)) for j in range(3)] for i in range(3)]
    aty = [sum((r[i] * y for r, y in zip(rows, ys, strict=True)), Fraction(0)) for i in range(3)]
    c0, c1, c2 = _solve(ata, aty)
    return {"c0": round(float(c0)), "c1": round(float(c1)), "c2": _round(float(c2), 1)}


def _solve(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    n = len(b)
    m = [[*row, rhs] for row, rhs in zip(a, b, strict=True)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                factor = m[r][col] / m[col][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def decide(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply PLAN.md's rule to the run of :data:`DECISION_TURNS` turns."""
    row = next(r for r in rows if r["turns"] == DECISION_TURNS)
    codecs: dict[str, dict[str, Any]] = row["codecs"]
    best = max(codecs, key=lambda c: codecs[c]["ratio"])
    qualifies = codecs[best]["ratio"] >= MIN_RATIO
    codec_bytes = codecs[best]["bytes"]
    prefix_under_codec = codec_bytes / codecs[best]["prefix_bytes"]
    prefix_raw = codec_bytes / row["prefix_raw_bytes"]
    new_schema = qualifies and max(prefix_under_codec, prefix_raw) > PREFIX_MARGIN
    return {
        "turns": DECISION_TURNS,
        "best_codec": best,
        "best_ratio": codecs[best]["ratio"],
        "codec_qualifies": qualifies,
        "prefix_under_codec_gain": _round(prefix_under_codec),
        "prefix_raw_against_codec": _round(prefix_raw),
        "new_schema": new_schema,
        "rule": f"compress with the best codec if it recovers >= {MIN_RATIO}x at "
        f"{DECISION_TURNS} turns; a new cassette schema with the prefix encoding only if it beats "
        f"that codec by more than {PREFIX_MARGIN}x",
        "choice": (
            "a new cassette schema (prefix encoding)"
            if new_schema
            else (f"transparent .jsonl.{best} cassettes" if qualifies else "no change")
        ),
    }


def size_report(cassettes: dict[int, Cassette], *, compress: bool = True) -> dict[str, Any]:
    synthetic = [sizes(cassettes[n].dumps(), compress=compress) for n in TURNS]
    real = []
    for path in REAL:
        row = sizes(path.read_text(encoding="utf-8"), compress=compress)
        real.append({"cassette": path.relative_to(ROOT).as_posix(), **row})
    fit = fit_exponent([(int(r["turns"]), int(r["raw_bytes"])) for r in synthetic])
    return {
        "command": COMMAND,
        "workload": {
            "turns": list(TURNS),
            "system_chars": SYSTEM_CHARS,
            "tool_result_chars": RESULT_CHARS,
            "reply_chars": REPLY_CHARS,
            "vocabulary": VOCABULARY,
            "seed": SEED,
        },
        "codec_versions": codec_versions(),
        "synthetic": synthetic,
        "fit": fit,
        "real": real,
        "decision": decide(synthetic) if compress else None,
    }


def _median_s(fn: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return round(statistics.median(times), 4)


def _load_packed(blob: bytes, unpack: Callable[[bytes], bytes]) -> Cassette:
    return Cassette.loads(unpack(blob).decode("utf-8"))


def timings(cassettes: dict[int, Cassette], repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in TURNS:
        text = cassettes[n].dumps()
        raw = text.encode()
        load: dict[str, Any] = {"jsonl": _median_s(partial(Cassette.loads, text), repeats)}
        compress_s: dict[str, Any] = {}
        for name, compress in CODECS.items():
            compress_s[name] = _median_s(partial(compress, raw), repeats)
            blob = compress(raw)
            load[name] = _median_s(partial(_load_packed, blob, DECOMPRESS[name]), repeats)
        rows.append(
            {
                "turns": n,
                "record_s": _median_s(partial(record_run, n), 1),
                "load_s": load,
                "compress_s": compress_s,
                "replay_full_s": _median_s(partial(_replay_ok, cassettes[n]), repeats),
            }
        )
        print(f"timed {n} turns: replay {rows[-1]['replay_full_s']} s")
    return rows


def _replay_ok(cassette: Cassette) -> None:
    result = replay(cassette, run)
    if result.error is not None or result.trajectory.count(StepStatus.APPROXIMATE):
        raise RuntimeError(f"replay of {cassette.header.cassette_id} failed: {result.error}")


def store_load(cassette: Cassette, runs: int, workdir: Path) -> dict[str, Any]:
    """Replay ``cassette`` ``runs`` times into a fresh SQLite store and measure it."""
    db = workdir / f"store-{len(cassette.envelopes)}.db"
    save_s: list[float] = []
    step_json = 0
    with Store.open(f"sqlite:///{db}", clock=lambda: FIXED_TIME) as store:
        run_id = ""
        for _ in range(runs):
            result = replay(cassette, run)
            for step in result.trajectory.steps:
                step_json += len(json.dumps(step.request)) + len(json.dumps(step.response))
            start = time.perf_counter()
            run_id = store.save_run(
                result,
                [],
                scenario=f"scale-{cassette.header.cassette_id}",
                cassette_path="bench",
                agent="eval.bench_scale:run",
                passed=True,
                started_at=FIXED_TIME,
                duration_ms=0.0,
            )
            save_s.append(time.perf_counter() - start)
        payload = len(store.get_run(run_id).model_dump_json().encode())
        get_s = _median_s(lambda: store.get_run(run_id), 5)
    db_bytes = db.stat().st_size
    return {
        "turns": sum(1 for e in cassette.envelopes if e.kind is Kind.MODEL),
        "runs": runs,
        "cassette_bytes": len(cassette.dumps().encode()),
        "save_run_median_s": round(statistics.median(save_s), 4),
        "save_run_max_s": round(max(save_s), 4),
        "db_bytes": db_bytes,
        "db_bytes_per_run": round(db_bytes / runs),
        "step_json_share": _round(step_json / db_bytes, 3),
        "get_run_payload_bytes": payload,
        "get_run_median_s": get_s,
    }


# --- entry points -------------------------------------------------------------------------


def record_all() -> dict[int, Cassette]:
    return {n: record_run(n) for n in TURNS}


def check(path: Path = SIZES) -> list[str]:
    """Raw and prefix-encoded sizes that differ from the committed report (empty: none)."""
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = size_report(record_all(), compress=False)
    problems = []
    for group in ("synthetic", "real"):
        for old, new in zip(committed[group], fresh[group], strict=True):
            for key in (
                "turns",
                "envelopes",
                "raw_bytes",
                "largest_line_bytes",
                "prefix_raw_bytes",
            ):
                if old[key] != new[key]:
                    label = old.get("cassette", f"{old['turns']} turns")
                    problems.append(f"{group} {label}: {key} {old[key]} -> {new[key]}")
    return problems


def _table(report: dict[str, Any]) -> Iterator[str]:
    yield "turns  raw_bytes  largest  gzip  bz2   xz     prefix  prefix+xz"
    for r in report["synthetic"]:
        c = r["codecs"]
        yield (
            f"{r['turns']:>5}  {r['raw_bytes']:>9}  {r['largest_line_bytes']:>7}  "
            f"{c['gzip']['ratio']:>4}  {c['bz2']['ratio']:>4}  {c['xz']['ratio']:>5}  "
            f"{r['prefix_raw_ratio']:>6}  {c['xz']['prefix_ratio']:>8}"
        )
    yield f"exponent {report['fit']['exponent']}; decision: {report['decision']['choice']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=Path, default=SIZES, help="byte-stable sizes report")
    parser.add_argument("--out", type=Path, default=OUT, help="timings report")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="compare sizes with --sizes")
    args = parser.parse_args(argv)
    configure_logging()
    if args.check:
        problems = check(args.sizes)
        for p in problems:
            print(p)
        print("sizes match" if not problems else f"{len(problems)} size(s) differ")
        return 1 if problems else 0
    cassettes = record_all()
    report = size_report(cassettes)
    args.sizes.parent.mkdir(parents=True, exist_ok=True)
    args.sizes.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for line in _table(report):
        print(line)
    with tempfile.TemporaryDirectory() as tmp:
        stores = [store_load(cassettes[n], runs, Path(tmp)) for n, runs in STORE_LOADS]
    timed = {
        "command": COMMAND,
        "environment": environment(),
        "codec_versions": codec_versions(),
        "runs": timings(cassettes, args.repeats),
        "store": stores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(timed, indent=2) + "\n", encoding="utf-8")
    for s in stores:
        print(
            f"store {s['runs']} x {s['turns']} turns: {s['db_bytes']} bytes, "
            f"save_run {s['save_run_median_s']} s, get_run {s['get_run_payload_bytes']} bytes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
