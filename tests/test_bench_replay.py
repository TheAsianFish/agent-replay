"""Tests for the recorder overhead and replay speed benchmark (``eval/bench_replay.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from agent_replay._logging import configure_logging
from agent_replay.cassette import Cassette
from agent_replay.ci import load_suite
from eval import bench_replay

LIVE_SUITE = bench_replay.LIVE_SUITE


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("REPLAY_JUDGE_OFFLINE", "1")
        mp.setenv("REPLAY_LOG_LEVEL", "ERROR")
        configure_logging()
        result = bench_replay.run(iterations=6, warmup=2, batch=4, repeats=1)
    configure_logging()
    # Through JSON, as the script writes it.
    loaded: dict[str, Any] = json.loads(json.dumps(result))
    return loaded


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def test_shape(report: dict[str, Any]) -> None:
    assert set(report) >= {
        "method",
        "command",
        "parameters",
        "datasets",
        "environment",
        "boundaries",
        "session",
        "scenarios",
        "summary",
    }
    assert report["command"] == bench_replay.COMMAND
    assert report["parameters"] == {
        "iterations": 6,
        "warmup": 2,
        "batch": 4,
        "repeats": 1,
        "seed": 0,
    }
    assert set(report["environment"]) == {
        "python",
        "implementation",
        "system",
        "machine",
        "cpu_model",
        "cpu_count",
    }
    assert [r["boundary"] for r in report["boundaries"]] == list(bench_replay.BOUNDARIES)
    names = [s.name for s in load_suite(LIVE_SUITE).scenario]
    assert [r["scenario"] for r in report["scenarios"]] == names


def test_timings_are_finite_and_non_negative_where_they_must_be(
    report: dict[str, Any],
) -> None:
    for row in report["boundaries"]:
        assert row["iterations"] == 6
        assert row["request_bytes"] > 0
        for group in ("bare", "recorded", "overhead", "replay_served"):
            for key in ("median_us", "p95_us"):
                assert _finite(row[group][key])
        # A paired difference can be negative on one noisy pair; the absolute times cannot.
        for group in ("bare", "recorded", "replay_served"):
            assert row[group]["median_us"] >= 0
            assert row[group]["p95_us"] >= row[group]["median_us"]
    assert report["session"]["median_us"] > 0
    for row in report["scenarios"]:
        for key in ("replay_median_ms", "replay_and_checks_median_ms"):
            assert _finite(row[key])
            assert row[key] > 0
        assert row["speedup_replay"] > 0
    summary = report["summary"]
    assert set(summary["overhead_median_us"]) == set(bench_replay.BOUNDARIES)
    assert all(_finite(v) for v in summary.values() if not isinstance(v, dict))


def test_recorded_durations_are_the_cassettes_latencies(report: dict[str, Any]) -> None:
    for row in report["scenarios"]:
        path = LIVE_SUITE.parent / row["cassette"]
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        envelopes = [line for line in lines if line["type"] == "envelope"]
        assert row["envelopes"] == len(envelopes)
        assert row["recorded_live_ms"] == round(math.fsum(e["latency_ms"] for e in envelopes), 3)
        model = [e["latency_ms"] for e in envelopes if e["kind"] == "model"]
        assert row["model_calls"] == len(model)
        assert row["recorded_model_ms"] == round(math.fsum(model), 3)
        assert row["recorded_live_ms"] > 1000  # live recordings: seconds, not microseconds
    total = math.fsum(r["recorded_live_ms"] for r in report["scenarios"])
    assert report["summary"]["recorded_live_ms_total"] == round(total, 3)


def test_recorded_duration_sums_every_envelope() -> None:
    cassette = Cassette.load(LIVE_SUITE.parent / "cassettes/live/cancel_pending_laptop.jsonl")
    d = bench_replay.recorded_duration(cassette)
    assert (d.envelopes, d.model_calls, d.tool_calls) == (11, 5, 6)
    assert d.total_ms == 8592.176
    assert d.total_ms == pytest.approx(d.model_ms + d.tool_ms, abs=1e-3)


def test_percentile_interpolates_between_ranks() -> None:
    assert bench_replay.percentile([1.0], 95) == 1.0
    assert bench_replay.percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5
    assert bench_replay.percentile(list(range(101)), 95) == 95
    with pytest.raises(ValueError, match="no values"):
        bench_replay.percentile([], 50)


def test_text_report_names_every_row(report: dict[str, Any]) -> None:
    text = bench_replay.to_text(report)
    for row in report["boundaries"]:
        assert f"| {row['boundary']} |" in text
    for row in report["scenarios"]:
        assert f"| {row['scenario']} |" in text


def test_main_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "1")
    out = tmp_path / "bench.json"
    args = ["--iterations", "2", "--warmup", "0", "--batch", "2", "--repeats", "1"]
    assert bench_replay.main([*args, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["parameters"]["iterations"] == 2
    with pytest.raises(SystemExit):
        bench_replay.main(["--iterations", "0"])


def test_sampling_interleaves_bare_and_recorded_calls_in_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The overhead is the median of per-pair differences; that is only fair if every
    # pair runs back to back with the order alternating, after the warm-up is dropped.
    order: list[str] = []
    boundary = next(b for b in bench_replay.boundaries() if b.name == "tool")

    def timed(fn: Any) -> int:
        order.append(
            "bare" if fn is boundary.bare else "recorded" if fn is boundary.recorded else "?"
        )
        fn()
        return len(order)

    monkeypatch.setattr(bench_replay, "_time_ns", timed)
    samples = bench_replay.sample(boundary, iterations=6, batch=4)

    assert len(samples.bare) == len(samples.recorded) == len(samples.served) == 6
    paired = order[:12]  # the first session's 4 pairs, then the second's 2
    pairs = [tuple(paired[i : i + 2]) for i in range(0, 12, 2)]
    assert pairs[:4] == [("bare", "recorded"), ("recorded", "bare")] * 2
    assert "?" not in order
    # Each recorded session is replayed in full afterwards: 6 served calls, timed too.
    assert order.count("recorded") == 12
    assert order.count("bare") == 6


def test_the_overhead_is_the_median_of_pair_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    # One slow pair (a stall that hit both calls) must not move the overhead: the
    # per-pair difference stays 50 ns, where a difference of medians would not.
    bare = [100, 100, 100, 1000]
    recorded = [150, 160, 170, 1050]
    boundary = next(b for b in bench_replay.boundaries() if b.name == "tool")
    monkeypatch.setattr(
        bench_replay,
        "sample",
        lambda *a, **k: bench_replay.Samples(bare, recorded, [1, 2, 3, 4]),
    )
    row = bench_replay.measure_boundary(boundary, iterations=4, warmup=0, batch=4)
    assert row["overhead"] == {"median_us": 0.055, "p95_us": pytest.approx(0.0695, abs=1e-3)}
    assert row["bare"] == {"median_us": 0.1, "p95_us": pytest.approx(0.865, abs=1e-3)}


def test_append_overhead_is_reported_per_boundary(report: dict[str, Any]) -> None:
    for row in report["boundaries"]:
        append = row["append"]
        assert set(append) == {
            "in_memory",
            "appended",
            "overhead",
            "finish_in_memory",
            "finish_appended",
        }
        assert all(_finite(v) for group in append.values() for v in group.values())
    summary = report["summary"]
    assert set(summary["append_overhead_median_us"]) == set(bench_replay.BOUNDARIES)
    assert summary["recorded_model_calls"] == sum(
        1
        for s in load_suite(LIVE_SUITE).scenario
        for e in Cassette.load(LIVE_SUITE.parent / s.cassette).envelopes
        if e.kind.value == "model"
    )
    worst = max(summary["append_overhead_median_us"].values())
    assert summary["append_worst_share_of_model_latency"] == pytest.approx(
        worst / (summary["recorded_model_latency_median_ms"] * 1000), rel=1e-3, abs=1e-8
    )


def test_append_sampling_alternates_sessions_and_writes_each_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    boundary = next(b for b in bench_replay.boundaries() if b.name == "tool")
    sessions: list[bool] = []
    real_session = bench_replay.RecordingSession

    def session(**kwargs: Any) -> Any:
        sessions.append(kwargs.get("out") is not None)
        return real_session(**kwargs)

    monkeypatch.setattr(bench_replay, "RecordingSession", session)
    samples = bench_replay.sample_append(boundary, iterations=6, batch=4, out_dir=tmp_path)
    assert len(samples.in_memory) == len(samples.appended) == 6
    assert len(samples.finish_in_memory) == len(samples.finish_appended) == 2
    # Two batches (4 calls, then 2): memory first, then appending first.
    assert sessions == [False, True, True, False]
    assert list(tmp_path.iterdir()) == []  # every appended cassette checked, then removed
