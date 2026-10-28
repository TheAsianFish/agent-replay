"""Tests for the cassette-bytes-at-scale benchmark (``eval/bench_scale.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_replay.cassette import Kind
from agent_replay.runner import replay
from agent_replay.trajectory import StepStatus
from eval import bench_scale as bs

ROOT = Path(__file__).resolve().parents[1]


def test_committed_raw_and_prefix_sizes_are_reproduced() -> None:
    """The CI half of `make bench-scale`: raw and prefix-encoded bytes are byte-stable."""
    assert bs.check() == []


def test_a_run_has_one_model_call_a_turn_and_replays_exactly() -> None:
    cassette = bs.record_run(5)
    kinds = [e.kind for e in cassette.envelopes]
    assert kinds.count(Kind.MODEL) == 5
    assert kinds.count(Kind.TOOL) == 4
    result = replay(cassette, bs.run)
    assert result.error is None
    assert result.trajectory.count(StepStatus.EXACT) == len(result.trajectory.steps)


def test_recording_is_byte_stable() -> None:
    assert bs.record_run(7).dumps() == bs.record_run(7).dumps()


def test_text_is_seeded_and_does_not_repeat() -> None:
    a, b = bs.words("record-1", 220), bs.words("record-2", 220)
    assert a == bs.words("record-1", 220)
    assert len(a) == 220
    assert a != b
    assert len(set(a.split()) & set(b.split())) < len(a.split()) // 2


@pytest.mark.parametrize(
    "path",
    [
        *bs.REAL,
        ROOT / "examples" / "retail_agent" / "cassettes" / "cancel_pending_laptop.jsonl",
    ],
    ids=lambda p: p.name,
)
def test_prefix_encoding_round_trips_real_cassettes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    encoded = bs.prefix_encode(text)
    assert bs.prefix_decode(encoded) == bs._normalised(text)
    assert len(encoded) < len(text)


def test_prefix_encoding_keeps_a_request_that_does_not_extend_the_last() -> None:
    def line(seq: int, messages: list[str], system: str = "s") -> str:
        request = {"system": system, "messages": messages}
        return json.dumps({"type": "envelope", "seq": seq, "kind": "model", "request": request})

    text = "\n".join(
        [
            '{"type":"header"}',
            line(0, ["a"]),
            line(1, ["a", "b"]),
            line(2, ["x"]),  # a new conversation: stored whole
            line(3, ["x", "y"], system="t"),  # extends it, with a changed field
            '{"type":"footer"}',
        ]
    )
    encoded = [json.loads(x) for x in bs.prefix_encode(text).split("\n") if x]
    assert encoded[2]["request"] == {"$prefix": 0, "$append": ["b"], "$set": {}}
    assert encoded[3]["request"] == {"system": "s", "messages": ["x"]}
    assert encoded[4]["request"] == {"$prefix": 2, "$append": ["y"], "$set": {"system": "t"}}
    assert bs.prefix_decode(bs.prefix_encode(text)) == bs._normalised(text)


def test_fit_recovers_an_exact_quadratic() -> None:
    points = [(n, 500 + 7000 * n + 300 * n * n) for n in (5, 20, 50, 100, 200)]
    fit = bs.fit_exponent(points)
    assert fit["quadratic"] == {"c0": 500, "c1": 7000, "c2": 300.0}
    local = [x["exponent"] for x in fit["between_sizes"]]
    assert local == sorted(local)  # the slope climbs towards 2 as the n**2 term takes over
    assert 1 < fit["exponent"] < 2


def _row(
    turns: int, ratios: dict[str, float], prefix_codec: float, prefix_raw: int
) -> dict[str, Any]:
    raw = 1_000_000
    return {
        "turns": turns,
        "prefix_raw_bytes": prefix_raw,
        "codecs": {
            name: {
                "bytes": round(raw / r),
                "ratio": r,
                "prefix_bytes": round(raw / r / prefix_codec),
            }
            for name, r in ratios.items()
        },
    }


def test_decision_picks_the_best_codec_over_ten_x() -> None:
    rows = [_row(100, {"gzip": 4.0, "xz": 100.0}, prefix_codec=1.1, prefix_raw=50_000)]
    decision = bs.decide(rows)
    assert decision["best_codec"] == "xz"
    assert decision["new_schema"] is False
    assert decision["choice"] == "transparent .jsonl.xz cassettes"


def test_decision_changes_the_schema_only_past_three_x() -> None:
    rows = [_row(100, {"xz": 20.0}, prefix_codec=3.5, prefix_raw=400_000)]
    assert bs.decide(rows)["new_schema"] is True
    rows = [_row(100, {"xz": 20.0}, prefix_codec=2.9, prefix_raw=400_000)]
    assert bs.decide(rows)["new_schema"] is False


def test_decision_changes_nothing_when_no_codec_reaches_ten_x() -> None:
    rows = [_row(100, {"gzip": 4.0, "xz": 9.0}, prefix_codec=5.0, prefix_raw=10_000)]
    decision = bs.decide(rows)
    assert decision["codec_qualifies"] is False
    assert decision["choice"] == "no change"


def test_committed_report_answers_the_window_hypothesis() -> None:
    report = json.loads(bs.SIZES.read_text(encoding="utf-8"))
    rows = {r["turns"]: r for r in report["synthetic"]}
    window = 32 * 1024
    # gzip recovers more while an envelope fits its 32 KB window and collapses once it does not
    assert rows[20]["largest_line_bytes"] < window < rows[50]["largest_line_bytes"]
    assert rows[20]["codecs"]["gzip"]["ratio"] > rows[50]["codecs"]["gzip"]["ratio"]
    assert rows[100]["codecs"]["gzip"]["ratio"] < 5
    assert rows[100]["codecs"]["xz"]["ratio"] >= bs.MIN_RATIO
    assert report["decision"]["choice"] == "transparent .jsonl.xz cassettes"
