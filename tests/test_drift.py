"""Tool-result drift: path parsing, the JSON comparison and the alignment with the recording."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import JsonValue

from agent_replay._json import fingerprint
from agent_replay.cassette import Cassette, Envelope, ErrorInfo, Footer, Header, Kind
from agent_replay.drift import (
    DriftReport,
    describe,
    first_difference,
    format_path,
    parse_path,
    tool_drift,
)
from agent_replay.errors import ConfigError
from agent_replay.trajectory import Step, StepStatus, Trajectory

# --- paths ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        ("created_at", ("created_at",)),
        ("items[*].price", ("items", "*", "price")),
        ("items[0].id", ("items", 0, "id")),
        ("*.etag", ("*", "etag")),
        ("a.b.c", ("a", "b", "c")),
        ("[2]", (2,)),
        ("[*]", ("*",)),
        ("grid[1][0]", ("grid", 1, 0)),
    ],
)
def test_parse_path(text: str, segments: tuple[Any, ...]) -> None:
    assert parse_path(text) == segments


@pytest.mark.parametrize("text", ["", ".a", "a.", "a..b", "a[x]", "a[-1]", "items[0]id", "a]"])
def test_parse_path_rejects_malformed_paths(text: str) -> None:
    with pytest.raises(ConfigError, match="ignore_fields"):
        parse_path(text)


def test_format_path() -> None:
    assert format_path(()) == "$"
    assert format_path(("items", 0, "status")) == "$.items[0].status"


# --- comparison -------------------------------------------------------------------


def test_equal_values_have_no_difference_whatever_the_key_order() -> None:
    assert first_difference({"a": 1, "b": [1, {"c": 2}]}, {"b": [1, {"c": 2}], "a": 1}) is None


@pytest.mark.parametrize(
    ("recorded", "live", "path"),
    [
        ({"status": "delivered"}, {"status": "pending"}, ("status",)),
        ({"a": 1}, {"a": 1, "b": 2}, ("b",)),  # a key only on one side
        ({"a": 1, "b": 2}, {"a": 1}, ("b",)),
        ([1, 2], [1, 2, 3], (2,)),  # a list grew
        ({"a": [1, 2]}, {"a": {"0": 1}}, ("a",)),  # an object where a list was
        (1, 1.0, ()),  # canonical JSON: an integer is not a float...
        (True, 1, ()),  # ...and a boolean is not an integer
        ("x", None, ()),
        # Keys are visited in sorted order, so the first difference is deterministic.
        ({"b": 1, "a": 1}, {"b": 2, "a": 2}, ("a",)),
    ],
)
def test_first_difference(recorded: JsonValue, live: JsonValue, path: tuple[Any, ...]) -> None:
    assert first_difference(recorded, live) == path


def test_ignored_paths_are_skipped_on_either_side() -> None:
    recorded = {"id": "u1", "fetched_at": "t0", "items": [{"sku": 1, "etag": "x"}]}
    live = {"id": "u1", "items": [{"sku": 1, "etag": "y"}], "trace": "abc"}
    ignore = [parse_path(p) for p in ("fetched_at", "items[*].etag", "trace")]

    assert first_difference(recorded, live, ignore) is None
    assert first_difference(recorded, live, ignore[:2]) == ("trace",)
    # An ignored path does not hide a difference next to it.
    live["items"][0]["sku"] = 2  # type: ignore[index]
    assert first_difference(recorded, live, ignore) == ("items", 0, "sku")


def test_a_wildcard_matches_one_segment_only() -> None:
    ignore = [parse_path("*.etag")]

    assert first_difference({"a": {"etag": 1}}, {"a": {"etag": 2}}, ignore) is None
    assert first_difference({"a": {"b": {"etag": 1}}}, {"a": {"b": {"etag": 2}}}, ignore) == (
        "a",
        "b",
        "etag",
    )


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=3),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=2), children, max_size=3)
    ),
    max_leaves=10,
)


@given(_json, _json)
def test_no_difference_exactly_when_the_canonical_json_is_equal(a: JsonValue, b: JsonValue) -> None:
    from agent_replay._json import canonical_json  # noqa: PLC0415

    assert (first_difference(a, b) is None) == (canonical_json(a) == canonical_json(b))


# --- alignment --------------------------------------------------------------------


def _envelope(
    seq: int,
    name: str,
    request: dict[str, JsonValue],
    response: JsonValue = None,
    *,
    kind: Kind = Kind.TOOL,
    error: ErrorInfo | None = None,
    parent: str | None = None,
) -> Envelope:
    return Envelope(
        seq=seq,
        kind=kind,
        name=name,
        fingerprint=fingerprint(kind, name, request),
        request=request,
        response=response,
        error=error,
        latency_ms=1.0,
        span_id=f"s{seq}",
        parent_span_id=parent,
    )


def _cassette(*envelopes: Envelope) -> Cassette:
    return Cassette(
        header=Header(cassette_id="c", created_at=datetime(2026, 1, 1, tzinfo=UTC)),
        envelopes=envelopes,
        footer=Footer(status="ok", envelope_count=len(envelopes)),
    )


def _step(
    index: int,
    name: str,
    request: dict[str, JsonValue],
    response: JsonValue = None,
    *,
    kind: Kind = Kind.TOOL,
    status: StepStatus = StepStatus.LIVE,
    error: ErrorInfo | None = None,
    parent: str | None = None,
) -> Step:
    return Step(
        index=index,
        kind=kind,
        name=name,
        status=status,
        fingerprint=fingerprint(kind, name, request),
        request=request,
        response=response,
        error=error,
        parent_span_id=parent,
    )


def test_matching_results_report_no_drift() -> None:
    cassette = _cassette(
        _envelope(0, "model", {"m": 1}, {"text": "hi"}, kind=Kind.MODEL),
        _envelope(1, "get_order", {"id": "A"}, {"status": "delivered"}),
        _envelope(2, "get_order", {"id": "B"}, {"status": "pending"}),
    )
    run = Trajectory(
        steps=(
            _step(0, "model", {"m": 1}, {"text": "hi"}, kind=Kind.MODEL, status=StepStatus.EXACT),
            _step(1, "get_order", {"id": "A"}, {"status": "delivered"}),
            _step(2, "get_order", {"id": "B"}, {"status": "pending"}),
        )
    )

    report = tool_drift(run, cassette)

    assert report == DriftReport(compared=2)
    assert describe(report) == "2 live tool result(s) matched the recording"


def test_a_changed_result_is_drift_at_the_first_differing_path() -> None:
    cassette = _cassette(
        _envelope(0, "get_order", {"id": "A"}, {"items": [{"sku": 1}], "status": "delivered"})
    )
    run = Trajectory(
        steps=(_step(0, "get_order", {"id": "A"}, {"items": [{"sku": 1}], "status": "pending"}),)
    )

    report = tool_drift(run, cassette)

    [drift] = report.drifted
    assert (drift.tool, drift.step, drift.envelope_seq) == ("get_order", 0, 0)
    assert (drift.path, drift.recorded, drift.live) == ("$.status", "delivered", "pending")
    assert describe(report) == (
        "1 of 1 live tool result(s) differ from the recording; first: get_order (step 0) at "
        '$.status: recorded "delivered", live "pending"'
    )


def test_ignore_fields_apply_to_their_tool_only() -> None:
    cassette = _cassette(
        _envelope(0, "get_order", {"id": "A"}, {"fetched_at": 1}),
        _envelope(1, "get_user", {"id": "u"}, {"fetched_at": 1}),
    )
    run = Trajectory(
        steps=(
            _step(0, "get_order", {"id": "A"}, {"fetched_at": 2}),
            _step(1, "get_user", {"id": "u"}, {"fetched_at": 2}),
        )
    )

    report = tool_drift(run, cassette, {"get_order": ["fetched_at"]})

    assert report.compared == 2
    assert [d.tool for d in report.drifted] == ["get_user"]


def test_calls_are_paired_by_arguments_first_then_in_order() -> None:
    """Calls made in another order still pair with the recording of the same arguments."""
    cassette = _cassette(
        _envelope(0, "get_order", {"id": "A"}, "a"),
        _envelope(1, "get_order", {"id": "B"}, "b"),
    )
    run = Trajectory(
        steps=(
            _step(0, "get_order", {"id": "B"}, "b"),
            _step(1, "get_order", {"id": "A"}, "a"),
        )
    )

    assert tool_drift(run, cassette) == DriftReport(compared=2)


def test_calls_with_other_arguments_or_none_left_are_not_compared() -> None:
    cassette = _cassette(_envelope(0, "get_order", {"id": "A"}, "a"))
    run = Trajectory(
        steps=(
            _step(0, "get_order", {"id": "Z"}, "z"),  # pairs with A, arguments differ
            _step(1, "get_order", {"id": "A"}, "a"),  # nothing left to pair with
            _step(2, "cancel", {"id": "A"}, "ok"),  # never recorded at all
        )
    )

    report = tool_drift(run, cassette)

    assert report == DriftReport(compared=0, arguments_differ=1, unrecorded=2)
    assert report.ok
    assert describe(report) == (
        "0 live tool result(s) matched the recording (1 with other arguments, 2 unrecorded "
        "not compared)"
    )


def test_served_and_fallback_steps_are_not_compared() -> None:
    cassette = _cassette(_envelope(0, "get_order", {"id": "A"}, "a"))
    run = Trajectory(
        steps=(
            _step(0, "get_order", {"id": "A"}, "other", status=StepStatus.EXACT),
            _step(1, "get_order", {"id": "A"}, "other", status=StepStatus.FALLBACK),
        )
    )

    assert tool_drift(run, cassette) == DriftReport()


def test_nested_calls_pair_with_nested_recordings_only() -> None:
    cassette = _cassette(
        _envelope(0, "lookup", {"id": "A"}, "top"),
        _envelope(1, "lookup", {"id": "A"}, "inner", parent="s0"),
    )
    run = Trajectory(
        steps=(
            _step(0, "lookup", {"id": "A"}, "inner", parent="s9"),
            _step(1, "lookup", {"id": "A"}, "top"),
        )
    )

    assert tool_drift(run, cassette) == DriftReport(compared=2)


def test_errors_compare_by_type_and_message() -> None:
    not_found = ErrorInfo(type="ValueError", module="builtins", message="order not found")
    cassette = _cassette(
        _envelope(0, "get_order", {"id": "A"}, error=not_found),
        _envelope(1, "get_order", {"id": "B"}, error=not_found),
        _envelope(2, "get_order", {"id": "C"}, {"status": "pending"}),
    )
    run = Trajectory(
        steps=(
            _step(0, "get_order", {"id": "A"}, error=not_found),
            _step(1, "get_order", {"id": "B"}, error=not_found.model_copy(update={"message": "x"})),
            _step(2, "get_order", {"id": "C"}, error=not_found),
        )
    )

    report = tool_drift(run, cassette, {"get_order": ["message"]})

    assert report.compared == 3
    first, second = report.drifted
    assert (first.step, first.path, first.live) == (1, "$.message", "x")
    # Raising where the recording returned is drift at the root, ignore_fields or not.
    assert (second.step, second.path) == (2, "$")
    assert second.recorded == {"status": "pending"}
    assert second.live == {"error": "ValueError", "message": "order not found"}
