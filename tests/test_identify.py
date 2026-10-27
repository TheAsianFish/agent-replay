"""Identifying arguments: parsing, served-call matching under each policy, tool drift and
the suite tables."""

from __future__ import annotations

import itertools
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from agent_replay._json import fingerprint
from agent_replay.cassette import Cassette, Envelope, Footer, Header, Kind
from agent_replay.ci import Overrides, load_suite, run_suite
from agent_replay.drift import DriftReport, describe, tool_drift
from agent_replay.errors import ConfigError, UnrecordedCallError
from agent_replay.identify import check_recorded, identity, parse_identify, values
from agent_replay.record.tools import tool
from agent_replay.session import RecordingSession, ReplaySession
from agent_replay.trajectory import Step, StepStatus, Trajectory

CLOCK = datetime(2026, 10, 27, tzinfo=UTC)
CALLS: list[str] = []


class Order(BaseModel):
    order_id: str
    status: str


@tool
def get_order(order_id: str, verbose: bool = False) -> Order:
    """Fetch an order."""
    CALLS.append(order_id)
    return Order(order_id=order_id, status=f"live:{order_id}")


@tool
def transfer(summary: str) -> str:
    """Hand over to a human."""
    return "transferred"


def record(*order_ids: str, summary: str | None = None) -> Cassette:
    counter = itertools.count()
    session = RecordingSession(clock=lambda: CLOCK, timer=lambda: next(counter) * 0.5)
    with session:
        for order_id in order_ids:
            get_order(order_id)
        if summary is not None:
            transfer(summary)
    return session.finish()


IDENTIFY = {"get_order": ["order_id"]}


@pytest.fixture(autouse=True)
def clear_calls() -> None:
    CALLS.clear()


# --- parsing ------------------------------------------------------------------------


def test_parse_identify_reads_paths_and_drops_duplicates() -> None:
    parsed = parse_identify({"get_order": ["order_id", "customer.id", "order_id"], "t": []})

    assert parsed == {"get_order": (("order_id",), ("customer", "id")), "t": ()}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"get_order": ["items[*].sku"]}, "has a wildcard"),
        ({"get_order": ["*"]}, "has a wildcard"),
        ({"get_order": ["a..b"]}, "identify: cannot parse 'a..b'"),
        ({"get_order": [""]}, "identify: an empty path"),
        ({"get_order": "order_id"}, 'give a list of argument paths, such as \\["order_id"\\]'),
    ],
)
def test_parse_identify_rejects_what_cannot_name_one_value(
    spec: dict[str, list[str]], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_identify(spec)


def test_identity_tells_an_absent_argument_from_null() -> None:
    paths = parse_identify({"t": ["a", "b.c"]})["t"]

    assert identity({"a": 1, "b": {"c": None}}, paths) != identity({"a": 1, "b": {}}, paths)
    assert identity({"a": 1, "x": 2}, paths) == identity({"a": 1, "x": 3}, paths)
    assert values({"a": 1, "b": {}}, paths) == {"a": 1}


def test_a_path_no_recording_has_is_a_config_error_naming_the_arguments() -> None:
    cassette = record("#W1")

    with pytest.raises(ConfigError, match=r"no recorded call has 'orderid'.*order_id, verbose"):
        check_recorded(cassette, parse_identify({"get_order": ["orderid"]}))
    check_recorded(cassette, parse_identify({"never_recorded": ["x"], "get_order": ["order_id"]}))


# --- serving ------------------------------------------------------------------------


def test_without_keys_another_entitys_recording_is_served_approximately() -> None:
    """The hazard: a read of #W9 is handed #W1's recorded order."""
    cassette = record("#W1")

    with ReplaySession(cassette) as replay:
        served = get_order("#W9")

    assert served == Order(order_id="#W1", status="live:#W1")
    assert [s.status for s in replay.trajectory.steps] == [StepStatus.APPROXIMATE]


def test_with_keys_a_call_for_an_unrecorded_entity_is_unrecorded_under_strict() -> None:
    cassette = record("#W1", "#W2")
    CALLS.clear()

    with (
        ReplaySession(cassette, identify=IDENTIFY) as replay,
        pytest.raises(UnrecordedCallError) as error,
    ):
        get_order("#W9")

    assert replay.fatal is error.value
    assert [s.status for s in replay.trajectory.steps] == [StepStatus.UNRECORDED]
    assert (
        "2 unused recording(s) of 'get_order' remain, none for order_id=\"#W9\" "
        "(the suite's identify keys)" in str(error.value)
    )
    assert CALLS == []


def test_with_keys_other_arguments_of_the_same_entity_are_still_approximate() -> None:
    cassette = record("#W1", "#W2")

    with ReplaySession(cassette, identify=IDENTIFY) as replay:
        second = get_order("#W2", verbose=True)
        first = get_order("#W1")

    assert (second.order_id, first.order_id) == ("#W2", "#W1")
    assert [s.status for s in replay.trajectory.steps] == [
        StepStatus.APPROXIMATE,
        StepStatus.EXACT,
    ]


def test_with_keys_a_repeated_read_is_unrecorded_once_its_recording_is_used() -> None:
    """Without keys the repeat would take #W2's recording; with them #W1 has none left."""
    cassette = record("#W1", "#W2")

    with ReplaySession(cassette, identify=IDENTIFY) as replay:
        get_order("#W1")
        with pytest.raises(UnrecordedCallError):
            get_order("#W1")

    assert [e.request for e in replay.unconsumed()] == [{"order_id": "#W2", "verbose": False}]


def test_fallback_runs_the_real_tool_for_an_unrecorded_entity() -> None:
    cassette = record("#W1")
    CALLS.clear()

    with ReplaySession(cassette, policy="fallback", identify=IDENTIFY) as replay:
        live = get_order("#W9")

    assert live == Order(order_id="#W9", status="live:#W9")
    assert CALLS == ["#W9"]
    assert [s.status for s in replay.trajectory.steps] == [StepStatus.FALLBACK]
    assert replay.fatal is None


def test_synth_stubs_an_unrecorded_entity() -> None:
    cassette = record("#W1")

    with ReplaySession(cassette, policy="synth", identify=IDENTIFY) as replay:
        stub = get_order("#W9")

    assert isinstance(stub, Order)
    assert [s.status for s in replay.trajectory.steps] == [StepStatus.SYNTH]


def test_a_tool_without_keys_keeps_the_approximate_fallback() -> None:
    """A reworded summary is the same transfer: no identify entry, still served."""
    cassette = record("#W1", summary="wants a refund to the other card")

    with ReplaySession(cassette, identify=IDENTIFY) as replay:
        get_order("#W1")
        assert transfer("asks for swapped refunds") == "transferred"

    assert [s.status for s in replay.trajectory.steps] == [
        StepStatus.EXACT,
        StepStatus.APPROXIMATE,
    ]


def test_an_empty_list_switches_a_tool_off() -> None:
    cassette = record("#W1")

    with ReplaySession(cassette, identify={"get_order": []}) as replay:
        get_order("#W9")

    assert [s.status for s in replay.trajectory.steps] == [StepStatus.APPROXIMATE]


def test_a_bad_identify_override_is_one_config_error(tmp_path: Path) -> None:
    suite = _identify_suite(tmp_path)

    with pytest.raises(ConfigError, match="wildcard"):
        run_suite(suite, overrides=Overrides(identify={"get_order": ("lines[*].sku",)}))


def test_a_typo_in_the_keys_fails_before_the_run() -> None:
    with pytest.raises(ConfigError, match="no recorded call has 'id'"):
        ReplaySession(record("#W1"), identify={"get_order": ["id"]})


# --- drift --------------------------------------------------------------------------


def _envelope(seq: int, name: str, request: dict[str, str], response: str) -> Envelope:
    return Envelope(
        seq=seq,
        kind=Kind.TOOL,
        name=name,
        fingerprint=fingerprint(Kind.TOOL, name, request),
        request=request,
        response=response,
        latency_ms=1.0,
        span_id=f"s{seq}",
    )


def _cassette(*envelopes: Envelope) -> Cassette:
    return Cassette(
        header=Header(cassette_id="c", created_at=CLOCK),
        envelopes=envelopes,
        footer=Footer(status="ok", envelope_count=len(envelopes)),
    )


def _live(index: int, name: str, request: dict[str, str], response: str) -> Step:
    return Step(
        index=index,
        kind=Kind.TOOL,
        name=name,
        status=StepStatus.LIVE,
        fingerprint=fingerprint(Kind.TOOL, name, request),
        request=request,
        response=response,
    )


def test_a_live_call_for_an_entity_no_recording_has_is_identity_drift() -> None:
    cassette = _cassette(_envelope(0, "get_order", {"order_id": "A"}, "a"))
    run = Trajectory(steps=(_live(0, "get_order", {"order_id": "Z"}, "z"),))

    without = tool_drift(run, cassette)
    report = tool_drift(run, cassette, identify={"get_order": ["order_id"]})

    assert without == DriftReport(arguments_differ=1)
    assert not report.ok
    [drift] = report.drifted
    assert (drift.reason, drift.envelope_seq, drift.live) == (
        "identity_unrecorded",
        None,
        {"order_id": "Z"},
    )
    assert report.identity_unrecorded == 1
    assert describe(report) == (
        "1 live tool call(s) for an entity no recording has (identity_unrecorded); first: "
        'get_order (step 0) with order_id="Z"'
    )


def test_identity_drift_counts_recordings_already_paired() -> None:
    """A second read of a recorded entity is not identity drift (the entity is known); with
    its recording paired it is counted as before."""
    cassette = _cassette(
        _envelope(0, "get_order", {"order_id": "A", "fields": "all"}, "a"),
        _envelope(1, "transfer", {"summary": "x"}, "ok"),
    )
    run = Trajectory(
        steps=(
            _live(0, "get_order", {"order_id": "A", "fields": "all"}, "a"),
            _live(1, "get_order", {"order_id": "A", "fields": "status"}, "a"),
            _live(2, "transfer", {"summary": "reworded"}, "ok"),
        )
    )

    report = tool_drift(run, cassette, identify={"get_order": ["order_id"]})

    assert report == DriftReport(compared=1, arguments_differ=1, unrecorded=1)


def test_a_tool_never_recorded_stays_unrecorded_with_keys() -> None:
    """A suite's default keys name tools some scenarios never record; those calls are
    counted as unrecorded, as without keys, not failed as identity drift."""
    cassette = _cassette(_envelope(0, "get_order", {"order_id": "A"}, "a"))
    run = Trajectory(steps=(_live(0, "get_user", {"user_id": "u"}, "u"),))

    assert tool_drift(run, cassette) == DriftReport(unrecorded=1)
    assert tool_drift(run, cassette, identify={"get_user": ["user_id"]}) == DriftReport(
        unrecorded=1
    )


def test_identity_drift_without_identifying_values_says_so() -> None:
    cassette = _cassette(_envelope(0, "get_order", {"order_id": "A"}, "a"))
    run = Trajectory(steps=(_live(0, "get_order", {"id": "A"}, "a"),))

    report = tool_drift(run, cassette, identify={"get_order": ["order_id"]})

    assert describe(report).endswith("get_order (step 0) with no identifying value")


def test_result_and_identity_drift_are_both_described() -> None:
    cassette = _cassette(_envelope(0, "get_order", {"order_id": "A"}, "a"))
    run = Trajectory(
        steps=(
            _live(0, "get_order", {"order_id": "A"}, "changed"),
            _live(1, "get_order", {"order_id": "B"}, "b"),
        )
    )

    report = tool_drift(run, cassette, identify={"get_order": ["order_id"]})

    assert [d.reason for d in report.drifted] == ["result", "identity_unrecorded"]
    assert describe(report).startswith("1 of 1 live tool result(s) differ from the recording")
    assert "; 1 live tool call(s) for an entity no recording has" in describe(report)


# --- suites -------------------------------------------------------------------------


def _suite(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "replay.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_a_suite_validates_its_identify_tables_at_load(tmp_path: Path) -> None:
    good = _suite(
        tmp_path,
        """
        [defaults.identify]
        get_order = ["order_id"]
        [[scenario]]
        name = "a"
        cassette = "a.jsonl"
        identify = { get_order = [] }
        """,
    )
    suite = load_suite(good)
    assert suite.defaults.identify == {"get_order": ("order_id",)}
    assert suite.scenario[0].identify == {"get_order": ()}

    bad = _suite(
        tmp_path,
        """
        [[scenario]]
        name = "a"
        cassette = "a.jsonl"
        identify = { get_order = ["lines[*].sku"] }
        """,
    )
    with pytest.raises(ConfigError, match=r"replay.toml: identify: tool 'get_order'.*wildcard"):
        load_suite(bad)


def _identify_suite(tmp_path: Path, scenario_identify: str = "") -> Path:
    cassette = record("#W1")
    cassette.dump(tmp_path / "a.jsonl")
    return _suite(
        tmp_path,
        f"""
        [defaults]
        agent = "tests.test_identify:reads_w9"
        [defaults.identify]
        get_order = ["order_id"]
        [[scenario]]
        name = "a"
        cassette = "a.jsonl"
        task = "t"
        {scenario_identify}
        """,
    )


def reads_w9(client: object, task: str) -> str:
    return get_order("#W9").order_id


def test_replay_ci_fails_a_read_of_an_unrecorded_entity(tmp_path: Path) -> None:
    suite = _identify_suite(tmp_path)

    report = run_suite(suite)
    [scenario] = report.scenarios
    assert not scenario.passed
    assert scenario.error is not None
    assert scenario.error.type == "UnrecordedCallError"

    without = run_suite(suite, overrides=Overrides(identify={}))
    assert without.passed
    assert without.scenarios[0].output == "#W1"


def test_a_scenario_entry_replaces_the_defaults_for_its_tool(tmp_path: Path) -> None:
    suite = _identify_suite(tmp_path, "identify = { get_order = [] }")

    assert run_suite(suite).passed


def test_a_typo_in_a_suite_breaks_the_scenario_with_the_fix(tmp_path: Path) -> None:
    suite = _identify_suite(tmp_path, 'identify = { get_order = ["orderid"] }')

    report = run_suite(suite)

    assert report.broken
    assert report.scenarios[0].error is not None
    assert "fix the path in the suite's identify table" in report.scenarios[0].error.message


def test_tools_live_drift_reports_a_read_of_an_unrecorded_entity(tmp_path: Path) -> None:
    suite = _identify_suite(tmp_path)
    overrides = Overrides(mode="tools-live", fail_on_tool_drift=True)

    report = run_suite(suite, overrides=overrides)
    [scenario] = report.scenarios
    [drift] = [c for c in scenario.checks if c.check == "no_tool_drift"]
    assert not drift.passed
    assert '(identity_unrecorded); first: get_order (step 0) with order_id="#W9"' in (drift.message)

    without = run_suite(suite, overrides=overrides.model_copy(update={"identify": {}}))
    assert without.passed


def _two_scenario_suite(tmp_path: Path, defaults: str, second: str = "") -> Path:
    record("#W1").dump(tmp_path / "orders.jsonl")
    record(summary="refund").dump(tmp_path / "transfer.jsonl")
    return _suite(
        tmp_path,
        f"""
        [defaults]
        agent = "tests.test_identify:reads_w9"
        [defaults.identify]
        {defaults}
        [[scenario]]
        name = "orders"
        cassette = "orders.jsonl"
        task = "t"
        [[scenario]]
        name = "transfer"
        cassette = "transfer.jsonl"
        task = "t"
        {second}
        """,
    )


def test_a_misspelt_tool_name_in_the_defaults_fails_the_suite(tmp_path: Path) -> None:
    """A tool no cassette recorded is never served, so the entry would do nothing."""
    suite = _two_scenario_suite(tmp_path, 'get_orders = ["order_id"]')

    with pytest.raises(
        ConfigError,
        match=r"replay.toml: identify: \[defaults.identify\], in any scenario's cassette: "
        r"no recorded call of tool 'get_orders' \(recorded tools: get_order, transfer\)",
    ):
        run_suite(suite)


def test_defaults_may_name_a_tool_only_some_scenarios_recorded(tmp_path: Path) -> None:
    suite = _two_scenario_suite(tmp_path, 'get_order = ["order_id"]')

    report = run_suite(suite)

    assert [s.ran for s in report.scenarios] == [True, True]


def test_a_scenario_entry_for_a_tool_its_cassette_lacks_fails_the_suite(tmp_path: Path) -> None:
    suite = _two_scenario_suite(
        tmp_path, 'get_order = ["order_id"]', 'identify = { get_order = ["order_id"] }'
    )

    with pytest.raises(
        ConfigError, match=r"scenario 'transfer', in its cassette: no recorded call of tool"
    ):
        run_suite(suite)
    # Switching a default off for that scenario is not a mistake.
    off = _two_scenario_suite(tmp_path, 'get_order = ["order_id"]', "identify = { get_order = [] }")
    assert [s.ran for s in run_suite(off).scenarios] == [True, True]


def test_the_defaults_are_not_checked_when_a_cassette_is_missing(tmp_path: Path) -> None:
    """The missing cassette may be the one that records the tool; its scenario says so."""
    suite = _two_scenario_suite(tmp_path, 'get_order = ["order_id"]')
    (tmp_path / "orders.jsonl").unlink()

    report = run_suite(suite)

    assert [s.ran for s in report.scenarios] == [False, True]
