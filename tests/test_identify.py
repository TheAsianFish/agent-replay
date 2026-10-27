"""Identifying arguments: parsing, served-call matching under each policy and the suite
tables."""

from __future__ import annotations

import itertools
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from agent_replay.cassette import Cassette
from agent_replay.ci import Overrides, load_suite, run_suite
from agent_replay.errors import ConfigError, UnrecordedCallError
from agent_replay.identify import check_recorded, identity, parse_identify, values
from agent_replay.record.tools import tool
from agent_replay.session import RecordingSession, ReplaySession
from agent_replay.trajectory import StepStatus

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


def test_a_typo_in_the_keys_fails_before_the_run() -> None:
    with pytest.raises(ConfigError, match="no recorded call has 'id'"):
        ReplaySession(record("#W1"), identify={"get_order": ["id"]})


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
