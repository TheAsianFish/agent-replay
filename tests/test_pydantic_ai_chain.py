"""The travel desk's rebooking chain: eight links, and the older bookings unchanged."""

from __future__ import annotations

import typing

from pydantic import TypeAdapter

from agent_replay.cassette import Cassette, Kind
from agent_replay.ci import load_suite
from examples.pydantic_ai_agent.tasks import TASKS
from examples.pydantic_ai_agent.tools import (
    BOOKINGS,
    CHAIN,
    Booking,
    RebookedBooking,
    convert_currency,
    get_booking,
    reset,
)
from tests.test_pydantic_ai_example import EXAMPLE, SUITES

SCENARIO = "trace_rebooking_chain"


def test_the_chain_is_eight_links_back_to_one_original() -> None:
    reset()
    ref: str | None = CHAIN[0]
    seen, paid = [], 0.0
    while ref is not None:
        booking = get_booking(ref)
        assert isinstance(booking, RebookedBooking)
        seen.append(booking.booking_ref)
        paid += booking.paid_usd
        ref = booking.rebooked_from
    assert seen == list(CHAIN)
    assert len(set(seen)) == len(seen) == 8
    assert (seen[-1], paid) == ("P6B2LH", 375.0)
    assert convert_currency(paid, "USD", "EUR") == 345.0
    assert TASKS[SCENARIO].answer == ("P6B2LH", "345")


def test_the_bookings_before_the_chain_read_exactly_as_recorded() -> None:
    # A field on Booking would add "rebooked_from": null to every recorded booking, so
    # the committed cassettes would drift in tools-live; the chain is a subclass instead.
    reset()
    for ref in ("QX7Y2K", "HB4M9T"):
        assert type(BOOKINGS[ref]) is Booking
        assert "rebooked_from" not in get_booking(ref).model_dump()
    for path in sorted(EXAMPLE.glob("cassettes/**/*.jsonl")):
        for e in Cassette.load(path).envelopes:
            if e.kind is Kind.TOOL and e.name == "get_booking" and e.error is None:
                assert isinstance(e.request, dict)
                live = get_booking(str(e.request["booking_ref"])).model_dump(mode="json")
                assert live == e.response, path.name


def test_a_served_link_keeps_its_reference() -> None:
    # Replay rebuilds a served result from the return annotation; a plain Booking would
    # drop rebooked_from and the live model would never see the next link.
    reset()
    returns = TypeAdapter(typing.get_type_hints(get_booking.fn)["return"])
    decoded = returns.validate_python(BOOKINGS[CHAIN[0]].model_dump(mode="json"))
    assert isinstance(decoded, RebookedBooking)
    assert decoded.rebooked_from == CHAIN[1]
    assert type(returns.validate_python(BOOKINGS["QX7Y2K"].model_dump(mode="json"))) is Booking


def test_the_short_scenarios_keep_their_limits() -> None:
    for suite in SUITES:
        scenarios = {s.name: s for s in load_suite(EXAMPLE / suite).scenario}
        for name, s in scenarios.items():
            limits = {c["type"]: c for c in s.checks if c["type"] in ("max_tool_calls", "budget")}
            long = name == SCENARIO
            assert limits["max_tool_calls"]["limit"] == (12 if long else 8)
            assert limits["budget"]["max_model_calls"] == (14 if long else 6)
