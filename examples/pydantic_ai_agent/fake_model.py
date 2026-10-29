"""A deterministic model for the travel desk, so the agent records with no model server.

It plays a careful agent: one lookup (one per link of the rebooking chain), then
the action or conversion that needs the lookup's result, then one sentence. What
it understands from each task's text is :data:`examples.pydantic_ai_agent.tasks.TASKS`;
every id and amount it passes on comes from tool results. ``replay ci`` in ``full``
mode does not use it; it only produces the committed fake-model cassettes.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

from agent_replay.fake import FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from examples.pydantic_ai_agent.tasks import task_for


def _result(request: FakeRequest, name: str) -> Any:
    """The latest result of ``name`` as JSON data, or ``None`` before it was called.

    pydantic-ai sends a list result as one text block per item, which the fake's
    request joins into one string, so a list comes back as consecutive objects.
    """
    result = request.latest_result(name)
    if result is None:
        return None
    decoder, text, values = json.JSONDecoder(), result.content.strip(), []
    while text:
        value, end = decoder.raw_decode(text)
        values.append(value)
        text = text[end:].lstrip()
    return values if len(values) != 1 or name == "search_flights" else values[0]


def policy(request: FakeRequest) -> Reply:  # noqa: PLR0911 - one branch per step of each task
    task = task_for(request.first_user_text())
    if task.name == "cheapest_fare_in_euros":
        flights = _result(request, "search_flights")
        if flights is None:
            return Reply(
                tool_calls=[ToolCall("search_flights", _route("SFO", "JFK", "2026-11-03"))]
            )
        cheapest = min(flights, key=lambda f: f["price_usd"])
        euros = _result(request, "convert_currency")
        if euros is None:
            args = {"amount": cheapest["price_usd"], "from_currency": "USD", "to_currency": "EUR"}
            return Reply(tool_calls=[ToolCall("convert_currency", args)])
        return Reply(text=f"The cheapest is {cheapest['flight_id']} at {euros:.2f} EUR.")
    if task.name == "book_earliest_flight":
        flights = _result(request, "search_flights")
        if flights is None:
            return Reply(
                tool_calls=[ToolCall("search_flights", _route("SEA", "LAX", "2026-11-05"))]
            )
        booking = _result(request, "book_flight")
        if booking is None:
            earliest = min(flights, key=lambda f: f["departs"])
            args = {"flight_id": earliest["flight_id"], "passenger": "Ana Lima"}
            return Reply(tool_calls=[ToolCall("book_flight", args)])
        return Reply(
            text=f"Booked {booking['flight_id']} for Ana Lima, reference {booking['booking_ref']}."
        )
    if task.name == "trace_rebooking_chain":
        return _trace_chain(request)
    booking = _result(request, "get_booking")
    if booking is None:
        return Reply(tool_calls=[ToolCall("get_booking", {"booking_ref": "QX7Y2K"})])
    if not booking["refundable"]:
        return Reply(text="Booking QX7Y2K is not refundable, so I left it in place.")
    if _result(request, "cancel_booking") is None:
        return Reply(tool_calls=[ToolCall("cancel_booking", {"booking_ref": "QX7Y2K"})])
    return Reply(text=f"Cancelled QX7Y2K; you get ${booking['paid_usd']:.2f} back.")


def _trace_chain(request: FakeRequest) -> Reply:
    """One ``get_booking`` per link, each on the reference the last one named, then one
    conversion of the summed payments."""
    links = [json.loads(r.content) for r in request.tool_results() if r.name == "get_booking"]
    if not links:
        return Reply(tool_calls=[ToolCall("get_booking", {"booking_ref": "K7W2QD"})])
    if links[-1]["rebooked_from"] is not None:
        args = {"booking_ref": links[-1]["rebooked_from"]}
        return Reply(tool_calls=[ToolCall("get_booking", args)])
    euros = _result(request, "convert_currency")
    if euros is None:
        total = sum(link["paid_usd"] for link in links)
        args = {"amount": total, "from_currency": "USD", "to_currency": "EUR"}
        return Reply(tool_calls=[ToolCall("convert_currency", args)])
    origin = links[-1]["booking_ref"]
    return Reply(text=f"The original booking is {origin}; you paid {euros:.2f} EUR in all.")


def _route(origin: str, destination: str, date: str) -> dict[str, Any]:
    return {"origin": origin, "destination": destination, "date": date}


def make_client() -> AsyncAnthropic:
    """The ``--client`` factory: the fake model behind the async client's interface."""
    return FakeAsyncAnthropic(policy).as_anthropic()
