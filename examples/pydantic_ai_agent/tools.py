"""The travel desk's tools: flight search, bookings and currency conversion.

The data is a small fixed world, so every task has one right answer the rules
can check. Writes (``book_flight``, ``cancel_booking``) act on :data:`BOOKINGS`,
which :func:`reset` restores at the start of every run. Each tool is an
``agent_replay`` tool (recorded and served from cassettes) and is handed to
pydantic-ai unchanged: it reads the name, docstring and annotations.

A bad argument (an unknown booking, a full flight) raises pydantic-ai's
``ModelRetry``, which goes back to the model as a retry prompt; any other
exception ends the whole run. A replay re-raises the recorded ``ModelRetry`` as a
``ModelRetry``, so the model sees the same prompt.
"""

from __future__ import annotations

import copy
import threading

from pydantic import BaseModel
from pydantic_ai import ModelRetry

from agent_replay import tool


class Flight(BaseModel):
    flight_id: str
    origin: str
    destination: str
    date: str
    departs: str
    """Local departure time, 24-hour ``HH:MM``."""
    price_usd: float
    seats_left: int


class Booking(BaseModel):
    booking_ref: str
    flight_id: str
    passenger: str
    status: str
    """``confirmed``, ``cancelled`` or ``rebooked`` (replaced by a later booking)."""
    refundable: bool
    paid_usd: float


class RebookedBooking(Booking):
    """A booking in a rebooking chain: ``rebooked_from`` names the booking it replaced,
    ``None`` on the chain's original. A subclass rather than a new field on
    :class:`Booking`, so the bookings that predate chains read exactly as recorded."""

    rebooked_from: str | None


FLIGHTS: tuple[Flight, ...] = (
    Flight(
        flight_id="UA512",
        origin="SFO",
        destination="JFK",
        date="2026-11-03",
        departs="07:15",
        price_usd=243.0,
        seats_left=4,
    ),
    Flight(
        flight_id="B6416",
        origin="SFO",
        destination="JFK",
        date="2026-11-03",
        departs="11:40",
        price_usd=189.0,
        seats_left=9,
    ),
    Flight(
        flight_id="AA178",
        origin="SFO",
        destination="JFK",
        date="2026-11-03",
        departs="22:05",
        price_usd=212.0,
        seats_left=0,
    ),
    Flight(
        flight_id="AS330",
        origin="SEA",
        destination="LAX",
        date="2026-11-05",
        departs="09:30",
        price_usd=129.0,
        seats_left=12,
    ),
    Flight(
        flight_id="DL2207",
        origin="SEA",
        destination="LAX",
        date="2026-11-05",
        departs="06:10",
        price_usd=158.0,
        seats_left=3,
    ),
    Flight(
        flight_id="WN1849",
        origin="SEA",
        destination="LAX",
        date="2026-11-05",
        departs="13:55",
        price_usd=99.0,
        seats_left=20,
    ),
)

USD_RATES: dict[str, float] = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.0}
"""Units of each currency per US dollar; fixed so the right answer never moves."""

_INITIAL_BOOKINGS: dict[str, Booking] = {
    "QX7Y2K": Booking(
        booking_ref="QX7Y2K",
        flight_id="UA512",
        passenger="Sam Ortiz",
        status="confirmed",
        refundable=True,
        paid_usd=243.0,
    ),
    "HB4M9T": Booking(
        booking_ref="HB4M9T",
        flight_id="B6416",
        passenger="Sam Ortiz",
        status="confirmed",
        refundable=False,
        paid_usd=189.0,
    ),
}
CHAIN_PASSENGER = "Lee Park"
CHAIN: tuple[str, ...] = (
    "K7W2QD",
    "R4N8XC",
    "T2M6PB",
    "H9F3VL",
    "Z5Q1JD",
    "C8X4RN",
    "M3V7TE",
    "P6B2LH",
)
"""A rebooking chain, newest first: each booking was rebooked from the next, and the
last is the original. Reading it back takes one ``get_booking`` per link, each on a
reference the previous result gave, so a model cannot fetch the links in parallel."""
CHAIN_FARE_USD = 200.0
CHAIN_FEE_USD = 25.0
"""The original fare and the change fee paid on each rebooking: 375.00 USD in all."""


def _chain() -> dict[str, Booking]:
    links = zip(CHAIN, (*CHAIN[1:], None), strict=True)
    return {
        ref: RebookedBooking(
            booking_ref=ref,
            flight_id=FLIGHTS[i % len(FLIGHTS)].flight_id,
            passenger=CHAIN_PASSENGER,
            status="confirmed" if i == 0 else "rebooked",
            refundable=False,
            paid_usd=CHAIN_FEE_USD if previous is not None else CHAIN_FARE_USD,
            rebooked_from=previous,
        )
        for i, (ref, previous) in enumerate(links)
    }


_INITIAL_BOOKINGS.update(_chain())
BOOKINGS: dict[str, Booking] = {}
_LOCK = threading.Lock()
_next_ref = 0


def reset() -> None:
    """Restore the bookings every run starts from."""
    global _next_ref  # noqa: PLW0603 - module state is the world the tools act on
    with _LOCK:
        BOOKINGS.clear()
        BOOKINGS.update(copy.deepcopy(_INITIAL_BOOKINGS))
        _next_ref = 0


@tool
def search_flights(origin: str, destination: str, date: str) -> list[Flight]:
    """Flights between two airports (IATA codes such as SFO) on a date (YYYY-MM-DD)."""
    o, d = origin.strip().upper(), destination.strip().upper()
    return [f for f in FLIGHTS if f.origin == o and f.destination == d and f.date == date.strip()]


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount between currencies (ISO codes such as USD, EUR), rounded to cents."""
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    if src not in USD_RATES or dst not in USD_RATES:
        raise ModelRetry(f"unknown currency; supported: {', '.join(USD_RATES)}")
    return round(amount / USD_RATES[src] * USD_RATES[dst], 2)


@tool
def get_booking(booking_ref: str) -> RebookedBooking | Booking:
    """A booking by its six-character reference."""
    with _LOCK:
        booking = BOOKINGS.get(booking_ref.strip().upper())
        if booking is None:
            raise ModelRetry(f"no booking {booking_ref!r}")
        return booking.model_copy()


@tool
def book_flight(flight_id: str, passenger: str) -> Booking:
    """Book one seat on a flight for a passenger (full name). Charges the fare."""
    global _next_ref  # noqa: PLW0603
    flight = next((f for f in FLIGHTS if f.flight_id == flight_id.strip().upper()), None)
    if flight is None:
        raise ModelRetry(f"no flight {flight_id!r}")
    if flight.seats_left <= 0:
        raise ModelRetry(f"flight {flight.flight_id} is full")
    with _LOCK:
        _next_ref += 1
        booking = Booking(
            booking_ref=f"NEW{_next_ref:03d}",
            flight_id=flight.flight_id,
            passenger=passenger.strip(),
            status="confirmed",
            refundable=False,
            paid_usd=flight.price_usd,
        )
        BOOKINGS[booking.booking_ref] = booking
        return booking.model_copy()


@tool
def cancel_booking(booking_ref: str) -> Booking:
    """Cancel a confirmed booking. Refundable bookings are refunded in full; others are not."""
    with _LOCK:
        booking = BOOKINGS.get(booking_ref.strip().upper())
        if booking is None:
            raise ModelRetry(f"no booking {booking_ref!r}")
        if booking.status != "confirmed":
            raise ModelRetry(f"booking {booking.booking_ref} is already {booking.status}")
        booking.status = "cancelled"
        return booking.model_copy()


ALL_TOOLS = (search_flights, convert_currency, get_booking, book_flight, cancel_booking)
WRITE_TOOLS = frozenset({"book_flight", "cancel_booking"})
