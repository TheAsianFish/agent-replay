"""The travel desk's tasks and what a correct run does for each."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    name: str
    text: str
    answer: tuple[str, ...]
    """Figures the final answer must contain, as written (``B6416``, ``173.88``)."""
    writes: tuple[tuple[str, dict[str, Any]], ...] = field(default=())
    """The consequential calls (``book_flight``, ``cancel_booking``), in order."""


TASKS: dict[str, Task] = {
    t.name: t
    for t in (
        Task(
            "cheapest_fare_in_euros",
            "What is the cheapest flight from SFO to JFK on 2026-11-03? "
            "Tell me its price in euros.",
            # B6416 at $189.00, at 0.92 EUR per USD. The flight is part of the answer: a
            # qwen2.5:7b run stated the right price under the wrong flight (UA512).
            answer=("B6416", "173.88"),
        ),
        Task(
            "book_earliest_flight",
            "Please book the earliest flight from SEA to LAX on 2026-11-05 for Ana Lima.",
            # DL2207 leaves at 06:10; the search lists it second. The traveller needs the
            # booking reference the tool returns.
            answer=("NEW001",),
            writes=(("book_flight", {"flight_id": "DL2207", "passenger": "Ana Lima"}),),
        ),
        Task(
            "cancel_refundable_booking",
            "I'm Sam Ortiz. Cancel my booking QX7Y2K if it is refundable, "
            "and tell me how much I get back.",
            # Refundable, paid $243.00.
            answer=("243",),
            writes=(("cancel_booking", {"booking_ref": "QX7Y2K"}),),
        ),
        Task(
            "trace_rebooking_chain",
            "I'm Lee Park. My booking K7W2QD was rebooked from an earlier booking, which "
            "was itself rebooked, and so on. Follow the chain back to the original "
            "booking and tell me its reference and the total I paid across every "
            "booking in the chain, in euros.",
            # Eight links (tools.CHAIN), one get_booking each, the original P6B2LH last:
            # $200.00 plus seven $25.00 change fees is $375.00, 345.00 EUR at 0.92.
            answer=("P6B2LH", "345"),
        ),
    )
}


def task_for(text: str) -> Task:
    for task in TASKS.values():
        if task.text == text:
            return task
    raise KeyError(f"not a travel desk task: {text!r}")
