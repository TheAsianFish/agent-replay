"""Correct-but-different paths: the retail fake model taking another route to the same outcome.

Each variant wraps :func:`examples.retail_agent.fake_model.policy`. Every one
leaves the database and the answer exactly as the original does; what changes
is the path (which reads, in what order, how often, what the transfer summary
says). A check that fails one of them in ``model-live`` raises a false alarm,
and a run that ends on an unrecorded call says the frozen world, not the agent,
decided it. ``eval/mutation_study.py`` replays them under ``strict`` and
``fallback``, with and without the suite's identifying arguments.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from agent_replay.fake import FakeAnthropic, FakeRequest, Policy, Reply, ToolCall
from examples.retail_agent.fake_model import HANDLERS, PLANS, State, policy
from examples.retail_agent.tasks import task_for
from examples.retail_agent.tools import TERMINAL_TOOLS, WRITE_TOOLS

Record = dict[str, Any]

_ITEM_INTENTS = frozenset({"return_else_cancel", "return", "exchange"})


def _reads(request: FakeRequest, name: str) -> list[Record]:
    return [
        json.loads(r.content) for r in request.tool_results() if r.name == name and not r.is_error
    ]


def _only_order_reads(reply: Reply) -> bool:
    return bool(reply.tool_calls) and all(c.name == "get_order_details" for c in reply.tool_calls)


def reads_only_needed_orders(request: FakeRequest) -> Reply:
    """Reads the user's orders one at a time and stops at the first holding the item
    (one order instead of three for the laptop), where the original reads them all."""
    reply = policy(request)
    task = task_for(request.first_user_text())
    if task is None or PLANS[task.id].intent not in _ITEM_INTENTS or not _only_order_reads(reply):
        return reply
    plan = PLANS[task.id]
    [user] = _reads(request, "get_user_details")[-1:]
    orders = {o["order_id"]: o for o in _reads(request, "get_order_details")}
    for order_id in user["orders"]:
        if order_id not in orders:
            return Reply(tool_calls=[ToolCall("get_order_details", {"order_id": order_id})])
        if any(item["name"] == plan.item for item in orders[order_id]["items"]):
            read = {**user, "orders": [o for o in user["orders"] if o in orders]}
            return HANDLERS[plan.intent](State(request, plan, read, orders))
    return reply


def swaps_parallel_lookups(request: FakeRequest) -> Reply:
    """Issues each turn's parallel lookups in the reverse order (the order of parallel
    tool calls is the model's choice)."""
    reply = policy(request)
    reads = [c for c in reply.tool_calls if c.name not in WRITE_TOOLS | TERMINAL_TOOLS]
    if len(reads) < 2 or len(reads) != len(reply.tool_calls):
        return reply
    return Reply(text=reply.text, tool_calls=list(reversed(reply.tool_calls)))


def _write(reply: Reply) -> ToolCall | None:
    return next((c for c in reply.tool_calls if c.name in WRITE_TOOLS), None)


def repeats_order_lookup(request: FakeRequest) -> Reply:
    """Reads the order once more right before writing to it."""
    reply = policy(request)
    write = _write(reply)
    if write is None:
        return reply
    order_id = write.input["order_id"]
    reads = sum(1 for o in _reads(request, "get_order_details") if o["order_id"] == order_id)
    if reads >= 2:
        return reply
    return Reply(
        text="Let me re-check the order first.",
        tool_calls=[ToolCall("get_order_details", {"order_id": order_id})],
    )


def verifies_after_write(request: FakeRequest) -> Reply:
    """Reads the order back after writing to it, then answers as before (from the
    write's result)."""
    reply = policy(request)
    results = request.tool_results()
    last = results[-1] if results else None
    if reply.tool_calls or last is None or last.name not in WRITE_TOOLS or last.is_error:
        return reply
    order_id = json.loads(last.content)["order_id"]
    return Reply(
        text="Let me confirm the change.",
        tool_calls=[ToolCall("get_order_details", {"order_id": order_id})],
    )


REWORDED_SUMMARY = (
    "Customer asks to refund each of their two orders to the other order's payment "
    "method; policy only allows the original method or a gift card, so a human must decide."
)


def rewords_transfer_summary(request: FakeRequest) -> Reply:
    """Transfers to a human with the same request summarised in other words."""
    reply = policy(request)
    return Reply(
        text=reply.text,
        tool_calls=[
            ToolCall(c.name, {"summary": REWORDED_SUMMARY})
            if c.name == "transfer_to_human_agents"
            else c
            for c in reply.tool_calls
        ],
    )


@dataclass(frozen=True)
class PathVariant:
    id: str
    description: str
    policy: Policy

    @property
    def client(self) -> str:
        return f"eval.mutations.paths:{self.id}_client"


PATH_VARIANTS: tuple[PathVariant, ...] = (
    PathVariant(
        "reads_only_needed_orders",
        "reads orders one at a time and stops at the one holding the item",
        reads_only_needed_orders,
    ),
    PathVariant(
        "swaps_parallel_lookups",
        "issues each turn's parallel lookups in the reverse order",
        swaps_parallel_lookups,
    ),
    PathVariant(
        "repeats_order_lookup",
        "reads the order again right before writing to it",
        repeats_order_lookup,
    ),
    PathVariant(
        "verifies_after_write",
        "reads the order back after writing, then answers",
        verifies_after_write,
    ),
    PathVariant(
        "rewords_transfer_summary",
        "transfers to a human with the summary in other words",
        rewords_transfer_summary,
    ),
)


def _factory(variant_policy: Policy) -> Callable[[], Anthropic]:
    def make_client() -> Anthropic:
        return FakeAnthropic(variant_policy).as_anthropic()

    return make_client


reads_only_needed_orders_client = _factory(reads_only_needed_orders)
swaps_parallel_lookups_client = _factory(swaps_parallel_lookups)
repeats_order_lookup_client = _factory(repeats_order_lookup)
verifies_after_write_client = _factory(verifies_after_write)
rewords_transfer_summary_client = _factory(rewords_transfer_summary)
