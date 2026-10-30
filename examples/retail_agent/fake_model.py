"""A deterministic model for the retail agent, so it can be recorded without an API key.

It plays a competent agent following the policy: authenticate the user, read
their profile and the relevant orders (several lookups in one turn), check the
order status and the policy, take exactly the consequential action the request
calls for, then answer in one sentence. What it *understands* from each task's
text is written down in :data:`PLANS` (a real model reads it from the message);
every id it passes to a write action comes from tool results.

Replaying in ``full`` mode does not use it at all; it only produces recordings.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from anthropic import Anthropic

from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall
from examples.retail_agent.tasks import task_for
from examples.retail_agent.tools import TERMINAL_TOOLS, WRITE_TOOLS

Intent = Literal[
    "return_else_cancel",
    "change_address2",
    "return",
    "exchange",
    "latest_total",
    "swap_refunds",
    "cancel",
]
Record = dict[str, Any]


@dataclass(frozen=True)
class Plan:
    """What a model takes away from reading the user's message."""

    intent: Intent
    first_name: str = ""
    last_name: str = ""
    zip: str = ""
    email: str = ""
    order_id: str = ""
    item: str = ""
    """Product name of the item the request is about, e.g. ``Laptop``."""
    reason: str = ""
    address2: str = ""
    option_change: dict[str, str] = field(default_factory=dict)


PLANS: dict[str, Plan] = {
    "cancel_pending_laptop": Plan(
        "return_else_cancel", "Emma", "Smith", "10192", item="Laptop", reason="no longer needed"
    ),
    "modify_address_suite": Plan(
        "change_address2", "Fatima", "Johnson", "78712", order_id="#W8665881", address2="Suite 641"
    ),
    "return_damaged_bicycle": Plan("return", "Sofia", "Li", "78260", item="Bicycle"),
    "exchange_waterproof_boots": Plan(
        "exchange",
        "Isabella",
        "Johansson",
        "32286",
        item="Hiking Boots",
        option_change={"waterproof": "yes"},
    ),
    "lookup_latest_order_total": Plan("latest_total", "Noah", "Ito", "98187"),
    "transfer_swapped_refunds": Plan("swap_refunds", email="mia.garcia2723@example.com"),
    # Hand-written (M18): the zip is wrong, so the lookup fails and the plan stops there.
    "account_not_found_wrong_zip": Plan(
        "cancel", "Noah", "Ito", "98178", order_id="#W6729841", reason="no longer needed"
    ),
}


@dataclass(frozen=True)
class State:
    """What the conversation has established so far."""

    request: FakeRequest
    plan: Plan
    user: Record
    orders: dict[str, Record]

    def order_with(self, item: str) -> tuple[Record, list[Record]] | None:
        """The user's first order containing ``item``, with the matching items."""
        for order_id in self.user["orders"]:
            order = self.orders[order_id]
            items = [i for i in order["items"] if i["name"] == item]
            if items:
                return order, items
        return None


def _json(result_content: str) -> Record:
    data: Record = json.loads(result_content)
    return data


def original_payment(order: Record) -> str:
    method: str = order["payment_history"][0]["payment_method_id"]
    return method


def policy(request: FakeRequest) -> Reply:  # noqa: PLR0911 - one return per conversation state
    task = task_for(request.first_user_text())
    if task is None:
        return Reply(text="Sorry, I can only help with the example retail tasks.")
    plan = PLANS[task.id]

    for name in sorted(WRITE_TOOLS | TERMINAL_TOOLS):
        done = request.latest_result(name)
        if done is not None:
            return finish(name, done.content, done.is_error)

    found = request.latest_result(
        "find_user_id_by_email" if plan.email else "find_user_id_by_name_zip"
    )
    if found is None:
        if plan.email:
            return Reply(
                text="Let me find your account.",
                tool_calls=[ToolCall("find_user_id_by_email", {"email": plan.email})],
            )
        return Reply(
            text="Let me find your account.",
            tool_calls=[
                ToolCall(
                    "find_user_id_by_name_zip",
                    {"first_name": plan.first_name, "last_name": plan.last_name, "zip": plan.zip},
                )
            ],
        )
    if found.is_error:
        return Reply(text="I couldn't find your account; please check your details.")

    orders = {
        o["order_id"]: o
        for o in (
            _json(r.content)
            for r in request.tool_results()
            if r.name == "get_order_details" and not r.is_error
        )
    }
    user_result = request.latest_result("get_user_details")
    if user_result is None:
        calls = [ToolCall("get_user_details", {"user_id": found.content})]
        if plan.order_id and plan.order_id not in orders:
            calls.append(ToolCall("get_order_details", {"order_id": plan.order_id}))
        return Reply(text="Let me look at your account.", tool_calls=calls)
    user = _json(user_result.content)

    if plan.order_id:
        wanted = [plan.order_id]
    elif plan.intent == "latest_total":
        wanted = user["orders"][-1:]  # orders are listed oldest first
    else:
        wanted = list(user["orders"])
    missing = [o for o in wanted if o not in orders]
    if missing:
        return Reply(tool_calls=[ToolCall("get_order_details", {"order_id": o}) for o in missing])
    return HANDLERS[plan.intent](State(request, plan, user, orders))


def return_else_cancel(state: State) -> Reply:
    match = state.order_with(state.plan.item)
    if match is None:
        return Reply(text=f"I couldn't find a {state.plan.item.lower()} in your orders.")
    order, items = match
    if order["status"] == "delivered":
        return return_items(state, order, items)
    if order["status"] != "pending":
        return Reply(text=f"Order {order['order_id']} is {order['status']}, so I can't change it.")
    return Reply(
        text=f"Order {order['order_id']} is still pending, so it can't be returned; "
        "I'll cancel it instead.",
        tool_calls=[
            ToolCall(
                "cancel_pending_order",
                {"order_id": order["order_id"], "reason": state.plan.reason},
            )
        ],
    )


def change_address2(state: State) -> Reply:
    order = state.orders[state.plan.order_id]
    if order["status"] != "pending":
        return Reply(text=f"Order {order['order_id']} is no longer pending, so I can't change it.")
    address = {**order["address"], "address2": state.plan.address2}
    keys = ("address1", "address2", "city", "state", "country", "zip")
    return Reply(
        text="The order is pending, so I'll update its address.",
        tool_calls=[
            ToolCall(
                "modify_pending_order_address",
                {"order_id": order["order_id"], **{k: address[k] for k in keys}},
            )
        ],
    )


def return_request(state: State) -> Reply:
    match = state.order_with(state.plan.item)
    if match is None:
        return Reply(text=f"I couldn't find a {state.plan.item.lower()} in your orders.")
    return return_items(state, *match)


def return_items(state: State, order: Record, items: list[Record]) -> Reply:
    if order["status"] != "delivered":
        return Reply(text=f"Order {order['order_id']} hasn't been delivered, so I can't return it.")
    return Reply(
        text=f"Order {order['order_id']} was delivered, so I'll request the return.",
        tool_calls=[
            ToolCall(
                "return_delivered_order_items",
                {
                    "order_id": order["order_id"],
                    "item_ids": [i["item_id"] for i in items],
                    "payment_method_id": original_payment(order),
                },
            )
        ],
    )


def exchange(state: State) -> Reply:
    match = state.order_with(state.plan.item)
    if match is None:
        return Reply(text=f"I couldn't find {state.plan.item.lower()} in your orders.")
    order, [item, *_] = match
    if order["status"] != "delivered":
        return Reply(
            text=f"Order {order['order_id']} hasn't been delivered, so I can't exchange it."
        )
    product = state.request.latest_result("get_product_details")
    if product is None:
        return Reply(
            text="Let me check the available options.",
            tool_calls=[ToolCall("get_product_details", {"product_id": item["product_id"]})],
        )
    wanted = {**item["options"], **state.plan.option_change}
    variants = _json(product.content)["variants"]
    new = next((v for v in variants.values() if v["options"] == wanted and v["available"]), None)
    if new is None:
        return Reply(text="Sorry, that option isn't available, so I can't make the exchange.")
    return Reply(
        text="That option is available, so I'll request the exchange.",
        tool_calls=[
            ToolCall(
                "exchange_delivered_order_items",
                {
                    "order_id": order["order_id"],
                    "item_ids": [item["item_id"]],
                    "new_item_ids": [new["item_id"]],
                    "payment_method_id": original_payment(order),
                },
            )
        ],
    )


def cancel(state: State) -> Reply:
    order = state.orders[state.plan.order_id]
    if order["status"] != "pending":
        return Reply(text=f"Order {order['order_id']} is {order['status']}, so I can't cancel it.")
    return Reply(
        text=f"Order {order['order_id']} is pending, so I'll cancel it.",
        tool_calls=[
            ToolCall(
                "cancel_pending_order",
                {"order_id": order["order_id"], "reason": state.plan.reason},
            )
        ],
    )


def latest_total(state: State) -> Reply:
    order = state.orders[state.user["orders"][-1]]
    paid = sum(p["amount"] for p in order["payment_history"] if p["transaction_type"] == "payment")
    return Reply(text=f"You paid ${paid:.2f} for your most recent order, {order['order_id']}.")


def swap_refunds(state: State) -> Reply:
    first, second = (state.orders[o] for o in state.user["orders"][:2])
    requests = [(first, original_payment(second)), (second, original_payment(first))]
    allowed = all(
        target == original_payment(order) or "gift_card" in target for order, target in requests
    )
    if allowed:
        return Reply(text="Both refunds are allowed; please tell me which items to return.")
    return Reply(
        text="Refunds can only go to the original payment method or a gift card, so I'll "
        "transfer you as you asked.",
        tool_calls=[
            ToolCall(
                "transfer_to_human_agents",
                {
                    "summary": "The user wants to refund each order to the opposite order's "
                    "payment method, but the agent cannot help."
                },
            )
        ],
    )


HANDLERS: dict[Intent, Callable[[State], Reply]] = {
    "return_else_cancel": return_else_cancel,
    "change_address2": change_address2,
    "return": return_request,
    "exchange": exchange,
    "latest_total": latest_total,
    "swap_refunds": swap_refunds,
    "cancel": cancel,
}


def finish(tool_name: str, content: str, is_error: bool) -> Reply:
    if is_error:
        return Reply(text=f"I couldn't complete that ({content}); a colleague will follow up.")
    if tool_name in TERMINAL_TOOLS:
        return Reply(text="A human agent will take it from here.")
    order = _json(content)
    if tool_name == "modify_pending_order_address":
        a = order["address"]
        return Reply(
            text=f"Done: order {order['order_id']} will ship to {a['address1']}, "
            f"{a['address2']}, {a['city']}, {a['state']} {a['zip']}."
        )
    return Reply(text=f"Done: order {order['order_id']} is now {order['status']}.")


def make_client() -> Anthropic:
    return FakeAnthropic(policy).as_anthropic()
