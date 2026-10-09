"""A deterministic model for the toy agent, so it can be recorded without an API key.

It plays the role a real model would: check the order and the policy in one
turn, refund if eligible, then answer. Replaying in ``full`` mode does not use
it at all (model responses come from the cassette); it only produces recordings.
"""

from __future__ import annotations

import json
import re

from anthropic import Anthropic

from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall

ORDER_ID = re.compile(r"\b[A-Z]\d{4}\b")


def policy(request: FakeRequest) -> Reply:  # noqa: PLR0911 - one return per conversation state
    refund = request.latest_result("issue_refund")
    if refund is not None:
        if refund.is_error:
            return Reply(text="I couldn't complete the refund; a colleague will follow up.")
        data = json.loads(refund.content)
        return Reply(
            text=f"Done: I refunded ${data['amount']:.2f} for order {data['order_id']} "
            f"(reference {data['refund_id']})."
        )

    order = request.latest_result("lookup_order")
    terms = request.latest_result("get_refund_policy")
    if order is None:
        match = ORDER_ID.search(request.first_user_text())
        if match is None:
            return Reply(text="Could you tell me your order number?")
        calls = [ToolCall("lookup_order", {"order_id": match.group()})]
        if terms is None:
            calls.append(ToolCall("get_refund_policy"))
        return Reply(text="Let me check the order and our refund policy.", tool_calls=calls)
    if order.is_error:
        return Reply(text="I couldn't find that order; please double-check the number.")
    if terms is None:
        return Reply(tool_calls=[ToolCall("get_refund_policy")])

    o = json.loads(order.content)
    p = json.loads(terms.content)
    if o["status"] != "delivered":
        return Reply(text=f"Order {o['order_id']} hasn't been delivered yet, so I can't refund it.")
    if o["delivered_days_ago"] > p["window_days"]:
        return Reply(
            text=f"Sorry, order {o['order_id']} was delivered {o['delivered_days_ago']} days ago, "
            f"outside our {p['window_days']}-day refund window."
        )
    return Reply(
        text="The order is eligible.",
        tool_calls=[ToolCall("issue_refund", {"order_id": o["order_id"], "amount": o["total"]})],
    )


def make_client() -> Anthropic:
    return FakeAnthropic(policy).as_anthropic()
