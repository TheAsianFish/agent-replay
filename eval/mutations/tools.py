"""Tool-code mutations: one retail tool with the same name, schema and description but
wrong behaviour, injected through ``RetailAgent.tools``.

Each mutant tool is built by :func:`same_tool` from a function with the real
tool's exact signature, so the ``tools=`` list the model sees (and every
recorded request) is unchanged. Only a replay that runs the tools for real
(``tools-live``, or a ``fallback`` call) can see them, and only through
something that observes the tool's effect: the database rule, or a later step.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, ClassVar, Literal

from anthropic import Anthropic

from agent_replay import Tool, tool
from examples.retail_agent import tools as retail
from examples.retail_agent.agent import TOOLS, RetailAgent, ToolTable

Record = dict[str, Any]
Agent = Callable[[Anthropic, str], str]


def same_tool[**P, R](real: Tool[P, R]) -> Callable[[Callable[P, R]], Tool[P, R]]:
    """Decorate a mutant function as ``real``'s stand-in: same name and description."""
    return tool(name=real.name, description=real.description)


def agent_with(mutant: Tool[..., Any]) -> Agent:
    """The retail agent with ``mutant`` in place of the real tool of the same name."""

    class WithMutantTool(RetailAgent):
        tools: ClassVar[ToolTable] = {**TOOLS, mutant.name: mutant}

    return WithMutantTool().run


@same_tool(retail.cancel_pending_order)
def _cancel_keeps_status(
    order_id: str, reason: Literal["no longer needed", "ordered by mistake"]
) -> Record:
    """Refunds and reports the order cancelled, but never saves the new status."""
    result = retail.cancel_pending_order.fn(order_id, reason)
    retail.DB.orders[order_id]["status"] = "pending"
    return result


@same_tool(retail.cancel_pending_order)
def _cancel_skips_gift_card_refund(
    order_id: str, reason: Literal["no longer needed", "ordered by mistake"]
) -> Record:
    """Cancels, but the gift-card balance is never credited back."""
    balances = copy.deepcopy({uid: u["payment_methods"] for uid, u in retail.DB.users.items()})
    result = retail.cancel_pending_order.fn(order_id, reason)
    for user_id, methods in balances.items():
        retail.DB.users[user_id]["payment_methods"] = methods
    return result


@same_tool(retail.modify_pending_order_address)
def _address_drops_address2(
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> Record:
    """Saves the address without its second line."""
    return retail.modify_pending_order_address.fn(order_id, address1, "", city, state, country, zip)


@same_tool(retail.return_delivered_order_items)
def _return_ignores_item_ids(order_id: str, item_ids: list[str], payment_method_id: str) -> Record:
    """Returns every item of the order, whichever were asked for."""
    every_item = [item["item_id"] for item in retail.DB.orders.get(order_id, {}).get("items", [])]
    return retail.return_delivered_order_items.fn(order_id, every_item, payment_method_id)


@same_tool(retail.exchange_delivered_order_items)
def _exchange_flips_price_difference(
    order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str
) -> Record:
    """Records the price difference with the sign flipped (refund instead of charge)."""
    result = retail.exchange_delivered_order_items.fn(
        order_id, item_ids, new_item_ids, payment_method_id
    )
    order = retail.DB.orders[order_id]
    order["exchange_price_difference"] = -order["exchange_price_difference"]
    result["exchange_price_difference"] = order["exchange_price_difference"]
    return result


@same_tool(retail.find_user_id_by_name_zip)
def _user_lookup_builds_id(first_name: str, last_name: str, zip: str) -> str:
    """Builds a user id from the name instead of looking the user up."""
    return f"{first_name}_{last_name}".lower()


@same_tool(retail.get_product_details)
def _product_details_all_unavailable(product_id: str) -> Record:
    """Reports every variant of the product as unavailable (a stale inventory join)."""
    product = retail.get_product_details.fn(product_id)
    for variant in product["variants"].values():
        variant["available"] = False
    return product


@same_tool(retail.get_order_details)
def _order_details_stale_status(order_id: str) -> Record:
    """Reports every order as still pending (the status read from a stale cache)."""
    order = retail.get_order_details.fn(order_id)
    order["status"] = "pending"
    return order


MUTANT_TOOLS: dict[str, Tool[..., Any]] = {
    "cancel_keeps_status": _cancel_keeps_status,
    "cancel_skips_gift_card_refund": _cancel_skips_gift_card_refund,
    "address_drops_address2": _address_drops_address2,
    "return_ignores_item_ids": _return_ignores_item_ids,
    "exchange_flips_price_difference": _exchange_flips_price_difference,
    "user_lookup_builds_id": _user_lookup_builds_id,
    "product_details_all_unavailable": _product_details_all_unavailable,
    "order_details_stale_status": _order_details_stale_status,
}
"""Mutation id -> the mutant tool (its ``name`` is the real tool it replaces)."""

cancel_keeps_status = agent_with(_cancel_keeps_status)
cancel_skips_gift_card_refund = agent_with(_cancel_skips_gift_card_refund)
address_drops_address2 = agent_with(_address_drops_address2)
return_ignores_item_ids = agent_with(_return_ignores_item_ids)
exchange_flips_price_difference = agent_with(_exchange_flips_price_difference)
user_lookup_builds_id = agent_with(_user_lookup_builds_id)
product_details_all_unavailable = agent_with(_product_details_all_unavailable)
order_details_stale_status = agent_with(_order_details_stale_status)
