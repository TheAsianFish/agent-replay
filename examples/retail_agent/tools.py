"""The retail agent's tools, ported from τ-bench's retail domain (MIT, Sierra).

Names, arguments, semantics and error messages follow
``tau_bench/envs/retail/tools/`` (see README.md for the commit). Two adaptations:

- Failures raise :class:`ValueError` with tau-bench's message (``"user not
  found"``) instead of returning ``"Error: user not found"``; the agent turns the
  exception back into an ``is_error`` tool result reading ``Error: <message>``.
- Detail tools return the record as JSON data (a copy) instead of a JSON string.

The tools operate on :data:`DB`, an in-memory copy of the vendored data that
:func:`reset_db` restores and :func:`db_hash` fingerprints, so the τ-bench reward
("the final database equals the expected one") can be computed.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from agent_replay import tool

DATA = Path(__file__).parent / "data"

Record = dict[str, Any]


class RetailDB:
    """Users, orders and products, keyed by id, exactly as in tau-bench's JSON files."""

    def __init__(self) -> None:
        self.users: dict[str, Record] = {}
        self.orders: dict[str, Record] = {}
        self.products: dict[str, Record] = {}
        self.reset()

    def reset(self) -> None:
        """Reload the vendored data, discarding every change (in place)."""
        for name in ("users", "orders", "products"):
            data = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
            setattr(self, name, data)

    def snapshot(self) -> dict[str, dict[str, Record]]:
        return {"users": self.users, "orders": self.orders, "products": self.products}

    def hash(self) -> str:
        """sha256 of the canonical JSON of the whole database."""
        canonical = json.dumps(
            self.snapshot(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


DB = RetailDB()

WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }
)
"""Tools that change the database."""

TERMINAL_TOOLS = frozenset({"transfer_to_human_agents"})
"""Tools that end the conversation without changing the database (tau-bench's terminate tools)."""


def reset_db() -> None:
    DB.reset()


def db_hash() -> str:
    return DB.hash()


def _order(order_id: str) -> Record:
    if order_id not in DB.orders:
        raise ValueError("order not found")
    return DB.orders[order_id]


@tool
def find_user_id_by_email(email: str) -> str:
    """Find user id by email. If the user is not found, the function will return an error message.

    email: the email of the user, such as 'something@example.com'.
    """
    for user_id, profile in DB.users.items():
        if profile["email"].lower() == email.lower():
            return user_id
    raise ValueError("user not found")


@tool
def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """Find user id by first name, last name, and zip code.

    If the user is not found, the function will return an error message. By
    default, find user id by email, and only call this function if the user is
    not found by email or cannot remember email. Example arguments: first_name
    'John', last_name 'Doe', zip '12345'.
    """
    for user_id, profile in DB.users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            return user_id
    raise ValueError("user not found")


@tool
def get_user_details(user_id: str) -> Record:
    """Get the details of a user, including their orders.

    user_id: the user id, such as 'sara_doe_496'.
    """
    if user_id in DB.users:
        return copy.deepcopy(DB.users[user_id])
    raise ValueError("user not found")


@tool
def get_order_details(order_id: str) -> Record:
    """Get the status and details of an order.

    order_id: the order id, such as '#W0000000'. Be careful there is a '#' symbol
    at the beginning of the order id.
    """
    return copy.deepcopy(_order(order_id))


@tool
def get_product_details(product_id: str) -> Record:
    """Get the inventory details of a product.

    product_id: the product id, such as '6086499569'. Be careful the product id is
    different from the item id.
    """
    if product_id in DB.products:
        return copy.deepcopy(DB.products[product_id])
    raise ValueError("product not found")


@tool
def cancel_pending_order(
    order_id: str, reason: Literal["no longer needed", "ordered by mistake"]
) -> Record:
    """Cancel a pending order.

    If the order is already processed or delivered, it cannot be cancelled. The
    agent needs to explain the cancellation detail and ask for explicit user
    confirmation (yes/no) to proceed. If the user confirms, the order status will
    be changed to 'cancelled' and the payment will be refunded. The refund will be
    added to the user's gift card balance immediately if the payment was made
    using a gift card, otherwise the refund would take 5-7 business days to
    process. The function returns the order details after the cancellation.

    order_id: such as '#W0000000' (with the '#'). reason: either 'no longer
    needed' or 'ordered by mistake'.
    """
    order = _order(order_id)
    if order["status"] != "pending":
        raise ValueError("non-pending order cannot be cancelled")
    if reason not in ("no longer needed", "ordered by mistake"):
        raise ValueError("invalid reason")

    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refunds.append(
            {
                "transaction_type": "refund",
                "amount": payment["amount"],
                "payment_method_id": payment_id,
            }
        )
        if "gift_card" in payment_id:  # refunded to the gift card immediately
            method = DB.users[order["user_id"]]["payment_methods"][payment_id]
            method["balance"] = round(method["balance"] + payment["amount"], 2)

    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)
    return copy.deepcopy(order)


@tool
def modify_pending_order_address(
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> Record:
    """Modify the shipping address of a pending order.

    The agent needs to explain the modification detail and ask for explicit user
    confirmation (yes/no) to proceed. order_id: such as '#W0000000'. address1:
    such as '123 Main St'; address2: such as 'Apt 1' or ''; city: such as 'San
    Francisco'; state: such as 'CA'; country: such as 'USA'; zip: such as '12345'.
    """
    order = _order(order_id)
    if order["status"] != "pending":
        raise ValueError("non-pending order cannot be modified")
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return copy.deepcopy(order)


@tool
def return_delivered_order_items(
    order_id: str, item_ids: list[str], payment_method_id: str
) -> Record:
    """Return some items of a delivered order.

    The order status will be changed to 'return requested'. The agent needs to
    explain the return detail and ask for explicit user confirmation (yes/no) to
    proceed. The user will receive follow-up email for how and where to return
    the item. order_id: such as '#W0000000'. item_ids: the item ids to be
    returned, each such as '1008292230'; there could be duplicate items in the
    list. payment_method_id: the payment method to receive the refund, such as
    'gift_card_0000000' or 'credit_card_0000000'; these can be looked up from the
    user or order details.
    """
    order = _order(order_id)
    if order["status"] != "delivered":
        raise ValueError("non-delivered order cannot be returned")

    if payment_method_id not in DB.users[order["user_id"]]["payment_methods"]:
        raise ValueError("payment method not found")
    if (
        "gift_card" not in payment_method_id
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        raise ValueError(
            "payment method should be either the original payment method or a gift card"
        )

    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            raise ValueError("some item not found")

    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id
    return copy.deepcopy(order)


@tool
def exchange_delivered_order_items(
    order_id: str, item_ids: list[str], new_item_ids: list[str], payment_method_id: str
) -> Record:
    """Exchange items in a delivered order to new items of the same product type.

    For a delivered order, return or exchange can be only done once by the agent.
    The agent needs to explain the exchange detail and ask for explicit user
    confirmation (yes/no) to proceed. order_id: such as '#W0000000'. item_ids:
    the item ids to be exchanged, each such as '1008292230'; there could be
    duplicates. new_item_ids: the item ids to be exchanged for; each new item id
    should match the item id in the same position and be of the same product.
    payment_method_id: the payment method to pay or receive refund for the item
    price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.
    """
    order = _order(order_id)
    if order["status"] != "delivered":
        raise ValueError("non-delivered order cannot be exchanged")

    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            raise ValueError(f"{item_id} not found")

    if len(item_ids) != len(new_item_ids):
        raise ValueError("the number of items to be exchanged should match")

    diff_price = 0.0
    for item_id, new_item_id in zip(item_ids, new_item_ids, strict=True):
        item = next(i for i in order["items"] if i["item_id"] == item_id)
        variants = DB.products[item["product_id"]]["variants"]
        if not (new_item_id in variants and variants[new_item_id]["available"]):
            raise ValueError(f"new item {new_item_id} not found or available")
        diff_price += variants[new_item_id]["price"] - item["price"]
    diff_price = round(diff_price, 2)

    methods = DB.users[order["user_id"]]["payment_methods"]
    if payment_method_id not in methods:
        raise ValueError("payment method not found")
    method = methods[payment_method_id]
    if method["source"] == "gift_card" and method["balance"] < diff_price:
        raise ValueError("insufficient gift card balance to pay for the price difference")

    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(item_ids)
    order["exchange_new_items"] = sorted(new_item_ids)
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = diff_price
    return copy.deepcopy(order)


_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}
_MAX_EXPONENT = 1000  # `9**9**9` would hang; eval in tau-bench has the same problem
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.expr) -> float:
    """Arithmetic only. tau-bench uses ``eval`` behind a character whitelist; this
    walks the same expressions without executing code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.

    expression: such as '2 + 2'. The expression can contain numbers, operators
    (+, -, *, /), parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        raise ValueError("invalid characters in expression")
    try:
        return str(round(float(_evaluate(ast.parse(expression, mode="eval").body)), 2))
    except Exception as exc:  # tau-bench reports any evaluation error as its message
        raise ValueError(str(exc)) from exc


@tool
def transfer_to_human_agents(summary: str) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue.

    Only transfer if the user explicitly asks for a human agent, or if the user's
    issue cannot be resolved by the agent with the available tools.
    """
    return "Transfer successful"


ALL_TOOLS = (
    find_user_id_by_email,
    find_user_id_by_name_zip,
    get_user_details,
    get_order_details,
    get_product_details,
    cancel_pending_order,
    modify_pending_order_address,
    return_delivered_order_items,
    exchange_delivered_order_items,
    calculate,
    transfer_to_human_agents,
)
