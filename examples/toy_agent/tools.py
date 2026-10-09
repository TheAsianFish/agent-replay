"""The toy agent's tools, backed by a fixed in-memory order table."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent_replay import tool


class Order(BaseModel):
    order_id: str
    status: Literal["processing", "shipped", "delivered"]
    total: float
    delivered_days_ago: int | None = None


class RefundPolicy(BaseModel):
    window_days: int
    requires_delivery: bool


class Refund(BaseModel):
    refund_id: str
    order_id: str
    amount: float


ORDERS: dict[str, Order] = {
    "A1001": Order(order_id="A1001", status="delivered", total=42.50, delivered_days_ago=5),
    "A1002": Order(order_id="A1002", status="delivered", total=120.00, delivered_days_ago=45),
    "A1003": Order(order_id="A1003", status="shipped", total=18.99),
}


@tool
def lookup_order(order_id: str) -> Order:
    """Look up an order by its id (for example A1001). Returns status, total and delivery age."""
    try:
        return ORDERS[order_id]
    except KeyError:
        raise KeyError(f"no order with id {order_id}") from None


@tool
def get_refund_policy() -> RefundPolicy:
    """Return the store's refund policy."""
    return RefundPolicy(window_days=30, requires_delivery=True)


@tool
def issue_refund(order_id: str, amount: float) -> Refund:
    """Refund `amount` dollars for an order. Only call after checking the order and the policy."""
    order = ORDERS[order_id]
    if amount > order.total:
        raise ValueError(f"refund {amount:.2f} exceeds order total {order.total:.2f}")
    return Refund(refund_id=f"RF-{order_id}", order_id=order_id, amount=amount)
