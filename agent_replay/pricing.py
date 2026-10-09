"""Per-model prices used to compute the cost of recorded model calls.

Prices are USD per million tokens, first-party Anthropic API rates, captured on
the date below. Cache writes use the 5-minute TTL rate (1.25x input). Update the
table and the date together.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_replay.cassette import Usage

PRICES_CAPTURED = "2026-10-08"


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
}


def cost_usd(model: str | None, usage: Usage) -> float | None:
    """Cost of one call, or ``None`` when the model is not in the price table."""
    if model is None or model not in PRICES:
        return None
    price = PRICES[model]
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )
    return round(total / 1_000_000, 8)
