"""Per-model prices used to compute the cost of recorded model calls.

Prices are USD per million tokens, first-party Anthropic API rates, captured on
the date below. Cache writes use the 5-minute TTL rate (1.25x input). Update the
table and the date together.

``REPLAY_PRICES_FILE`` names a JSON object ``{model: {input, output, cache_read,
cache_write}}`` (USD per million tokens) that adds models to the table or
replaces their prices: negotiated rates, other providers, or a model released
after the capture date. The file is re-read whenever its path, size or
modification time changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent_replay.cassette import Usage
from agent_replay.errors import ConfigError

PRICES_CAPTURED = "2026-10-08"
PRICES_FILE_ENV = "REPLAY_PRICES_FILE"


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


class _PriceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write: float = Field(ge=0)


_FILE_ADAPTER: TypeAdapter[dict[str, _PriceEntry]] = TypeAdapter(dict[str, _PriceEntry])
_FIX = (
    f"fix the file or unset {PRICES_FILE_ENV}; it must be a JSON object like "
    '{"my-model": {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0}} '
    "in USD per million tokens"
)
_cache: dict[tuple[str, int, int], dict[str, ModelPrice]] = {}


def _read_prices_file(path: Path) -> dict[str, ModelPrice]:
    where = f"{PRICES_FILE_ENV}={str(path)!r}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{where} cannot be read ({exc.strerror or exc}); {_FIX}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{where} is not valid JSON ({exc}); {_FIX}") from exc
    try:
        entries = _FILE_ADAPTER.validate_python(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'top level'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{where} is invalid ({problems}); {_FIX}") from exc
    return {model: ModelPrice(**entry.model_dump()) for model, entry in entries.items()}


def prices() -> dict[str, ModelPrice]:
    """The price table: :data:`PRICES`, extended and overridden by ``REPLAY_PRICES_FILE``.

    Raises :class:`ConfigError` naming the variable when the file is missing,
    not JSON, or not a table of four non-negative prices per model.
    """
    raw = os.environ.get(PRICES_FILE_ENV)
    if not raw:
        return PRICES
    path = Path(raw)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ConfigError(
            f"{PRICES_FILE_ENV}={raw!r} cannot be read ({exc.strerror or exc}); point it at a "
            "JSON price file or unset it"
        ) from exc
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        table = {**PRICES, **_read_prices_file(path)}
        _cache.clear()
        _cache[key] = table
    return _cache[key]


def cost_usd(model: str | None, usage: Usage) -> float | None:
    """Cost of one call, or ``None`` when the model is not in the price table."""
    table = prices()
    if model is None or model not in table:
        return None
    price = table[model]
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )
    return round(total / 1_000_000, 8)
