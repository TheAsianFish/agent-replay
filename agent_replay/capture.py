"""Clock and randomness, captured so replay sees the values the recording saw.

Agent code that reads the time or draws random numbers through these helpers
behaves exactly like the standard library outside a session. Inside a
recording each value is recorded as a ``clock`` or ``random`` envelope, and on
replay (in every mode) the recorded value is returned instead, so "orders
older than 30 days" or a sampled retry delay come out the same as recorded.

Values are recorded as JSON: datetimes as ISO 8601 strings, UUIDs as their
canonical text, ``choice`` as the index it drew (so any sequence works).
"""

from __future__ import annotations

import random as _random
import time as _time
import uuid as _uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import JsonValue, TypeAdapter

from agent_replay._json import fingerprint
from agent_replay.cassette import Kind
from agent_replay.session import Call, current_session

_rng = _random.Random()  # noqa: S311 - simulation randomness for agents, not cryptography


def _capture[T](
    kind: Kind,
    name: str,
    request: dict[str, JsonValue],
    produce: Callable[[], T],
    encode: Callable[[T], JsonValue],
    decode: Callable[[JsonValue], T],
) -> T:
    session = current_session()
    if session is None:
        return produce()
    call: Call[T] = Call(
        kind=kind,
        name=name,
        request=request,
        fingerprint=fingerprint(kind, name, request),
        invoke=produce,
        encode=encode,
        decode=decode,
    )
    return session.intercept(call)


_DATETIME = TypeAdapter(datetime)
_FLOAT = TypeAdapter(float)
_INT = TypeAdapter(int)


def _int_in(low: int, high: int) -> Callable[[JsonValue], int]:
    def decode(data: JsonValue) -> int:
        value = _INT.validate_python(data, strict=True)
        if not low <= value <= high:
            raise ValueError(f"recorded {value} is outside [{low}, {high}]")
        return value

    return decode


def now() -> datetime:
    """The current time as an aware UTC datetime, like ``datetime.now(UTC)``."""
    return _capture(
        Kind.CLOCK,
        "clock.now",
        {},
        lambda: datetime.now(UTC),
        lambda value: value.isoformat(),
        _DATETIME.validate_python,
    )


def time() -> float:
    """Seconds since the epoch, like ``time.time()``."""
    return _capture(
        Kind.CLOCK, "clock.time", {}, _time.time, lambda value: value, _FLOAT.validate_python
    )


def random() -> float:
    """A float in [0, 1), like ``random.random()``."""
    return _capture(
        Kind.RANDOM, "random.random", {}, _rng.random, lambda value: value, _FLOAT.validate_python
    )


def randint(a: int, b: int) -> int:
    """An integer in [a, b], both ends included, like ``random.randint``."""
    return _capture(
        Kind.RANDOM,
        "random.randint",
        {"a": a, "b": b},
        lambda: _rng.randint(a, b),
        lambda value: value,
        _int_in(a, b),
    )


def choice[T](seq: Sequence[T]) -> T:
    """A random element of a non-empty sequence, like ``random.choice``.

    The drawn index is what gets recorded, so the elements need not be JSON.
    """
    if not seq:
        raise IndexError("cannot choose from an empty sequence")
    index = _capture(
        Kind.RANDOM,
        "random.choice",
        {"n": len(seq)},
        lambda: _rng.randrange(len(seq)),
        lambda value: value,
        _int_in(0, len(seq) - 1),
    )
    return seq[index]


def uuid4() -> _uuid.UUID:
    """A random UUID, like ``uuid.uuid4()``."""
    return _capture(
        Kind.RANDOM,
        "random.uuid4",
        {},
        _uuid.uuid4,
        str,
        lambda data: _uuid.UUID(str(data)),
    )


def seed(value: int) -> None:
    """Seed the generator behind :func:`random`, :func:`randint` and :func:`choice`."""
    _rng.seed(value)
