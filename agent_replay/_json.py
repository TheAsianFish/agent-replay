"""JSON conversion, canonical serialisation and request fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import anthropic
from pydantic import JsonValue
from pydantic_core import PydanticSerializationError, to_jsonable_python


def _compact_sdk_models(value: object) -> object:
    """Dump Anthropic SDK models without their ``None`` fields, recursively.

    SDK types mark every optional field ``X | None = None``, so dropping ``None``
    loses nothing and keeps cassettes compact; it also makes a content block that
    went through a cassette round trip serialise exactly like the original. Other
    models keep their ``None`` values: a required ``x: int | None`` must survive.
    """
    if isinstance(value, anthropic.BaseModel):
        return value.model_dump(mode="json", exclude_none=True, by_alias=True)
    if isinstance(value, dict):
        return {k: _compact_sdk_models(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_compact_sdk_models(v) for v in value]
    return value


def to_jsonable(value: object) -> JsonValue:
    """Convert ``value`` (SDK models, pydantic models, dataclasses, tuples, ...) to JSON data.

    Raises :class:`TypeError` for values JSON cannot represent.
    """
    try:
        converted: JsonValue = to_jsonable_python(_compact_sdk_models(value))
    except (PydanticSerializationError, UnicodeDecodeError) as exc:
        raise TypeError(str(exc)) from exc
    return converted


def canonical_json(value: object) -> str:
    """Serialise to the canonical form used for hashing: sorted keys, no whitespace."""
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(
    kind: str, name: str, request: object, *, ignore: frozenset[str] = frozenset()
) -> str:
    """Stable hash identifying a call by what was asked, not when.

    ``ignore`` drops top-level request fields that vary between otherwise
    identical calls (for example ``metadata`` carrying a request id).
    """
    if isinstance(request, Mapping):
        request = {k: v for k, v in request.items() if k not in ignore}
    payload = canonical_json({"kind": kind, "name": name, "request": request})
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
