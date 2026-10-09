"""JSON conversion, canonical serialisation and request fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import anthropic
from pydantic import JsonValue
from pydantic_core import PydanticSerializationError, to_jsonable_python


def _compact_sdk_models(value: object) -> object:
    """Normalise SDK values before conversion, recursively.

    - Anthropic SDK models are dumped without their ``None`` fields. SDK types
      mark every optional field ``X | None = None``, so dropping ``None`` loses
      nothing and keeps cassettes compact; it also makes a content block that
      went through a cassette round trip serialise exactly like the original.
      Other models keep their ``None`` values: a required ``x: int | None`` must
      survive.
    - ``NOT_GIVEN`` / ``omit`` dict values are dropped at every depth, as the SDK
      itself does before sending a request.
    - Sets become lists sorted by canonical JSON, so the recorded order (and the
      fingerprint) does not depend on hash randomisation.
    """
    if isinstance(value, anthropic.BaseModel):
        return value.model_dump(mode="json", exclude_none=True, by_alias=True)
    if isinstance(value, dict):
        return {
            k: _compact_sdk_models(v)
            for k, v in value.items()
            if not isinstance(v, anthropic.NotGiven | anthropic.Omit)
        }
    if isinstance(value, list | tuple):
        return [_compact_sdk_models(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((_compact_sdk_models(v) for v in value), key=canonical_json)
    return value


def _check_exact(value: JsonValue) -> None:
    """Reject converted values a JSON cassette cannot reproduce exactly."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                f"{value!r} is not finite and cannot be recorded; JSON has no inf or nan"
            )
    elif isinstance(value, str):
        _check_unicode(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_unicode(k)
            _check_exact(v)
    elif isinstance(value, list):
        for v in value:
            _check_exact(v)


def _check_unicode(text: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _invalid_unicode(exc) from exc


def _invalid_unicode(exc: UnicodeEncodeError) -> TypeError:
    return TypeError(
        f"{exc.object[exc.start : exc.end]!r} cannot be recorded: strings must be valid "
        "Unicode (no lone surrogates); decode bytes with errors='replace' first"
    )


def to_jsonable(value: object) -> JsonValue:
    """Convert ``value`` (SDK models, pydantic models, dataclasses, tuples, ...) to JSON data.

    Raises :class:`TypeError` for values JSON cannot represent exactly: unknown
    types, non-finite floats and strings that are not valid Unicode.
    """
    try:
        converted: JsonValue = to_jsonable_python(_compact_sdk_models(value))
    except UnicodeEncodeError as exc:
        raise _invalid_unicode(exc) from exc
    except (PydanticSerializationError, UnicodeDecodeError) as exc:
        raise TypeError(str(exc)) from exc
    _check_exact(converted)
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
    try:
        encoded = payload.encode("utf-8")
    except UnicodeEncodeError as exc:  # to_jsonable already rejects these; belt and braces
        raise _invalid_unicode(exc) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
