"""Cassette schema v1, frozen.

These models are a copy of the v1 schema as it shipped, kept separate from the
current models in :mod:`agent_replay.cassette` so that changing the current
schema can never change how a v1 file is read. A v1 cassette is validated
against them and then migrated in memory (:func:`migrate`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KindV1(StrEnum):
    MODEL = "model"
    TOOL = "tool"


class ErrorInfoV1(_Strict):
    type: str
    module: str | None = None
    message: str


class UsageV1(_Strict):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class HeaderV1(_Strict):
    type: Literal["header"] = "header"
    schema_version: Literal[1] = 1
    cassette_id: str
    created_at: datetime
    agent: str | None = None
    task: str | None = None
    recorder: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EnvelopeV1(_Strict):
    type: Literal["envelope"] = "envelope"
    seq: int
    kind: KindV1
    name: str
    fingerprint: str
    request: JsonValue
    response: JsonValue = None
    error: ErrorInfoV1 | None = None
    usage: UsageV1 | None = None
    cost_usd: float | None = None
    latency_ms: float
    span_id: str
    parent_span_id: str | None = None


class FooterV1(_Strict):
    type: Literal["footer"] = "footer"
    status: Literal["ok", "error"]
    output: JsonValue = None
    error: ErrorInfoV1 | None = None
    envelope_count: int


def migrate(
    header: HeaderV1, envelopes: list[EnvelopeV1], footer: FooterV1
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]], dict[str, JsonValue]]:
    """v1 -> v2 as plain JSON, ready to validate against the v2 models.

    v2 only added envelope kinds (``clock``, ``random``), so every v1 field
    carries over unchanged and only the version number moves.
    """
    v2_header = header.model_dump(mode="json")
    v2_header["schema_version"] = 2
    return (
        v2_header,
        [e.model_dump(mode="json") for e in envelopes],
        footer.model_dump(mode="json"),
    )
