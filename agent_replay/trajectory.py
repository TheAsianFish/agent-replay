"""What an agent did during one run: the boundary calls it made, in the order they started."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.cassette import ErrorInfo, Kind, Usage


class StepStatus(StrEnum):
    LIVE = "live"
    """The call really happened (recording, or a kind this replay mode runs live)."""
    EXACT = "exact"
    """Served from the cassette; the request matched the recorded fingerprint."""
    APPROXIMATE = "approximate"
    """Served from the cassette, but the request differed from the recording."""
    UNRECORDED = "unrecorded"
    """The cassette had nothing to serve and the policy is ``strict``; the run failed here."""
    FALLBACK = "fallback"
    """Not in the cassette; performed for real under the ``fallback`` policy."""
    SYNTH = "synth"
    """Not in the cassette; answered with a schema-valid stub under the ``synth`` policy."""


SERVED = frozenset({StepStatus.EXACT, StepStatus.APPROXIMATE})
"""Statuses of steps answered from the cassette."""


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    kind: Kind
    name: str
    status: StepStatus
    fingerprint: str
    request: JsonValue
    response: JsonValue = None
    error: ErrorInfo | None = None
    envelope_seq: int | None = None
    """The source cassette envelope that served this step; when recording, the one it wrote."""
    usage: Usage | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    """Measured for live steps; the recorded latency for served steps."""
    span_id: str | None = None
    parent_span_id: str | None = None
    """Set when the call ran inside another intercepted call (a tool calling a tool)."""


class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    steps: tuple[Step, ...] = ()

    def top_level(self) -> list[Step]:
        """Steps the agent took itself, not ones nested inside another call."""
        return [s for s in self.steps if s.parent_span_id is None]

    def tool_steps(self) -> list[Step]:
        """The agent's own tool calls, in the order they started."""
        return [s for s in self.top_level() if s.kind is Kind.TOOL]

    def tool_names(self) -> list[str]:
        return [s.name for s in self.tool_steps()]

    def model_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind is Kind.MODEL]

    def count(self, status: StepStatus) -> int:
        return sum(1 for s in self.steps if s.status is status)
