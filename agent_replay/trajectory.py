"""What an agent did during one run: the ordered boundary calls it made."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay.cassette import ErrorInfo, Kind


class StepStatus(StrEnum):
    LIVE = "live"
    """The call really happened (recording)."""
    EXACT = "exact"
    """Served from the cassette; the request matched the recorded fingerprint."""
    APPROXIMATE = "approximate"
    """Served from the cassette, but the request differed from the recording."""
    UNRECORDED = "unrecorded"
    """The cassette had nothing to serve; the run stopped here."""


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
    """Which cassette envelope served (or, when recording, recorded) this step."""


class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    steps: tuple[Step, ...] = ()

    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind is Kind.TOOL]

    def tool_names(self) -> list[str]:
        return [s.name for s in self.tool_steps()]

    def count(self, status: StepStatus) -> int:
        return sum(1 for s in self.steps if s.status is status)
