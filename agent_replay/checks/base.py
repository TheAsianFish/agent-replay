"""The check protocol and its result type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

if TYPE_CHECKING:
    from agent_replay.runner import RunResult


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    passed: bool
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
    advisory: bool = False
    """Reported but never gating: a failed advisory check does not fail the run.
    An uncalibrated ``judge`` check is advisory (PLAN.md decision 5)."""

    @property
    def blocking(self) -> bool:
        """True when this result fails the run: it failed and is not advisory."""
        return not self.passed and not self.advisory


class Check(Protocol):
    """Anything with a name that turns a run into a pass/fail verdict."""

    @property
    def name(self) -> str: ...

    def __call__(self, run: RunResult, /) -> CheckResult: ...
