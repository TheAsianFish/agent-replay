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


class Check(Protocol):
    """Anything with a name that turns a run into a pass/fail verdict."""

    @property
    def name(self) -> str: ...

    def __call__(self, run: RunResult, /) -> CheckResult: ...
