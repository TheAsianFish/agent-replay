"""Checks over which tools the agent called.

``required_tools``, ``forbidden_tools`` and ``max_tool_calls`` look at the agent's own
top-level tool calls (``Trajectory.tool_steps``); calls nested inside another tool
are implementation detail and are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult


def _names(tools: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tools, str):
        raise TypeError("tools must be a sequence of tool names, not a single string")
    return tuple(tools)


@dataclass(frozen=True)
class RequiredTools:
    """Every listed tool was called at least once."""

    tools: tuple[str, ...]

    @property
    def name(self) -> str:
        return "required_tools"

    def __call__(self, run: RunResult, /) -> CheckResult:
        called = set(run.trajectory.tool_names())
        missing = [t for t in self.tools if t not in called]
        details: dict[str, JsonValue] = {"required": list(self.tools), "missing": list(missing)}
        if missing:
            message = f"required tool(s) never called: {', '.join(missing)}"
        else:
            message = f"all {len(self.tools)} required tool(s) were called"
        return CheckResult(check=self.name, passed=not missing, message=message, details=details)


def required_tools(tools: Sequence[str]) -> RequiredTools:
    """Build a ``required_tools`` check. See :class:`RequiredTools`."""
    return RequiredTools(tools=_names(tools))


@dataclass(frozen=True)
class ForbiddenTools:
    """None of the listed tools was called."""

    tools: tuple[str, ...]

    @property
    def name(self) -> str:
        return "forbidden_tools"

    def __call__(self, run: RunResult, /) -> CheckResult:
        forbidden = set(self.tools)
        calls = [(s.index, s.name) for s in run.trajectory.tool_steps() if s.name in forbidden]
        details: dict[str, JsonValue] = {
            "forbidden": list(self.tools),
            "calls": [{"tool": name, "step": index} for index, name in calls],
        }
        if calls:
            where = ", ".join(f"{name} at step {index}" for index, name in calls)
            message = f"forbidden tool(s) called: {where}"
        else:
            message = "no forbidden tool was called"
        return CheckResult(check=self.name, passed=not calls, message=message, details=details)


def forbidden_tools(tools: Sequence[str]) -> ForbiddenTools:
    """Build a ``forbidden_tools`` check. See :class:`ForbiddenTools`."""
    return ForbiddenTools(tools=_names(tools))


@dataclass(frozen=True)
class MaxToolCalls:
    """Top-level tool calls (all, or only those to ``tool``) number at most ``limit``."""

    limit: int
    tool: str | None = None

    @property
    def name(self) -> str:
        return "max_tool_calls"

    def __call__(self, run: RunResult, /) -> CheckResult:
        steps = run.trajectory.tool_steps()
        count = sum(1 for s in steps if self.tool is None or s.name == self.tool)
        what = "tool call(s)" if self.tool is None else f"call(s) to {self.tool}"
        passed = count <= self.limit
        details: dict[str, JsonValue] = {"limit": self.limit, "count": count, "tool": self.tool}
        verdict = "within" if passed else "over"
        message = f"{count} {what}, {verdict} the limit of {self.limit}"
        return CheckResult(check=self.name, passed=passed, message=message, details=details)


def max_tool_calls(limit: int, *, tool: str | None = None) -> MaxToolCalls:
    """Build a ``max_tool_calls`` check. See :class:`MaxToolCalls`."""
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return MaxToolCalls(limit=limit, tool=tool)
