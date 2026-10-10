"""Checks: assertions over a replayed (or recorded) run."""

from typing import TYPE_CHECKING, Any

from agent_replay.checks.base import Check, CheckResult
from agent_replay.checks.budget import Budget, budget
from agent_replay.checks.rule import Rule, rule
from agent_replay.checks.schema import ArgSchema, OutputSchema, arg_schema, output_schema
from agent_replay.checks.spec import build_check
from agent_replay.checks.tool_sequence import ToolSequence, tool_sequence
from agent_replay.checks.tools import (
    ForbiddenTools,
    MaxToolCalls,
    RequiredTools,
    forbidden_tools,
    max_tool_calls,
    required_tools,
)

if TYPE_CHECKING:
    from agent_replay.judge.check import Judge, judge


def __getattr__(name: str) -> Any:
    # ``judge`` lives in agent_replay.judge, which imports this package's base
    # module; importing it lazily keeps either package importable first.
    if name in ("Judge", "judge"):
        from agent_replay.judge import check  # noqa: PLC0415

        return getattr(check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArgSchema",
    "Budget",
    "Check",
    "CheckResult",
    "ForbiddenTools",
    "Judge",
    "MaxToolCalls",
    "OutputSchema",
    "RequiredTools",
    "Rule",
    "ToolSequence",
    "arg_schema",
    "budget",
    "build_check",
    "forbidden_tools",
    "judge",
    "max_tool_calls",
    "output_schema",
    "required_tools",
    "rule",
    "tool_sequence",
]
