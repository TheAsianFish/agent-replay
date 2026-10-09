"""``tool_sequence``: did the agent call the expected tools in the expected order?"""

from __future__ import annotations

import difflib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue

from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

SequenceMode = Literal["exact", "in_order"]


def _matched_prefix(needle: Sequence[str], haystack: Sequence[str]) -> int:
    """How many items of ``needle`` appear, in order, in ``haystack`` (greedy, which is optimal)."""
    it = iter(haystack)
    return sum(1 for _ in itertools.takewhile(lambda item: item in it, needle))


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    return _matched_prefix(needle, haystack) == len(needle)


def _diff(expected: Sequence[str], actual: Sequence[str]) -> list[str]:
    """Line diff of the two sequences: ``"  x"`` kept, ``"- x"`` missing, ``"+ x"`` extra."""
    lines: list[str] = []
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            lines += [f"  {name}" for name in expected[i1:i2]]
            continue
        lines += [f"- {name}" for name in expected[i1:i2]]
        lines += [f"+ {name}" for name in actual[j1:j2]]
    return lines


@dataclass(frozen=True)
class ToolSequence:
    """Compare the tools the run called with ``expected``.

    ``exact``: the same tools, same order, nothing extra. ``in_order``: the
    expected tools appear in that order, with other calls allowed in between.
    A tool call the cassette could not serve still counts as called: the agent
    did try to make it.
    """

    expected: tuple[str, ...]
    mode: SequenceMode = "exact"

    @property
    def name(self) -> str:
        return "tool_sequence"

    def __call__(self, run: RunResult, /) -> CheckResult:
        actual = run.trajectory.tool_names()
        expected = list(self.expected)
        exact = self.mode == "exact"
        passed = actual == expected if exact else _is_subsequence(expected, actual)
        details: dict[str, JsonValue] = {
            "mode": self.mode,
            "expected": list(expected),
            "actual": list(actual),
        }
        if passed:
            message = f"{len(actual)} tool call(s) match the expected sequence ({self.mode})"
        elif exact:
            diff = _diff(expected, actual)
            details["diff"] = list(diff)
            # Leading unchanged lines are the common prefix, so this is a position
            # in both sequences.
            first = next(i for i, line in enumerate(diff) if not line.startswith("  "))
            missing = [line[2:] for line in diff if line.startswith("- ")]
            extra = [line[2:] for line in diff if line.startswith("+ ")]
            parts = [f"tool call #{first + 1} differs"]
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if extra:
                parts.append(f"unexpected {', '.join(extra)}")
            message = "; ".join(parts)
        else:
            matched = _matched_prefix(expected, actual)
            details["missing"] = list(expected[matched:])
            after = f" after {expected[matched - 1]}" if matched else ""
            message = f"missing {', '.join(expected[matched:])} in order{after}"
        return CheckResult(check=self.name, passed=passed, message=message, details=details)


def tool_sequence(expected: Sequence[str], *, mode: SequenceMode = "exact") -> ToolSequence:
    """Build a ``tool_sequence`` check. See :class:`ToolSequence`."""
    if isinstance(expected, str):
        raise TypeError("expected must be a sequence of tool names, not a single string")
    return ToolSequence(expected=tuple(expected), mode=mode)
