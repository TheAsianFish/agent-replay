"""Ground-truth rules for the travel desk, for use as ``rule`` checks.

Each takes the run and the task's name and returns ``(passed, message)``. They
judge the outcome (the consequential calls, the figure in the answer), not the
path, so a model that reaches the right result another way still passes; the
suite's ``in_order`` sequences hold the path to its essentials.
"""

from __future__ import annotations

import re
from typing import Any

from agent_replay.runner import RunResult
from examples.pydantic_ai_agent.tasks import TASKS
from examples.pydantic_ai_agent.tools import WRITE_TOOLS

Verdict = tuple[bool, str]


def _calls(run: RunResult) -> list[tuple[str, dict[str, Any]]]:
    return [
        (step.name, dict(step.request) if isinstance(step.request, dict) else {})
        for step in run.trajectory.tool_steps()
    ]


def _normalised(arguments: dict[str, Any]) -> dict[str, Any]:
    return {k: v.strip().upper() if isinstance(v, str) else v for k, v in arguments.items()}


def writes_match(run: RunResult, task: str) -> Verdict:
    """The bookings and cancellations equal the task's, in order (case and spacing ignored)."""
    expected = [(n, _normalised(a)) for n, a in TASKS[task].writes]
    actual = [(n, _normalised(a)) for n, a in _calls(run) if n in WRITE_TOOLS]
    if actual == expected:
        return True, f"{len(actual)} write(s) as expected"
    return False, f"expected writes {expected}, got {actual}"


def answer_states(run: RunResult, task: str) -> Verdict:
    """The final answer contains each of the task's figures (``B6416`` and ``173.88``, ...)."""
    output = run.output if isinstance(run.output, str) else ""
    text = output.replace(",", "")
    # 243 must not match 1243 or 24.3; 243.00 is fine.
    missing = [
        figure
        for figure in TASKS[task].answer
        if not re.search(rf"(?<![\w.]){re.escape(figure)}(?![\w]|\.\d*[1-9])", text)
    ]
    if not missing:
        return True, f"answer states {', '.join(TASKS[task].answer)}"
    return False, f"answer does not state {', '.join(missing)}: {output[:200]!r}"


def looked_up_before_writing(run: RunResult) -> Verdict:
    """Every write comes after a lookup (``search_flights`` or ``get_booking``)."""
    looked_up = False
    for name, _ in _calls(run):
        if name in ("search_flights", "get_booking"):
            looked_up = True
        elif name in WRITE_TOOLS and not looked_up:
            return False, f"{name} before any lookup"
    return True, "every write follows a lookup"
