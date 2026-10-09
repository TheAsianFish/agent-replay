"""Domain rules over a run's tool calls, for use as ``rule`` checks.

Each rule takes the run's tool calls as ``(name, arguments)`` pairs in call
order (:func:`tool_calls` extracts them from a ``RunResult``) and returns
``(passed, message)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_replay.runner import RunResult
from examples.retail_agent.tasks import Action
from examples.retail_agent.tools import TERMINAL_TOOLS, WRITE_TOOLS

ToolCalls = Sequence[tuple[str, Mapping[str, Any]]]
Verdict = tuple[bool, str]

AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})


def tool_calls(run: RunResult) -> list[tuple[str, dict[str, Any]]]:
    """The run's tool calls as ``(name, arguments)`` pairs, in the order they started."""
    calls = []
    for step in run.trajectory.tool_steps():
        arguments = step.request if isinstance(step.request, dict) else {}
        calls.append((step.name, dict(arguments)))
    return calls


def _normalised(name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if name in TERMINAL_TOOLS:
        return name, {}  # a transfer's summary is free text; only the transfer itself counts
    # The tools store item lists sorted, so their order does not change the outcome.
    return name, {k: sorted(v) if isinstance(v, list) else v for k, v in arguments.items()}


def write_actions_match(calls: ToolCalls, expected_actions: Sequence[Action]) -> Verdict:
    """The consequential calls (writes and transfers) equal tau-bench's ground truth, in order."""
    consequential = WRITE_TOOLS | TERMINAL_TOOLS
    actual = [_normalised(n, a) for n, a in calls if n in consequential]
    expected = [_normalised(a.name, a.arguments) for a in expected_actions]
    if actual == expected:
        return True, f"{len(actual)} consequential action(s) as expected"
    for i, (got, want) in enumerate(zip(actual, expected, strict=False)):
        if got != want:
            return False, f"action #{i + 1}: expected {want}, got {got}"
    if len(actual) > len(expected):
        return False, f"unexpected extra action {actual[len(expected)]}"
    return False, f"missing action {expected[len(actual)]}"


def authenticated_before_write(calls: ToolCalls) -> Verdict:
    """The user was looked up by email or name + zip before any write."""
    for name, _ in calls:
        if name in AUTH_TOOLS:
            return True, f"authenticated with {name}"
        if name in WRITE_TOOLS:
            return False, f"{name} called before authenticating the user"
    return True, "no write actions"


def order_checked_before_write(calls: ToolCalls) -> Verdict:
    """Every written order was fetched with get_order_details before the write."""
    seen: set[object] = set()
    for name, arguments in calls:
        if name == "get_order_details":
            seen.add(arguments.get("order_id"))
        elif name in WRITE_TOOLS and arguments.get("order_id") not in seen:
            return False, f"{name} on {arguments.get('order_id')} without checking the order first"
    return True, "every written order was checked first"


def no_write_actions(calls: ToolCalls) -> Verdict:
    """Nothing in the database was changed (for lookups and refusals)."""
    writes = [name for name, _ in calls if name in WRITE_TOOLS]
    return (not writes, f"unexpected write action(s): {writes}" if writes else "no writes")


def outputs_match(output: object, expected_outputs: Sequence[str]) -> Verdict:
    """tau-bench's output check: each expected string occurs in the answer (case and commas
    ignored)."""
    answer = str(output or "").lower().replace(",", "")
    missing = [o for o in expected_outputs if o.lower() not in answer]
    return (not missing, f"answer lacks {missing}" if missing else "all expected outputs present")
