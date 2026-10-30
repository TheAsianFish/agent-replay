"""Domain rules over a run's tool calls, for use as ``rule`` checks.

The predicates take the run's tool calls as ``(name, arguments)`` pairs in call
order (:func:`tool_calls` extracts them from a ``RunResult``) and returns
``(passed, message)``.
"""

from __future__ import annotations

import copy
import functools
import re
from collections.abc import Mapping, Sequence
from typing import Any

from agent_replay.runner import RunResult
from agent_replay.trajectory import StepStatus
from examples.retail_agent.tasks import ALL_TASKS, Action, expected_db_hash
from examples.retail_agent.tools import DB, DB_LOCK, TERMINAL_TOOLS, WRITE_TOOLS, db_hash

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


# --- rule checks: ``fn(run, **args)``, referenced from replay.toml ---------------


def matches_ground_truth(run: RunResult, task: str) -> Verdict:
    """τ-bench's ground truth for ``task``: its consequential actions, then its outputs."""
    expected = ALL_TASKS[task]
    verdict = write_actions_match(tool_calls(run), expected.expected_actions)
    if not verdict[0]:
        return verdict
    return outputs_match(run.output, expected.expected_outputs)


NOT_FOUND = re.compile(
    r"\b(?:could\s*n[o'\u2019]?t|can\s*not|can[\u2019']t|unable to|was\s*n[o'\u2019]?t able to|"
    r"did\s*n[o'\u2019]?t|failed to)\s+(?:find|locate|verify|identify|match)\s+(?:\w+\s+){0,3}?"
    r"(?:account|user|customer|record|identity|you)s?\b"
    r"|\b(?:account|user|customer|record)s?\b[^.!?]{0,30}?"
    r"\bnot (?:be )?(?:found|located|verified)\b"
    r"|\bno (?:matching |such )?(?:account|user|customer|record)s?\b"
    r"|\b(?:does|do)\s*n[o'\u2019]?t match (?:\w+\s+){0,2}?(?:account|user|customer|record)s?\b",
    re.IGNORECASE,
)
"""An answer that says the account could not be found: "I couldn't find your account",
"user not found", "no account matches". The failure must name the account, the user or
their identity, so "I couldn't find a laptop in your orders" or "the payment did not
match" (no write, but not this reply) is not read as one."""


def reports_account_not_found(run: RunResult) -> Verdict:
    """Nothing was written and the answer says the account was not found (M18's task)."""
    writes = [name for name, _ in tool_calls(run) if name in WRITE_TOOLS]
    if writes:
        return False, f"wrote {writes} for a user who was never found"
    if NOT_FOUND.search(str(run.output or "")) is None:
        return False, "the answer does not say the account was not found"
    return True, "no writes, and the answer says the account was not found"


def checked_before_writing(run: RunResult) -> Verdict:
    """The user was authenticated and every written order fetched before any write."""
    calls = tool_calls(run)
    for verdict in (authenticated_before_write(calls), order_checked_before_write(calls)):
        if not verdict[0]:
            return verdict
    return True, "authenticated and checked every order before writing"


# Steps that really ran the tool, so their effect on the database happened.
_RAN = frozenset({StepStatus.LIVE, StepStatus.FALLBACK})


@functools.cache
def _expected_db(task: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """The ground truth's database hash and tables for ``task``, computed once (leaving
    the database as it found it).

    Computing it resets and replays the shared database, so it holds ``DB_LOCK``:
    concurrent runs (``replay flake``) reset the database when they start, and a
    reset in the middle of the replay would cache a wrong expected state.
    """
    with DB_LOCK:
        saved = copy.deepcopy(DB.snapshot())
        try:
            digest = expected_db_hash(ALL_TASKS[task])
            return digest, copy.deepcopy(DB.snapshot())
        finally:
            DB.users, DB.orders, DB.products = saved["users"], saved["orders"], saved["products"]


def db_matches_ground_truth(run: RunResult, task: str) -> Verdict:
    """τ-bench's reward: the database after the run equals the one the ground truth leaves.

    Only meaningful when the run's tools really ran (``tools-live``, or every tool
    call performed under ``fallback``): a tool served from the cassette never
    touched the database. Then it passes vacuously and says so. It reads the
    retail ``DB`` as the run left it, so check the run before starting another;
    ``RetailAgent.run`` resets the database when it starts.
    """
    served = [s.name for s in run.trajectory.tool_steps() if s.status not in _RAN]
    if served:
        return True, (
            f"not applicable: {len(served)} tool call(s) were served from the cassette, "
            "so the run did not touch the database"
        )
    digest, expected = _expected_db(task)
    if db_hash() == digest:
        return True, "the database equals the ground truth's"
    actual = DB.snapshot()
    differing = [
        f"{table} {key}"
        for table in sorted(expected)
        for key in sorted(expected[table].keys() | actual[table].keys())
        if expected[table].get(key) != actual[table].get(key)
    ]
    return False, f"the database differs from the ground truth's in {', '.join(differing)}"
