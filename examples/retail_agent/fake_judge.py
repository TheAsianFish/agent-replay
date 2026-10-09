"""A deterministic stand-in for the LLM judge on the retail rubric, for offline tests.

``make_judge()`` returns a :class:`~agent_replay.judge.FakeJudge` that decides
each criterion of ``rubric.yaml`` from the transcript with fixed rules, so a
suite can exercise the ``judge`` check (caching, gating, reporting) without a
model. It is a test double, not a substitute for the real judge: it knows
nothing the rules in :mod:`examples.retail_agent.rules` do not.

Use it from a suite with ``provider = "examples.retail_agent.fake_judge:make_judge"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_replay.judge import FakeJudge, JudgeInput, Rubric
from examples.retail_agent.rules import authenticated_before_write, order_checked_before_write
from examples.retail_agent.tools import TERMINAL_TOOLS, WRITE_TOOLS

Decision = bool | tuple[bool, str]


def _calls(judge_input: JudgeInput) -> list[tuple[str, dict[str, Any]]]:
    return [
        (c.name, dict(c.input) if isinstance(c.input, dict) else {}) for c in judge_input.tool_calls
    ]


def _resolved(judge_input: JudgeInput) -> Decision:
    if judge_input.error is not None:
        return False, f"the agent did not finish ({judge_input.error.type})"
    failed = [
        c.name
        for c in judge_input.tool_calls
        if c.name in WRITE_TOOLS | TERMINAL_TOOLS and c.error is not None
    ]
    if failed:
        return False, f"consequential call(s) failed: {', '.join(failed)}"
    return True, "the agent finished and every consequential call succeeded"


def _policy_followed(judge_input: JudgeInput) -> Decision:
    calls = _calls(judge_input)
    for verdict in (authenticated_before_write(calls), order_checked_before_write(calls)):
        if not verdict[0]:
            return verdict
    return True, "authenticated and fetched every order before acting on it"


def _accurate_report(judge_input: JudgeInput) -> Decision:
    if not isinstance(judge_input.output, str) or not judge_input.output.strip():
        return False, "there is no final reply"
    return _resolved(judge_input)


_RULES = {
    "resolved": _resolved,
    "policy_followed": _policy_followed,
    "accurate_report": _accurate_report,
}


def decide(rubric: Rubric, judge_input: JudgeInput) -> Mapping[str, Decision]:
    """One decision per rubric criterion; criteria without a rule pass."""
    return {
        criterion_id: _RULES[criterion_id](judge_input)
        if criterion_id in _RULES
        else (True, "no rule for this criterion in the fake judge")
        for criterion_id in rubric.ids
    }


def make_judge() -> FakeJudge:
    return FakeJudge(decide, name="retail-rules")
