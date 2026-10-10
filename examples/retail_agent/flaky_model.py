"""A simulated flaky model for the retail agent, so ``replay flake`` can be shown offline.

It is :func:`examples.retail_agent.fake_model.policy` with one slip a real
model makes now and then: on its first turn it asks for its first lookup twice.
Whether a run slips is drawn once per client from ``random.Random(seed)``,
with a probability that depends on the task (:data:`SLIP_RATES`), so the suite
has steady, flaky and broken scenarios side by side::

    uv run replay flake examples/retail_agent/replay.toml --no-store \\
        --client examples.retail_agent.flaky_model:make_client

``replay flake`` passes each run its own seed, so the outcomes are the same on
every invocation. Replayed with tools served from the cassette, the repeated
lookup has no recording left and the strict policy ends the run.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from anthropic import Anthropic

from agent_replay.fake import FakeAnthropic, FakeRequest, Reply
from examples.retail_agent.fake_model import policy
from examples.retail_agent.tasks import task_for

SLIP_RATES: dict[str, float] = {
    "cancel_pending_laptop": 0.0,
    "modify_address_suite": 0.0,
    "return_damaged_bicycle": 0.05,
    "exchange_waterproof_boots": 0.3,
    "lookup_latest_order_total": 0.0,
    "transfer_swapped_refunds": 1.0,
}
"""Probability that a run of each task slips; tasks not listed never do."""


def slipping_policy(draw: float) -> Callable[[FakeRequest], Reply]:
    """The fake model's policy, slipping on this run when ``draw`` is below the task's rate."""

    def flaky(request: FakeRequest) -> Reply:
        reply = policy(request)
        task = task_for(request.first_user_text())
        rate = SLIP_RATES.get(task.id, 0.0) if task is not None else 0.0
        first_turn = not request.tool_results()
        if draw < rate and first_turn and reply.tool_calls:
            return Reply(text=reply.text, tool_calls=[*reply.tool_calls, reply.tool_calls[0]])
        return reply

    return flaky


def make_client(seed: int = 0) -> Anthropic:
    draw = random.Random(seed).random()  # noqa: S311 - simulation, not cryptography
    return FakeAnthropic(slipping_policy(draw)).as_anthropic()
