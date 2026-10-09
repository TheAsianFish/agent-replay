"""Six τ-bench retail test tasks, adapted to a single turn.

tau-bench drives the agent with a simulated user over several turns; here the
instruction states the user's identity and the whole request up front, and the
user's confirmation of the write action is given in advance. The ground truth
(``expected_actions``, ``expected_outputs``) comes unchanged from tau-bench's
``tasks_test.py`` via ``data/tasks.json`` (rebuilt by ``vendor.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from examples.retail_agent.tools import (
    ALL_TOOLS,
    DATA,
    TERMINAL_TOOLS,
    WRITE_TOOLS,
    db_hash,
    reset_db,
)


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Task:
    id: str
    tau_bench_index: int
    """Index into tau-bench's retail ``TASKS_TEST``."""
    user_id: str
    instruction: str
    """The single-turn task text given to the agent."""
    source_instruction: str
    """tau-bench's instruction for its simulated user, kept for reference."""
    expected_actions: tuple[Action, ...]
    """tau-bench's ground-truth consequential actions (writes and transfers), in order."""
    expected_outputs: tuple[str, ...]
    """Strings the final answer must contain (tau-bench's ``outputs``)."""


# task id -> (tau-bench index, single-turn instruction)
_ADAPTED: dict[str, tuple[int, str]] = {
    "cancel_pending_laptop": (
        69,
        "I'm Emma Smith, zip code 10192. I want to return the laptop I ordered because I "
        "found a better deal elsewhere, for a full refund. If it can't be returned, cancel "
        "the order instead because I no longer need it. Yes, I confirm.",
    ),
    "modify_address_suite": (
        17,
        "I'm Fatima Johnson, zip code 78712. Please change order #W8665881 to be delivered "
        "to Suite 641 instead; the rest of the address stays the same. Yes, I confirm.",
    ),
    "return_damaged_bicycle": (
        53,
        "I'm Sofia Li, zip code 78260. The bicycle I received was damaged during delivery. "
        "I want to return it with the refund going to my original credit card. Yes, I confirm.",
    ),
    "exchange_waterproof_boots": (
        27,
        "I'm Isabella Johansson, zip code 32286. I want to exchange the hiking boots I "
        "received for the exact same item except that it is waterproof, paying any "
        "difference with the original payment method. Yes, I confirm.",
    ),
    "lookup_latest_order_total": (
        67,
        "I'm Noah Ito, zip code 98187. How much did I pay for the order I most recently placed?",
    ),
    "transfer_swapped_refunds": (
        10,
        "I'm Mia Garcia, mia.garcia2723@example.com. I want to return everything I ordered "
        "in both of my orders, with each order refunded to the other order's payment "
        "method. If that is not possible, transfer me to a human agent.",
    ),
}

_READ_TOOLS = frozenset(t.name for t in ALL_TOOLS) - WRITE_TOOLS - TERMINAL_TOOLS


def _load() -> dict[str, Task]:
    source = {
        t["tau_bench_index"]: t
        for t in json.loads((DATA / "tasks.json").read_text(encoding="utf-8"))
    }
    tasks: dict[str, Task] = {}
    for task_id, (index, instruction) in _ADAPTED.items():
        raw = source[index]
        tasks[task_id] = Task(
            id=task_id,
            tau_bench_index=index,
            user_id=raw["user_id"],
            instruction=instruction,
            source_instruction=raw["instruction"],
            expected_actions=tuple(
                Action(a["name"], a["arguments"])
                for a in raw["actions"]
                if a["name"] not in _READ_TOOLS
            ),
            expected_outputs=tuple(raw["outputs"]),
        )
    return tasks


TASKS: dict[str, Task] = _load()


def task_for(instruction: str) -> Task | None:
    """The task whose instruction is exactly ``instruction``, if any."""
    return next((t for t in TASKS.values() if t.instruction == instruction), None)


def expected_db_hash(task: Task) -> str:
    """The τ-bench reward target: the database after applying the ground-truth actions.

    Resets the database first and leaves it in the expected state.
    """
    tools = {t.name: t for t in ALL_TOOLS}
    reset_db()
    for action in task.expected_actions:
        tools[action.name](**action.arguments)
    return db_hash()
