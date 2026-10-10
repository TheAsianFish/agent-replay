"""The retail mutation suite (PLAN.md D4): known-bad variants of ``examples/retail_agent``
and benign refactors of it, for ``eval/mutation_study.py``.

A mutation is broken by construction and targets one layer:

- ``agent``: the orchestration code (``agents.py``, plus the three seeded mutants
  in ``examples/retail_agent/mutants.py``);
- ``model``: a simulated prompt or model regression, the fake model with one
  behaviour changed (``models.py``), served as the ``model-live`` client;
- ``tool``: one tool's code, same name and schema, wrong behaviour (``tools.py``).

Each entry names the agent and the client to replay with (``module:attr``).
The client only matters where the model runs live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Target = Literal["agent", "model", "tool"]

REAL_AGENT = "examples.retail_agent.agent:run"
FAKE_MODEL = "examples.retail_agent.fake_model:make_client"


@dataclass(frozen=True)
class Mutation:
    id: str
    target: Target
    description: str
    """The realistic bug it simulates, in one line."""
    agent: str = REAL_AGENT
    client: str = FAKE_MODEL


@dataclass(frozen=True)
class Refactor:
    """A behaviour-preserving change: any failure it causes is a false alarm."""

    id: str
    description: str
    agent: str
    client: str = FAKE_MODEL


def _agent(id: str, description: str, module: str = "eval.mutations.agents") -> Mutation:
    return Mutation(id, "agent", description, agent=f"{module}:{id}")


def _model(id: str, description: str) -> Mutation:
    return Mutation(id, "model", description, client=f"eval.mutations.models:{id}")


def _tool(id: str, description: str) -> Mutation:
    return Mutation(id, "tool", description, agent=f"eval.mutations.tools:{id}")


_SEEDED = "examples.retail_agent.mutants"

MUTATIONS: tuple[Mutation, ...] = (
    # --- agent: orchestration code ---
    _agent(
        "skips_order_lookup",
        "answers get_order_details with an error instead of running it",
        _SEEDED,
    ),
    _agent(
        "wrong_write_argument", "changes one argument of every write before calling it", _SEEDED
    ),
    _agent("first_tool_only", "runs only the first tool call of each turn", _SEEDED),
    _agent("drops_error_flag", "reports failed tool calls without is_error"),
    _agent("truncates_tool_results", "cuts every tool result to 120 characters"),
    _agent("max_turns_too_low", "gives up after three model turns"),
    _agent("returns_empty_answer", "does the work but returns an empty final answer"),
    _agent("repeats_write_calls", "calls every write tool a second time after it succeeded"),
    _agent("mismatches_tool_use_ids", "pairs parallel tool results with the wrong tool_use_id"),
    _agent("unregisters_product_tool", "leaves get_product_details out of the tool table"),
    _agent("strips_policy", "sends the system prompt without the retail policy"),
    _agent("truncates_history", "sends only the last four messages to the model"),
    # --- model: prompt or model regressions ---
    _model("skips_authentication", "acts without looking the user up by email or name + zip"),
    _model("skips_order_check", "writes without fetching the order's details first"),
    _model("wrong_cancel_reason", "cancels with the other reason than the user gave"),
    _model("product_ids_as_item_ids", "passes product ids where item ids are expected"),
    _model("refunds_to_first_method", "refunds to the first payment method on file"),
    _model("suite_in_address1", "puts the suite into address line 1 and empties line 2"),
    _model("wrong_exchange_variant", "exchanges for another available variant than asked"),
    _model("transfers_instead_of_acting", "transfers to a human instead of taking the action"),
    _model("transfers_after_acting", "takes the action and then also transfers to a human"),
    _model("claims_without_acting", "says the request is done without calling the write tool"),
    _model("repeats_lookups", "re-reads the user's profile instead of answering, until cut off"),
    _model("rounds_reported_total", "rounds dollar amounts in the final answer to whole dollars"),
    # --- tool: tool code ---
    _tool("cancel_keeps_status", "cancel_pending_order refunds but never saves the status"),
    _tool("cancel_skips_gift_card_refund", "cancel_pending_order never credits the gift card"),
    _tool("address_drops_address2", "modify_pending_order_address drops address line 2"),
    _tool("return_ignores_item_ids", "return_delivered_order_items returns the whole order"),
    _tool(
        "exchange_flips_price_difference",
        "exchange_delivered_order_items records the price difference with the wrong sign",
    ),
    _tool("user_lookup_builds_id", "find_user_id_by_name_zip builds an id instead of looking up"),
    _tool("product_details_all_unavailable", "get_product_details marks every variant unavailable"),
    _tool("order_details_stale_status", "get_order_details reports every order as pending"),
)

_BENIGN = "eval.mutations.benign"

REFACTORS: tuple[Refactor, ...] = (
    Refactor(
        "extracted_loop", "loop body extracted into ask/step methods", f"{_BENIGN}:extracted_loop"
    ),
    Refactor(
        "parallel_tools",
        "one turn's tool calls run in a ContextThreadPoolExecutor, results in order",
        f"{_BENIGN}:parallel_tools",
    ),
    Refactor(
        "compact_tool_json",
        "tool results serialised with sorted keys and no whitespace",
        f"{_BENIGN}:compact_tool_json",
    ),
    Refactor("more_turns", "turn limit raised from 12 to 20", f"{_BENIGN}:more_turns"),
    Refactor(
        "async_agent",
        "the loop ported to AsyncAnthropic",
        f"{_BENIGN}:async_agent",
        client=f"{_BENIGN}:make_async_client",
    ),
)
