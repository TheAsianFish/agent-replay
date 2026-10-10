"""Simulated prompt and model regressions: the retail fake model with one behaviour changed.

Each mutation wraps :func:`examples.retail_agent.fake_model.policy` and rewrites
the :class:`Reply` it would have given, the way a prompt edit or a model upgrade
changes what the model does while the agent code stays the same. Each has a
``module:factory`` client (``FakeAnthropic(mutated).as_anthropic()``) for
``model-live`` replays.

In ``full`` and ``tools-live`` the model's replies come from the cassette, so
these mutations cannot show there: that is what the cut point means, not a gap
to paper over.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from typing import Any

from anthropic import Anthropic

from agent_replay._json import to_jsonable
from agent_replay.fake import FakeAnthropic, FakeRequest, Policy, Reply, ToolCall
from examples.retail_agent.agent import TOOLS
from examples.retail_agent.fake_model import policy
from examples.retail_agent.tools import WRITE_TOOLS

Record = dict[str, Any]
Transform = Callable[[FakeRequest, Reply], Reply]

_ITEM_TOOLS = frozenset({"return_delivered_order_items", "exchange_delivered_order_items"})


def mutated(transform: Transform) -> Policy:
    """The fake model's policy with ``transform`` applied to every reply."""

    def mutated_policy(request: FakeRequest) -> Reply:
        return transform(request, policy(request))

    return mutated_policy


def _client(transform: Transform) -> Anthropic:
    return FakeAnthropic(mutated(transform)).as_anthropic()


def _rewrite(reply: Reply, fn: Callable[[ToolCall], Iterable[ToolCall]]) -> Reply:
    """``reply`` with each tool call replaced by what ``fn`` makes of it."""
    return Reply(text=reply.text, tool_calls=[new for call in reply.tool_calls for new in fn(call)])


def _results(request: FakeRequest, name: str) -> list[Record]:
    """The successful results of tool ``name`` so far, parsed."""
    return [
        json.loads(r.content) for r in request.tool_results() if r.name == name and not r.is_error
    ]


def _order(request: FakeRequest, order_id: object) -> Record | None:
    return next(
        (o for o in _results(request, "get_order_details") if o["order_id"] == order_id), None
    )


def _with_input(call: ToolCall, **changes: Any) -> ToolCall:
    return ToolCall(call.name, {**call.input, **changes})


# --- the model skips a call it should make -----------------------------------------


def _privately(call: ToolCall) -> tuple[str, bool]:
    """Run ``call``'s tool function directly (no session sees it): what the model
    "already knows" when it skips the call."""
    try:
        result = TOOLS[call.name].fn(**call.input)
    except ValueError as exc:
        return f"Error: {exc}", True
    return (result if isinstance(result, str) else json.dumps(to_jsonable(result))), False


def _knowing(request: FakeRequest, known: list[tuple[ToolCall, tuple[str, bool]]]) -> FakeRequest:
    """``request`` as if the ``known`` calls had been made right after the user's message."""
    if not known:
        return request
    uses = [
        {"type": "tool_use", "id": f"toolu_known_{i}", "name": c.name, "input": dict(c.input)}
        for i, (c, _) in enumerate(known)
    ]
    results = [
        {"type": "tool_result", "tool_use_id": f"toolu_known_{i}", "content": text, "is_error": err}
        for i, (_, (text, err)) in enumerate(known)
    ]
    messages = [
        request.messages[0],
        {"role": "assistant", "content": uses},
        {"role": "user", "content": results},
        *request.messages[1:],
    ]
    return FakeRequest(request.model, request.system, messages, request.tools)


def skipping(names: frozenset[str]) -> Policy:
    """The fake model, never calling the tools in ``names`` but acting as if it had."""

    def skipping_policy(request: FakeRequest) -> Reply:
        known: list[tuple[ToolCall, tuple[str, bool]]] = []
        for _ in range(10):  # each round answers at least one skipped call
            reply = policy(_knowing(request, known))
            skipped = [c for c in reply.tool_calls if c.name in names]
            if not skipped:
                return reply
            known += [(c, _privately(c)) for c in skipped]
            kept = [c for c in reply.tool_calls if c.name not in names]
            if kept:
                return Reply(text=reply.text, tool_calls=kept)
        raise AssertionError("the skipping policy did not converge")

    return skipping_policy


def skips_authentication() -> Anthropic:
    return FakeAnthropic(
        skipping(frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"}))
    ).as_anthropic()


def skips_order_check() -> Anthropic:
    return FakeAnthropic(skipping(frozenset({"get_order_details"}))).as_anthropic()


# --- the model gets an argument wrong ---------------------------------------------

_OTHER_REASON = {"no longer needed": "ordered by mistake", "ordered by mistake": "no longer needed"}


def wrong_cancel_reason() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        return _rewrite(
            reply,
            lambda c: [
                _with_input(c, reason=_OTHER_REASON[str(c.input["reason"])])
                if c.name == "cancel_pending_order"
                else c
            ],
        )

    return _client(transform)


def product_ids_as_item_ids() -> Anthropic:
    def transform(request: FakeRequest, reply: Reply) -> Reply:
        def fix(call: ToolCall) -> list[ToolCall]:
            order = _order(request, call.input.get("order_id"))
            if call.name not in _ITEM_TOOLS or order is None:
                return [call]
            products = {i["item_id"]: i["product_id"] for i in order["items"]}
            item_ids = call.input["item_ids"]
            assert isinstance(item_ids, list)
            return [_with_input(call, item_ids=[products[str(i)] for i in item_ids])]

        return _rewrite(reply, fix)

    return _client(transform)


def refunds_to_first_method() -> Anthropic:
    def transform(request: FakeRequest, reply: Reply) -> Reply:
        users = _results(request, "get_user_details")
        if not users:
            return reply
        first = next(iter(users[-1]["payment_methods"]))
        return _rewrite(
            reply,
            lambda c: [_with_input(c, payment_method_id=first) if c.name in _ITEM_TOOLS else c],
        )

    return _client(transform)


def suite_in_address1() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        def fix(call: ToolCall) -> list[ToolCall]:
            if call.name != "modify_pending_order_address":
                return [call]
            line = f"{call.input['address1']}, {call.input['address2']}"
            return [_with_input(call, address1=line, address2="")]

        return _rewrite(reply, fix)

    return _client(transform)


def wrong_exchange_variant() -> Anthropic:
    def transform(request: FakeRequest, reply: Reply) -> Reply:
        products = _results(request, "get_product_details")

        def fix(call: ToolCall) -> list[ToolCall]:
            if call.name != "exchange_delivered_order_items" or not products:
                return [call]
            old, new = call.input["item_ids"], call.input["new_item_ids"]
            assert isinstance(old, list)
            assert isinstance(new, list)
            # Another available variant of the product: the right product, the wrong options.
            other = next(
                v["item_id"]
                for v in products[-1]["variants"].values()
                if v["available"] and v["item_id"] not in (*old, *new)
            )
            return [_with_input(call, new_item_ids=[other])]

        return _rewrite(reply, fix)

    return _client(transform)


# --- the model takes the wrong action ---------------------------------------------


def transfers_instead_of_acting() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        return _rewrite(
            reply,
            lambda c: [
                ToolCall("transfer_to_human_agents", {"summary": "The user wants a change made."})
                if c.name in WRITE_TOOLS
                else c
            ],
        )

    return _client(transform)


def transfers_after_acting() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        return _rewrite(
            reply,
            lambda c: (
                [
                    c,
                    ToolCall(
                        "transfer_to_human_agents", {"summary": "Change made; please review."}
                    ),
                ]
                if c.name in WRITE_TOOLS
                else [c]
            ),
        )

    return _client(transform)


def claims_without_acting() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        if not any(c.name in WRITE_TOOLS for c in reply.tool_calls):
            return reply
        return Reply(text="Done: your request has been processed.")

    return _client(transform)


def repeats_lookups() -> Anthropic:
    def transform(request: FakeRequest, reply: Reply) -> Reply:
        if reply.tool_calls:
            return reply
        found = [
            r
            for r in request.tool_results()
            if r.name.startswith("find_user_id") and not r.is_error
        ]
        if not found:
            return reply
        return Reply(
            text="Let me double-check your account.",
            tool_calls=[ToolCall("get_user_details", {"user_id": found[-1].content})],
        )

    return _client(transform)


# --- the model misreports ---------------------------------------------------------


def rounds_reported_total() -> Anthropic:
    def transform(_: FakeRequest, reply: Reply) -> Reply:
        if reply.text is None or reply.tool_calls:
            return reply
        return Reply(text=re.sub(r"\$(\d+)\.\d\d", r"$\1", reply.text))

    return _client(transform)
