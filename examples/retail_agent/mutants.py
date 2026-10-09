"""Deliberately broken variants of the retail agent. Replay checks must catch them."""

from __future__ import annotations

from anthropic.types import ToolResultBlockParam, ToolUseBlock

from examples.retail_agent.agent import RetailAgent, call_tool, error_result, execute


class SkipsOrderLookup(RetailAgent):
    """Never looks orders up (drops a verification step) but still takes the write actions."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        return [
            error_result(b.id, "lookup skipped") if b.name == "get_order_details" else execute(b)
            for b in blocks
        ]


# One argument changed per write tool: the kind of slip a prompt or parsing change causes.
_SWAPPED_REASON = {
    "no longer needed": "ordered by mistake",
    "ordered by mistake": "no longer needed",
}


def _corrupt(name: str, arguments: dict[str, object]) -> dict[str, object]:
    args = dict(arguments)
    if name == "cancel_pending_order" and isinstance(args.get("reason"), str):
        args["reason"] = _SWAPPED_REASON.get(str(args["reason"]), "no longer needed")
    elif name == "modify_pending_order_address":
        args["address2"] = ""
    elif name in ("return_delivered_order_items", "exchange_delivered_order_items"):
        args["payment_method_id"] = "gift_card_0000000"
    return args


class WrongWriteArgument(RetailAgent):
    """Changes one argument of every write action before calling the tool."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        return [call_tool(b.id, b.name, _corrupt(b.name, b.input)) for b in blocks]


class FirstToolOnly(RetailAgent):
    """Handles only the first tool call of each turn, a classic parallel-tool-use bug."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        return [execute(blocks[0])]


class NoPromptCache(RetailAgent):
    """Drops prompt caching: same behaviour, every turn pays full price for the prefix."""

    prompt_cache = False


skips_order_lookup = SkipsOrderLookup().run
wrong_write_argument = WrongWriteArgument().run
first_tool_only = FirstToolOnly().run
no_prompt_cache = NoPromptCache().run
