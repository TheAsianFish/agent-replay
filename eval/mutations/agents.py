"""Orchestration-code mutations: the retail agent with one bug in its own loop.

Each is a :class:`RetailAgent` subclass (the three seeded mutants in
``examples/retail_agent/mutants.py`` are reused as they are) whose ``run`` is
referenced from the catalogue. The model and the tools are untouched, so every
replay mode exercises the bug.
"""

from __future__ import annotations

from typing import ClassVar

from anthropic import Anthropic, omit
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock

from examples.retail_agent.agent import (
    MODEL,
    SINGLE_TURN_NOTE,
    SYSTEM,
    TOOLS,
    RetailAgent,
    ToolTable,
    final_text,
)
from examples.retail_agent.ollama import OllamaRetailAgent
from examples.retail_agent.tools import WRITE_TOOLS, reset_db


def _without_error_flag(result: ToolResultBlockParam) -> ToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": result["tool_use_id"],
        "content": result.get("content", ""),
    }


class DropsErrorFlag(RetailAgent):
    """Reports a failed tool as an ordinary result: ``is_error`` is lost in a refactor."""

    def execute(self, block: ToolUseBlock) -> ToolResultBlockParam:
        return _without_error_flag(super().execute(block))


class TruncatesToolResults(RetailAgent):
    """Cuts every tool result to 120 characters, as a careless context-saving change would."""

    def execute(self, block: ToolUseBlock) -> ToolResultBlockParam:
        result = super().execute(block)
        content = result.get("content", "")
        if isinstance(content, str):
            return {**result, "content": content[:120]}
        return result


class MaxTurnsTooLow(RetailAgent):
    """Gives up after three model turns; every task needs four or more."""

    max_turns = 3


class ReturnsEmptyAnswer(RetailAgent):
    """Does all the work, then loses the final answer (returns ``""``)."""

    def run(self, client: Anthropic, task: str) -> str:
        super().run(client, task)
        return ""


class RepeatsWriteCalls(RetailAgent):
    """Retries every write once more after it succeeded (a retry wrapper with the test inverted)."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        results = [self.execute(b) for b in blocks]
        for block in blocks:
            if block.name in WRITE_TOOLS:
                self.execute(block)
        return results


class MismatchesToolUseIds(RetailAgent):
    """Pairs parallel tool results with the wrong ``tool_use_id`` (ids taken in reverse)."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        results = [self.execute(b) for b in blocks]
        return [
            {**result, "tool_use_id": block.id}
            for result, block in zip(results, reversed(blocks), strict=True)
        ]


class UnregistersProductTool(RetailAgent):
    """Leaves ``get_product_details`` out of the agent's tool table (a registration slip)."""

    tools: ClassVar[ToolTable] = {
        name: t for name, t in TOOLS.items() if name != "get_product_details"
    }


class _CopiedLoop(RetailAgent):
    """``RetailAgent.run`` copied with two seams, for mutants that change the request."""

    def system(self) -> str:
        return SYSTEM

    def history(self, messages: list[MessageParam]) -> list[MessageParam]:
        return messages

    def run(self, client: Anthropic, task: str) -> str:
        reset_db()
        messages: list[MessageParam] = [{"role": "user", "content": task}]
        for _ in range(self.max_turns):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.system(),
                tools=[t.to_anthropic() for t in self.tools.values()],
                messages=self.history(messages),
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"} if self.prompt_cache else omit,
            )
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
            if response.stop_reason != "tool_use" or not tool_uses:
                return final_text(response.content)
            messages.append({"role": "user", "content": self.handle_tool_uses(tool_uses)})
        raise RuntimeError(f"agent did not finish within {self.max_turns} turns")


class StripsPolicy(_CopiedLoop):
    """Sends the deployment note without the retail policy in front of it."""

    def system(self) -> str:
        return SINGLE_TURN_NOTE.lstrip()


class TruncatesHistory(_CopiedLoop):
    """Sends only the last four messages, dropping the user's request once the
    conversation grows (a context-window "optimisation")."""

    def history(self, messages: list[MessageParam]) -> list[MessageParam]:
        return messages[-4:]


# --- reads of an entity the recording never saw (M11) ------------------------------
#
# Both change only a read's identifying argument, never a write's, so the request
# rules (every written order fetched first) still see the right ids. Without
# identifying arguments the replay serves the right recording to the wrong call.


class GuessesUserId(RetailAgent):
    """Fetches the profile with an id guessed from the name (``emma_smith``) instead of
    the one the lookup returned (``emma_smith_8564``)."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        return [
            self.call_tool(
                b.id, b.name, {**b.input, "user_id": str(b.input["user_id"]).rsplit("_", 1)[0]}
            )
            if b.name == "get_user_details"
            else self.execute(b)
            for b in blocks
        ]


def _mistyped(order_id: str) -> str:
    """``#W2417020`` -> ``#W2417021``: the last digit off by one."""
    return order_id[:-1] + str((int(order_id[-1]) + 1) % 10)


class ReadsWrongOrder(RetailAgent):
    """When it reads several orders at once, asks for the last one with its last digit off
    by one (an indexing slip in the code that fans the reads out)."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        reads = [b for b in blocks if b.name == "get_order_details"]
        last = reads[-1] if len(reads) > 1 else None
        return [
            self.call_tool(
                b.id, b.name, {**b.input, "order_id": _mistyped(str(b.input["order_id"]))}
            )
            if b is last
            else self.execute(b)
            for b in blocks
        ]


drops_error_flag = DropsErrorFlag().run
truncates_tool_results = TruncatesToolResults().run
max_turns_too_low = MaxTurnsTooLow().run
returns_empty_answer = ReturnsEmptyAnswer().run
repeats_write_calls = RepeatsWriteCalls().run
mismatches_tool_use_ids = MismatchesToolUseIds().run
unregisters_product_tool = UnregistersProductTool().run
strips_policy = StripsPolicy().run
truncates_history = TruncatesHistory().run
guesses_user_id = GuessesUserId().run
reads_wrong_order = ReadsWrongOrder().run


class OllamaDropsErrorFlag(OllamaRetailAgent, DropsErrorFlag):
    """:class:`DropsErrorFlag` on the local model (M18's live mutant arm)."""


drops_error_flag_ollama = OllamaDropsErrorFlag().run
