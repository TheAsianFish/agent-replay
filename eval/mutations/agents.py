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


drops_error_flag = DropsErrorFlag().run
truncates_tool_results = TruncatesToolResults().run
max_turns_too_low = MaxTurnsTooLow().run
returns_empty_answer = ReturnsEmptyAnswer().run
repeats_write_calls = RepeatsWriteCalls().run
mismatches_tool_use_ids = MismatchesToolUseIds().run
unregisters_product_tool = UnregistersProductTool().run
strips_policy = StripsPolicy().run
truncates_history = TruncatesHistory().run
