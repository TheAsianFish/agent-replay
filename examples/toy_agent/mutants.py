"""Deliberately broken variants of the toy agent. Replay checks must catch them."""

from __future__ import annotations

from anthropic import Anthropic
from anthropic.types import ToolResultBlockParam, ToolUseBlock

from examples.toy_agent.agent import SupportAgent, execute


class FirstToolOnly(SupportAgent):
    """Handles only the first tool call of each turn, a classic parallel-tool-use bug."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        return [execute(blocks[0])]


class CrashAfterAnswer(SupportAgent):
    """Does everything right, then raises: the tool sequence is fine but the run is not."""

    def run(self, client: Anthropic, task: str) -> str:
        answer = super().run(client, task)
        raise RuntimeError(f"crashed after answering: {answer[:20]}...")


first_tool_only = FirstToolOnly().run
crash_after_answer = CrashAfterAnswer().run
