"""The toy support agent: a plain Anthropic tool-use loop.

Nothing here knows about recording or replay except the ``@tool`` decorator on
the tools; the client it receives is wrapped by whoever runs it.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, ToolResultBlockParam, ToolUseBlock

from agent_replay import Tool
from agent_replay._json import to_jsonable
from examples.toy_agent.tools import get_refund_policy, issue_refund, lookup_order

MODEL = "claude-opus-5-5"
SYSTEM = (
    "You are a customer-support agent for an online store. Before issuing a refund, "
    "look up the order and the refund policy (you may do both at once). Refund only "
    "delivered orders inside the policy window, for at most the order total. "
    "Finish with one short sentence for the customer."
)
TOOLS: dict[str, Tool[..., Any]] = {
    t.name: t for t in (lookup_order, get_refund_policy, issue_refund)
}


class SupportAgent:
    max_turns = 8

    def run(self, client: Anthropic, task: str) -> str:
        messages: list[MessageParam] = [{"role": "user", "content": task}]
        for _ in range(self.max_turns):
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                tools=[t.to_anthropic() for t in TOOLS.values()],
                messages=messages,
                output_config={"effort": "low"},
            )
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
            if response.stop_reason != "tool_use" or not tool_uses:
                return final_text(response.content)
            messages.append({"role": "user", "content": self.handle_tool_uses(tool_uses)})
        raise RuntimeError(f"agent did not finish within {self.max_turns} turns")

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        """Run every requested tool; all results go back in one user message."""
        return [execute(block) for block in blocks]


def execute(block: ToolUseBlock) -> ToolResultBlockParam:
    try:
        tool = TOOLS[block.name]
        result = tool(**block.input)
    except Exception as exc:
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"error: {exc}",
            "is_error": True,
        }
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(to_jsonable(result)),
    }


def final_text(content: list[ContentBlock]) -> str:
    return " ".join(b.text for b in content if b.type == "text").strip()


run = SupportAgent().run
