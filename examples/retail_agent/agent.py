"""The retail agent: a plain Anthropic tool-use loop over τ-bench's retail tools.

Like the toy agent, nothing here knows about recording or replay except the
``@tool`` decorator on the tools; the client it receives is wrapped by whoever
runs it. The tools act on ``tools.DB``; call ``tools.reset_db()`` between runs.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, ToolResultBlockParam, ToolUseBlock

from agent_replay import Tool
from agent_replay._json import to_jsonable
from examples.retail_agent.tools import ALL_TOOLS, DATA

MODEL = "claude-opus-5-5"

# Stable content first (the policy, then the adaptation note) so it can be cached.
POLICY = (DATA / "wiki.md").read_text(encoding="utf-8")
SINGLE_TURN_NOTE = """

## This deployment

- The user writes once and cannot answer follow-up questions. Their message states
  who they are and everything they want, and their confirmation (yes) of the
  actions they ask for is given in it.
- You may call several read-only tools at once. Take each consequential action
  once, after checking the order's status and the policy.
- Finish with one short sentence for the user.
"""
SYSTEM = POLICY + SINGLE_TURN_NOTE

TOOLS: dict[str, Tool[..., Any]] = {t.name: t for t in ALL_TOOLS}


class RetailAgent:
    max_turns = 12

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
                # Caches the growing prefix (tools, policy, earlier turns) for the next turn.
                cache_control={"type": "ephemeral"},
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
    return call_tool(block.id, block.name, block.input)


def call_tool(tool_use_id: str, name: str, arguments: dict[str, Any]) -> ToolResultBlockParam:
    """Run one tool; a failure becomes an ``is_error`` result worded like tau-bench's."""
    tool = TOOLS.get(name)
    if tool is None:
        return error_result(tool_use_id, f"unknown tool {name}")
    try:
        result = tool(**arguments)
    except Exception as exc:  # every tool failure is reported to the model, not raised
        return error_result(tool_use_id, exc)
    content = result if isinstance(result, str) else json.dumps(to_jsonable(result))
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def error_result(tool_use_id: str, error: object) -> ToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": f"Error: {error}",
        "is_error": True,
    }


def final_text(content: list[ContentBlock]) -> str:
    return " ".join(b.text for b in content if b.type == "text").strip()


run = RetailAgent().run
