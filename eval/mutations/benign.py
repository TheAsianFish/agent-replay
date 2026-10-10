"""Benign refactors of the retail agent: changes a team makes that keep its behaviour.

The mutation study replays the suite against each in every configuration; any
failure is a false alarm. Some change the bytes of a request (key order,
whitespace), so the replay serves those steps ``approximate`` rather than
``exact``; that is reported, and is not a failure unless a suite sets
``fail_on_approximate``.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic, AsyncAnthropic, omit
from anthropic.types import Message, MessageParam, ToolResultBlockParam, ToolUseBlock

from agent_replay import ContextThreadPoolExecutor
from agent_replay._json import to_jsonable
from agent_replay.fake import FakeAsyncAnthropic
from examples.retail_agent.agent import MODEL, SYSTEM, RetailAgent, error_result, final_text
from examples.retail_agent.fake_model import policy
from examples.retail_agent.tools import reset_db


class ExtractedLoop(RetailAgent):
    """The loop body split into ``ask`` (one model call) and ``step`` (one turn)."""

    def ask(self, client: Anthropic, messages: list[MessageParam]) -> Message:
        return client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=[t.to_anthropic() for t in self.tools.values()],
            messages=messages,
            output_config={"effort": "low"},
            cache_control={"type": "ephemeral"} if self.prompt_cache else omit,
        )

    def step(self, client: Anthropic, messages: list[MessageParam]) -> str | None:
        """One turn; the final answer when the model is done, else ``None``."""
        response = self.ask(client, messages)
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if response.stop_reason != "tool_use" or not tool_uses:
            return final_text(response.content)
        messages.append({"role": "user", "content": self.handle_tool_uses(tool_uses)})
        return None

    def run(self, client: Anthropic, task: str) -> str:
        reset_db()
        messages: list[MessageParam] = [{"role": "user", "content": task}]
        for _ in range(self.max_turns):
            answer = self.step(client, messages)
            if answer is not None:
                return answer
        raise RuntimeError(f"agent did not finish within {self.max_turns} turns")


class ParallelTools(RetailAgent):
    """Runs one turn's tool calls concurrently, results kept in request order."""

    def handle_tool_uses(self, blocks: list[ToolUseBlock]) -> list[ToolResultBlockParam]:
        with ContextThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(self.execute, blocks))


class CompactToolJson(RetailAgent):
    """Serialises tool results as compact JSON with sorted keys (same data, other bytes)."""

    def call_tool(
        self, tool_use_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolResultBlockParam:
        tool = self.tools.get(name)
        if tool is None:
            return error_result(tool_use_id, f"unknown tool {name}")
        try:
            result = tool(**arguments)
        except Exception as exc:  # every tool failure is reported to the model, not raised
            return error_result(tool_use_id, exc)
        content = (
            result
            if isinstance(result, str)
            else json.dumps(to_jsonable(result), sort_keys=True, separators=(",", ":"))
        )
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}

    def execute(self, block: ToolUseBlock) -> ToolResultBlockParam:
        return self.call_tool(block.id, block.name, block.input)


class MoreTurns(RetailAgent):
    """A higher turn limit, for longer conversations."""

    max_turns = 20


class AsyncRetailAgent(RetailAgent):
    """The same loop on ``AsyncAnthropic`` (tools stay synchronous)."""

    async def run_async(self, client: AsyncAnthropic, task: str) -> str:
        reset_db()
        messages: list[MessageParam] = [{"role": "user", "content": task}]
        for _ in range(self.max_turns):
            response = await client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                tools=[t.to_anthropic() for t in self.tools.values()],
                messages=messages,
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"} if self.prompt_cache else omit,
            )
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
            if response.stop_reason != "tool_use" or not tool_uses:
                return final_text(response.content)
            messages.append({"role": "user", "content": self.handle_tool_uses(tool_uses)})
        raise RuntimeError(f"agent did not finish within {self.max_turns} turns")


def make_async_client() -> AsyncAnthropic:
    """The retail fake model behind ``await client.messages.create``, for ``model-live``."""
    return FakeAsyncAnthropic(policy).as_anthropic()


extracted_loop = ExtractedLoop().run
parallel_tools = ParallelTools().run
compact_tool_json = CompactToolJson().run
more_turns = MoreTurns().run
async_agent = AsyncRetailAgent().run_async
