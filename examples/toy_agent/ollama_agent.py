"""The toy support agent on a local model: the same loop and tools over Ollama's chat API.

``run`` is the agent; ``make_ollama_client`` returns a real ``ollama.Client``
(``OLLAMA_HOST`` picks the server) for recording against a local model, and
``ollama_fake.make_client`` a deterministic fake for offline recording.
Tool failures go back as ``role: tool`` messages starting with ``error:``,
since Ollama tool messages have no error flag.
"""

from __future__ import annotations

import json
from typing import Any

import ollama
from ollama import ChatResponse, Message

from agent_replay._json import to_jsonable
from examples.toy_agent.agent import SYSTEM, TOOLS

MODEL = "llama3.2"
OLLAMA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_anthropic()["input_schema"],
        },
    }
    for t in TOOLS.values()
]


class OllamaSupportAgent:
    max_turns = 8

    def run(self, client: ollama.Client, task: str) -> str:
        messages: list[dict[str, Any] | Message] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]
        for _ in range(self.max_turns):
            response: ChatResponse = client.chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS)
            messages.append(response.message)
            calls = response.message.tool_calls or []
            if not calls:
                return (response.message.content or "").strip()
            messages.extend(execute(call) for call in calls)
        raise RuntimeError(f"agent did not finish within {self.max_turns} turns")


def execute(call: Message.ToolCall) -> dict[str, Any]:
    name = call.function.name
    try:
        result = TOOLS[name](**call.function.arguments)
    except Exception as exc:
        return {"role": "tool", "tool_name": name, "content": f"error: {exc}"}
    return {"role": "tool", "tool_name": name, "content": json.dumps(to_jsonable(result))}


def make_ollama_client() -> ollama.Client:
    """A client for the local Ollama server (no request is made until the agent runs)."""
    return ollama.Client()


run = OllamaSupportAgent().run
