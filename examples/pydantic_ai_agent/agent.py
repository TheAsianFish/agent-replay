"""A tool-using pydantic-ai agent: a travel desk that searches, books and cancels flights.

Nothing here knows about recording except the ``@tool`` decorator on the tools
(:mod:`.tools`). The model client is injected the way pydantic-ai documents it,
through ``AnthropicProvider(anthropic_client=...)``; pydantic-ai then calls
``client.beta.messages.create``, which is the surface Replay records. Recording
on a local model needs nothing paid: Ollama serves the Anthropic Messages API,
so the same ``AsyncAnthropic`` points at it with ``base_url``::

    client = replay.wrap(AsyncAnthropic(base_url="http://localhost:11434", api_key="ollama"))
    agent = build_agent(client)

``replay record`` / ``replay ci`` / ``replay flake`` wrap the client themselves
(wrapping twice is a no-op), so :func:`run` takes it as given.
"""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from examples.pydantic_ai_agent.tools import ALL_TOOLS, reset

MODEL = "qwen2.5:7b"
"""The local model the committed live cassettes were recorded on, through Ollama."""

INSTRUCTIONS = """\
You are the travel desk of a small agency. Use the tools to answer; never guess
flight numbers, prices or booking details. The traveller writes once and cannot
answer questions: their message holds everything you need, including their
consent to the action they ask for. Look a booking up before you change it.
Give a tool only values from the traveller's message or from earlier tool
results: when a call needs another call's result, wait for that result.
Airports are IATA codes (SFO) and dates are YYYY-MM-DD. Finish with one short
sentence that states the result, with any amount and its currency.
"""

SETTINGS = AnthropicModelSettings(max_tokens=1024)
"""Bounded output; sampling is left at the model's defaults, as a team would ship it, so
``replay flake`` measures the run-to-run variation the agent really has."""


def ollama_url() -> str:
    return os.environ.get("REPLAY_OLLAMA_URL", "http://localhost:11434")


def make_client() -> AsyncAnthropic:
    """The live client for ``--client``: the Anthropic SDK pointed at a local Ollama server."""
    return AsyncAnthropic(base_url=ollama_url(), api_key="ollama", max_retries=0, timeout=600.0)


def build_agent(client: AsyncAnthropic, model: str = MODEL) -> Agent[None, str]:
    provider = AnthropicProvider(anthropic_client=client)
    return Agent(
        AnthropicModel(model, provider=provider),
        instructions=INSTRUCTIONS,
        tools=ALL_TOOLS,
        model_settings=SETTINGS,
    )


async def run(client: AsyncAnthropic, task: str) -> str:
    """One task, start to finish, from the same bookings every time."""
    reset()
    result = await build_agent(client).run(task)
    return result.output


async def run_streaming(client: AsyncAnthropic, task: str) -> str:
    """:func:`run` through ``agent.run_stream``: the same requests with ``stream=True``."""
    reset()
    async with build_agent(client).run_stream(task) as result:
        return await result.get_output()
