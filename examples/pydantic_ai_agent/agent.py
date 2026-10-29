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
from typing import Literal

from anthropic import AsyncAnthropic
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from examples.pydantic_ai_agent.tools import ALL_TOOLS, reset

MODEL = "qwen2.5:7b"
"""The local model the committed live cassettes were recorded on, through Ollama."""

GUARD = """\
Give a tool only values from the traveller's message or from earlier tool
results: when a call needs another call's result, wait for that result.
"""
"""The guard line M9 added after qwen2.5:7b called tools with values it had not read yet."""

_TEMPLATE = """\
You are the travel desk of a small agency. Use the tools to answer; never guess
flight numbers, prices or booking details. The traveller writes once and cannot
answer questions: their message holds everything you need, including their
consent to the action they ask for. Look a booking up before you change it.
{guard}Airports are IATA codes (SFO) and dates are YYYY-MM-DD. Finish with one short
sentence that states the result, with any amount and its currency.
"""

PROMPT_VARIANTS = {
    "recording": _TEMPLATE.format(guard=GUARD),
    "no_guard": _TEMPLATE.format(guard=""),
}
"""The instructions by name: ``recording`` is the prompt the cassettes were recorded with;
``no_guard`` is that prompt without :data:`GUARD`, the prompt change M15 replays."""
PromptVariant = Literal["recording", "no_guard"]
INSTRUCTIONS = PROMPT_VARIANTS["recording"]

SETTINGS = AnthropicModelSettings(max_tokens=1024)
"""Bounded output; sampling is left at the model's defaults, as a team would ship it, so
``replay flake`` measures the run-to-run variation the agent really has."""


def live_model() -> str:
    """The model a live run talks to: ``$REPLAY_OLLAMA_MODEL``, else :data:`MODEL`."""
    return os.environ.get("REPLAY_OLLAMA_MODEL") or MODEL


def ollama_url() -> str:
    return os.environ.get("REPLAY_OLLAMA_URL", "http://localhost:11434")


def make_client() -> AsyncAnthropic:
    """The live client for ``--client``: the Anthropic SDK pointed at a local Ollama server."""
    return AsyncAnthropic(base_url=ollama_url(), api_key="ollama", max_retries=0, timeout=600.0)


def build_agent(
    client: AsyncAnthropic,
    model: str | None = None,
    *,
    prompt_variant: PromptVariant = "recording",
    temperature: float | None = None,
) -> Agent[None, str]:
    """The travel desk on ``model`` (default :func:`live_model`). ``temperature=None``
    leaves sampling at the model's default, as the cassettes were recorded."""
    provider = AnthropicProvider(anthropic_client=client)
    settings = SETTINGS if temperature is None else {**SETTINGS, "temperature": temperature}
    return Agent(
        AnthropicModel(model or live_model(), provider=provider),
        instructions=PROMPT_VARIANTS[prompt_variant],
        tools=ALL_TOOLS,
        model_settings=settings,
    )


async def run(
    client: AsyncAnthropic,
    task: str,
    *,
    prompt_variant: PromptVariant = "recording",
    temperature: float | None = None,
) -> str:
    """One task, start to finish, from the same bookings every time."""
    reset()
    agent = build_agent(client, prompt_variant=prompt_variant, temperature=temperature)
    result = await agent.run(task)
    return result.output


async def run_no_guard(client: AsyncAnthropic, task: str) -> str:
    """:func:`run` with the ``no_guard`` prompt (a suite's ``agent`` takes no arguments)."""
    return await run(client, task, prompt_variant="no_guard")


async def run_temperature_0(client: AsyncAnthropic, task: str) -> str:
    """:func:`run` with greedy sampling, ``temperature=0``."""
    return await run(client, task, temperature=0.0)


async def run_streaming(client: AsyncAnthropic, task: str) -> str:
    """:func:`run` through ``agent.run_stream``: the same requests with ``stream=True``."""
    reset()
    async with build_agent(client).run_stream(task) as result:
        return await result.get_output()
