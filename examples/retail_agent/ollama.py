"""The retail agent on a local model through Ollama (M18).

Same prompt, tools and loop as :mod:`examples.retail_agent.agent`; only the model
name and the client differ. Ollama serves the Anthropic Messages API, so the agent's
``Anthropic`` client points at it with ``base_url``, as the pydantic-ai example's does::

    uv run replay record examples.retail_agent.ollama:run \\
        --client examples.retail_agent.ollama:make_client -t "..." -o out.jsonl

The model is ``$REPLAY_OLLAMA_MODEL`` (default :data:`MODEL`) and the server
``$REPLAY_OLLAMA_URL`` (default ``http://localhost:11434``).
"""

from __future__ import annotations

import os

from anthropic import Anthropic

from examples.retail_agent.agent import RetailAgent

MODEL = "qwen2.5:7b"


def ollama_model() -> str:
    """The model a live run talks to: ``$REPLAY_OLLAMA_MODEL``, else :data:`MODEL`."""
    return os.environ.get("REPLAY_OLLAMA_MODEL") or MODEL


def ollama_url() -> str:
    return os.environ.get("REPLAY_OLLAMA_URL", "http://localhost:11434")


def make_client() -> Anthropic:
    # A 7B model on CPU takes minutes to read the retail policy on a cold cache.
    return Anthropic(base_url=ollama_url(), api_key="ollama", max_retries=0, timeout=900.0)


class OllamaRetailAgent(RetailAgent):
    """The retail agent with its model taken from the environment when it runs."""

    @property
    def model(self) -> str:  # type: ignore[override]  # read at run time, not import time
        return ollama_model()


run = OllamaRetailAgent().run
