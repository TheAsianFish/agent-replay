"""A deterministic local model for the Ollama toy agent, so it records without a server.

It reuses the Anthropic toy model's policy unchanged: :class:`FakeOllamaRequest`
reads Ollama's ``role: tool`` messages through the same helpers.
"""

from __future__ import annotations

from agent_replay.fake_ollama import FakeOllama
from examples.toy_agent.fake_model import policy


def make_client() -> FakeOllama:
    return FakeOllama(policy)
