"""Interception points: wrapped model clients and decorated tools."""

from __future__ import annotations

from typing import cast

from agent_replay.record._base import ProviderAdapter, WrappedClient
from agent_replay.record.anthropic import ANTHROPIC
from agent_replay.record.ollama import OLLAMA
from agent_replay.record.tools import Tool, tool

ADAPTERS: list[ProviderAdapter] = [ANTHROPIC, OLLAMA]
"""Registered providers, asked in order; the first that accepts a client wraps it."""


def wrap[C](client: C) -> C:
    """Return ``client`` with its model calls routed through the active session.

    Outside a session the proxy calls straight through. Wrapping twice is a
    no-op. Supports ``anthropic.Anthropic`` and ``AsyncAnthropic`` (and anything
    exposing the same ``messages.create``, such as
    :class:`agent_replay.fake.FakeAnthropic`). Raises :class:`TypeError` for
    anything else.
    """
    if isinstance(client, WrappedClient):
        return client
    for adapter in ADAPTERS:
        if adapter.accepts(client):
            return cast(C, adapter.wrap(client))
    names = ", ".join(a.name for a in ADAPTERS)
    raise TypeError(
        f"cannot wrap {type(client).__name__}: it is not a supported model client ({names})"
    )


def adapter_for_calls(names: list[str]) -> ProviderAdapter:
    """The provider that recorded model calls with these names (Anthropic if none did)."""
    for adapter in ADAPTERS:
        if any(adapter.owns(name) for name in names):
            return adapter
    return ANTHROPIC


__all__ = [
    "ADAPTERS",
    "ProviderAdapter",
    "Tool",
    "WrappedClient",
    "adapter_for_calls",
    "tool",
    "wrap",
]
