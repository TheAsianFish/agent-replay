"""The retail agent with its model swapped, for ``replay flake`` in ``model-live``.

Same prompt, tools and loop as :mod:`examples.retail_agent.agent`; only the
model differs. Replaying the Opus recordings with tools served from the cassette
asks whether the cheaper model reaches the same outcomes, and how often::

    uv run replay flake examples/retail_agent/replay.toml --no-store \\
        --cassette-dir examples/retail_agent/cassettes/live \\
        --client anthropic --agent examples.retail_agent.variants:run_haiku
"""

from __future__ import annotations

from examples.retail_agent.agent import RetailAgent


class HaikuRetailAgent(RetailAgent):
    model = "claude-haiku-5-5"


run_haiku = HaikuRetailAgent().run
