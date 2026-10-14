"""The dashboard's read-only HTTP API over the run store (``replay serve``).

Needs the ``server`` extra: ``pip install 'agent-replay[server]'``.
"""

from agent_replay.server.app import create_app

__all__ = ["create_app"]
