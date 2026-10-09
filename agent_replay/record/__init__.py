"""Interception points: wrapped model clients and decorated tools."""

from agent_replay.record.anthropic import wrap
from agent_replay.record.tools import Tool, tool

__all__ = ["Tool", "tool", "wrap"]
