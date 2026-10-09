"""agent-replay: record-and-replay regression testing for LLM agents."""

from agent_replay.cassette import Cassette, Envelope, Kind
from agent_replay.errors import (
    AgentLoadError,
    CassetteError,
    RecordedException,
    ReplayError,
    UnrecordedCallError,
)
from agent_replay.record import Tool, tool, wrap
from agent_replay.session import RecordingSession, ReplaySession, current_session
from agent_replay.trajectory import Step, StepStatus, Trajectory

__version__ = "0.1.0"

__all__ = [
    "AgentLoadError",
    "Cassette",
    "CassetteError",
    "Envelope",
    "Kind",
    "RecordedException",
    "RecordingSession",
    "ReplayError",
    "ReplaySession",
    "Step",
    "StepStatus",
    "Tool",
    "Trajectory",
    "UnrecordedCallError",
    "__version__",
    "current_session",
    "tool",
    "wrap",
]
