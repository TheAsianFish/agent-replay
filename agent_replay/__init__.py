"""agent-replay: record-and-replay regression testing for LLM agents."""

from agent_replay import capture
from agent_replay.capture import choice, now, randint, random, uuid4
from agent_replay.cassette import Cassette, Envelope, Kind
from agent_replay.errors import (
    AgentLoadError,
    CassetteError,
    ConfigError,
    RecordedException,
    ReplayError,
    StoreError,
    UnrecordedCallError,
)
from agent_replay.record import Tool, tool, wrap
from agent_replay.runner import RunResult, record, replay
from agent_replay.session import (
    Policy,
    RecordingSession,
    ReplayMode,
    ReplaySession,
    current_session,
)
from agent_replay.threads import ContextThreadPoolExecutor
from agent_replay.trajectory import Step, StepStatus, Trajectory

__version__ = "0.1.0"

__all__ = [
    "AgentLoadError",
    "Cassette",
    "CassetteError",
    "ConfigError",
    "ContextThreadPoolExecutor",
    "Envelope",
    "Kind",
    "Policy",
    "RecordedException",
    "RecordingSession",
    "ReplayError",
    "ReplayMode",
    "ReplaySession",
    "RunResult",
    "Step",
    "StepStatus",
    "StoreError",
    "Tool",
    "Trajectory",
    "UnrecordedCallError",
    "__version__",
    "capture",
    "choice",
    "current_session",
    "now",
    "randint",
    "random",
    "record",
    "replay",
    "tool",
    "uuid4",
    "wrap",
]
