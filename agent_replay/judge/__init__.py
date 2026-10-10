"""The ``judge`` check: grade a run against a rubric with an LLM.

A judge verdict is advisory until a calibration report shows it agrees with
human labels (PLAN.md decision 5).
"""

from agent_replay.judge.cache import CacheEntry, JudgeCache
from agent_replay.judge.calibration import (
    CalibrationReport,
    CriterionCalibration,
    CriterionGate,
    gating,
)
from agent_replay.judge.check import Judge, judge
from agent_replay.judge.providers import (
    AnthropicJudge,
    FakeJudge,
    JudgeProvider,
    OllamaJudge,
    make_provider,
)
from agent_replay.judge.rubric import Criterion, Rubric
from agent_replay.judge.transcript import AgentError, JudgeInput, ToolCallRecord
from agent_replay.judge.verdict import (
    PROMPT_VERSION,
    CriterionVerdict,
    JudgeOutput,
    JudgeResponse,
)

__all__ = [
    "PROMPT_VERSION",
    "AgentError",
    "AnthropicJudge",
    "CacheEntry",
    "CalibrationReport",
    "Criterion",
    "CriterionCalibration",
    "CriterionGate",
    "CriterionVerdict",
    "FakeJudge",
    "Judge",
    "JudgeCache",
    "JudgeInput",
    "JudgeOutput",
    "JudgeProvider",
    "JudgeResponse",
    "OllamaJudge",
    "Rubric",
    "ToolCallRecord",
    "gating",
    "judge",
    "make_provider",
]
