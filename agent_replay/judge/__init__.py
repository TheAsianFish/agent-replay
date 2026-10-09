"""The ``judge`` check: grade a run against a rubric with an LLM.

A judge verdict is advisory until a calibration report shows it agrees with
human labels (PLAN.md decision 5).
"""

from agent_replay.judge.rubric import Criterion, Rubric

__all__ = ["Criterion", "Rubric"]
