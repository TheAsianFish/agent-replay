"""What the judge sees of a run: the task, the agent's tool calls and its final output.

The judge never sees the agent's own model turns (they are bound to the model
that produced them, and grading the trajectory is the point), only this
transcript built from the run's trajectory. Long values are clipped so one huge
tool result cannot blow the judge's context or cost.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay._json import canonical_json

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

DEFAULT_MAX_CHARS = 2000
"""Default cap, in characters of compact JSON, for each value in the transcript."""


def clip(value: JsonValue, max_chars: int) -> JsonValue:
    """``value`` unchanged if its JSON is at most ``max_chars`` long, else a clipped string.

    A string is clipped as text; any other value is serialised to compact JSON
    and clipped as that. The clipped text ends with a visible
    ``…[truncated N chars]`` marker.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return f"{text[:max_chars]}…[truncated {len(text) - max_chars} chars]"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentError(_Frozen):
    type: str
    message: str


class ToolCallRecord(_Frozen):
    """One of the agent's own tool calls, in the order the calls started."""

    name: str
    input: JsonValue
    result: JsonValue = None
    error: AgentError | None = None


class JudgeInput(_Frozen):
    """The judge's whole view of one run. :meth:`render` is the user message it receives."""

    task: str
    output: JsonValue
    error: AgentError | None = None
    tool_calls: tuple[ToolCallRecord, ...] = ()
    model_call_count: int = 0
    tool_call_count: int = 0

    @classmethod
    def from_run(cls, run: RunResult, *, max_chars: int = DEFAULT_MAX_CHARS) -> JudgeInput:
        """Build the transcript of ``run``: top-level tool calls only, each value clipped."""
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        calls = tuple(
            ToolCallRecord(
                name=step.name,
                input=clip(step.request, max_chars),
                result=None if step.error is not None else clip(step.response, max_chars),
                error=(
                    AgentError(
                        type=step.error.type, message=str(clip(step.error.message, max_chars))
                    )
                    if step.error is not None
                    else None
                ),
            )
            for step in run.trajectory.tool_steps()
        )
        error = (
            AgentError(type=run.error.type, message=str(clip(run.error.message, max_chars)))
            if run.error is not None
            else None
        )
        return cls(
            task=str(clip(run.task, max_chars)),
            output=clip(run.output, max_chars),
            error=error,
            tool_calls=calls,
            model_call_count=len(run.trajectory.model_steps()),
            tool_call_count=len(calls),
        )

    @property
    def fingerprint(self) -> str:
        """sha256 over the canonical JSON of the transcript."""
        payload = canonical_json(self.model_dump(mode="json"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render(self) -> str:
        """The transcript as the judge's user message: one tagged section per part."""
        parts = [
            "Grade this run of the agent against the rubric.",
            "",
            "<task>",
            self.task,
            "</task>",
            "",
            f'<tool_calls count="{self.tool_call_count}" model_calls="{self.model_call_count}">',
        ]
        if not self.tool_calls:
            parts.append("(the agent made no tool calls)")
        for i, call in enumerate(self.tool_calls, start=1):
            parts += [f"## Call {i}: {call.name}", "Input:", _pretty(call.input)]
            if call.error is not None:
                parts += [f"Error ({call.error.type}): {call.error.message}"]
            else:
                parts += ["Result:", _pretty(call.result)]
            parts.append("")
        parts.append("</tool_calls>")
        parts.append("")
        if self.error is not None:
            parts += [
                "<agent_error>",
                f"The agent did not finish: {self.error.type}: {self.error.message}",
                "</agent_error>",
                "",
            ]
        parts += ["<final_output>", _pretty(self.output), "</final_output>"]
        return "\n".join(parts) + "\n"


def _pretty(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)
