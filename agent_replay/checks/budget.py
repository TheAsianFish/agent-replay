"""``budget``: absolute cost, token, model-call and latency limits over a run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agent_replay.baseline import measure, percentile_nearest_rank
from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

__all__ = ["Budget", "budget", "percentile_nearest_rank"]


@dataclass(frozen=True)
class Budget:
    """Absolute limits over one run: cost, tokens, model calls, p95 model latency.

    Served steps carry their recorded cost and latency, so under a ``full`` replay
    with the ``strict`` policy this re-measures the recording: the agent cannot
    make a model call the cassette lacks, and a slower model does not show. The
    limits bite in ``model-live`` (real calls, real cost and latency) and under
    ``fallback`` or ``synth``, where the agent may make calls the recording never
    saw. Relative limits against a baseline arrive with M3.
    """

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_model_calls: int | None = None
    max_p95_latency_ms: float | None = None

    @property
    def name(self) -> str:
        return "budget"

    def __call__(self, run: RunResult, /) -> CheckResult:
        measured: dict[str, float | int | None] = measure(run).model_dump()
        limits: list[tuple[str, str, float | int | None]] = [
            ("cost_usd", "max_cost_usd", self.max_cost_usd),
            ("tokens", "max_tokens", self.max_tokens),
            ("model_calls", "max_model_calls", self.max_model_calls),
            ("p95_latency_ms", "max_p95_latency_ms", self.max_p95_latency_ms),
        ]
        details: dict[str, JsonValue] = {}
        exceeded: list[str] = []
        for key, limit_key, limit in limits:
            value = measured[key]
            details[key] = value
            details[limit_key] = limit
            if limit is not None and value is not None and value > limit:
                exceeded.append(f"{key} {value:g} > {limit_key} {limit:g}")
        if exceeded:
            message = "over budget: " + "; ".join(exceeded)
        else:
            active = [k for k, _, lim in limits if lim is not None]
            message = f"within budget ({', '.join(active)})"
        return CheckResult(check=self.name, passed=not exceeded, message=message, details=details)


def budget(
    *,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    max_model_calls: int | None = None,
    max_p95_latency_ms: float | None = None,
) -> Budget:
    """Build a ``budget`` check. See :class:`Budget`. At least one limit is required."""
    given = (max_cost_usd, max_tokens, max_model_calls, max_p95_latency_ms)
    if all(g is None for g in given):
        raise ValueError(
            "budget needs at least one limit: max_cost_usd, max_tokens, max_model_calls "
            "or max_p95_latency_ms"
        )
    if any(g is not None and g < 0 for g in given):
        raise ValueError("budget limits must be >= 0")
    return Budget(max_cost_usd, max_tokens, max_model_calls, max_p95_latency_ms)
