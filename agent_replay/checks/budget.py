"""``budget``: cost, token, model-call and latency limits over a run.

Limits are absolute (``max_cost_usd``) or relative to a recorded baseline
(``max_cost_increase_pct``); see :mod:`agent_replay.baseline`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agent_replay.baseline import ScenarioBaseline, measure, percentile_nearest_rank
from agent_replay.checks.base import CheckResult

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

__all__ = ["RELATIVE_LIMITS", "Budget", "budget", "percentile_nearest_rank"]

# (metric, absolute limit, relative limit), in report order.
_METRICS = (
    ("cost_usd", "max_cost_usd", "max_cost_increase_pct"),
    ("tokens", "max_tokens", "max_tokens_increase_pct"),
    ("model_calls", "max_model_calls", "max_model_calls_increase_pct"),
    ("p95_latency_ms", "max_p95_latency_ms", "max_p95_latency_increase_pct"),
)
RELATIVE_LIMITS = tuple(relative for _, _, relative in _METRICS)
"""The names of the relative limits; each needs a baseline."""


def _show(key: str, value: float) -> str:
    return f"{value:.4f}" if key == "cost_usd" else f"{value:g}"


def _relative(
    key: str,
    relative_key: str,
    value: float | None,
    base_value: float | None,
    pct: float,
    delta_pct: float | None,
) -> tuple[str, str]:
    """Apply one relative limit: ``("over" | "within" | "skipped", message part)``."""
    if value is None or base_value is None:
        where = " and ".join(
            w for w, v in (("the run", value), ("the baseline", base_value)) if v is None
        )
        return "skipped", f"{relative_key} (no latency samples in {where})"
    allowed = base_value * (1 + pct / 100)
    if value <= allowed or math.isclose(value, allowed, rel_tol=1e-9):
        if delta_pct is None:
            return "within", f"{key} = baseline 0"
        return "within", f"{key} {delta_pct:+.1f}%"
    if delta_pct is None:
        return "over", (
            f"{key} {_show(key, value)} is over baseline 0 (limit +{pct:g}%; any increase "
            "over a zero baseline exceeds it)"
        )
    return "over", (
        f"{key} {_show(key, value)} is {delta_pct:+.1f}% over baseline "
        f"{_show(key, base_value)} (limit +{pct:g}%)"
    )


@dataclass(frozen=True)
class Budget:
    """Limits over one run: cost, tokens, model calls, p95 model latency.

    Each metric can have an absolute limit (``max_cost_usd``: value <= limit)
    and a relative one against ``baseline`` (``max_cost_increase_pct``: value
    <= baseline * (1 + pct / 100); equality passes). A baseline value of 0
    allows only 0: any positive value exceeds it. A relative latency limit is
    skipped, and the skip named in the message and in ``details["skipped"]``,
    when the run or the baseline has no latency samples.

    Served steps carry their recorded cost and latency, so under a ``full`` replay
    with the ``strict`` policy this re-measures the recording: the agent cannot
    make a model call the cassette lacks, and a slower model does not show. The
    limits bite in ``model-live`` (real calls, real cost and latency) and under
    ``fallback`` or ``synth``, where the agent may make calls the recording never
    saw; a relative limit also catches a re-recorded cassette that got dearer.
    """

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_model_calls: int | None = None
    max_p95_latency_ms: float | None = None
    max_cost_increase_pct: float | None = None
    max_tokens_increase_pct: float | None = None
    max_model_calls_increase_pct: float | None = None
    max_p95_latency_increase_pct: float | None = None
    baseline: ScenarioBaseline | None = None

    @property
    def name(self) -> str:
        return "budget"

    def __call__(self, run: RunResult, /) -> CheckResult:
        measured: dict[str, float | int | None] = measure(run).model_dump()
        base: dict[str, float | int | None] | None = (
            self.baseline.model_dump(exclude={"runs"}) if self.baseline is not None else None
        )
        details: dict[str, JsonValue] = {}
        exceeded: list[str] = []
        compared: list[str] = []
        skipped: list[str] = []
        for key, absolute_key, relative_key in _METRICS:
            value = measured[key]
            limit: float | None = getattr(self, absolute_key)
            details[key] = value
            details[absolute_key] = limit
            if limit is not None and value is None:
                skipped.append(f"{absolute_key} (no latency samples in the run)")
            elif limit is not None and value is not None and value > limit:
                exceeded.append(f"{key} {value:g} > {absolute_key} {limit:g}")
            if base is None:
                continue
            base_value = base[key]
            pct: float | None = getattr(self, relative_key)
            delta = None if value is None or base_value is None else round(value - base_value, 8)
            delta_pct = (
                None if delta is None or not base_value else round(delta / base_value * 100, 4)
            )
            details[f"baseline_{key}"] = base_value
            details[relative_key] = pct
            details[f"{key}_delta"] = delta
            details[f"{key}_delta_pct"] = delta_pct
            if pct is not None:
                outcome, text = _relative(key, relative_key, value, base_value, pct, delta_pct)
                {"over": exceeded, "within": compared, "skipped": skipped}[outcome].append(text)
        if base is not None or skipped:
            details["skipped"] = list(skipped)
        if exceeded:
            message = "over budget: " + "; ".join(exceeded)
        else:
            limited = ", ".join(
                k
                for k, a, _ in _METRICS
                if getattr(self, a) is not None and measured[k] is not None
            )
            parts = [limited] if limited else []
            if compared:
                parts.append("vs baseline: " + ", ".join(compared))
            message = "within budget" + (f" ({'; '.join(parts)})" if parts else "")
        if skipped:
            message += "; skipped " + ", ".join(skipped)
        return CheckResult(check=self.name, passed=not exceeded, message=message, details=details)


def budget(
    *,
    max_cost_usd: float | None = None,
    max_tokens: int | None = None,
    max_model_calls: int | None = None,
    max_p95_latency_ms: float | None = None,
    max_cost_increase_pct: float | None = None,
    max_tokens_increase_pct: float | None = None,
    max_model_calls_increase_pct: float | None = None,
    max_p95_latency_increase_pct: float | None = None,
    baseline: ScenarioBaseline | None = None,
) -> Budget:
    """Build a ``budget`` check. See :class:`Budget`.

    At least one absolute or relative limit is required, every limit is >= 0,
    and relative limits need ``baseline``.
    """
    absolute = (max_cost_usd, max_tokens, max_model_calls, max_p95_latency_ms)
    relative = {
        "max_cost_increase_pct": max_cost_increase_pct,
        "max_tokens_increase_pct": max_tokens_increase_pct,
        "max_model_calls_increase_pct": max_model_calls_increase_pct,
        "max_p95_latency_increase_pct": max_p95_latency_increase_pct,
    }
    given = (*absolute, *relative.values())
    if all(g is None for g in given):
        raise ValueError(
            "budget needs at least one limit: max_cost_usd, max_tokens, max_model_calls "
            "or max_p95_latency_ms, or a relative one (" + ", ".join(RELATIVE_LIMITS) + ")"
        )
    if any(g is not None and g < 0 for g in given):
        raise ValueError("budget limits must be >= 0")
    used = [name for name, pct in relative.items() if pct is not None]
    if used and baseline is None:
        raise ValueError(
            f"relative limits ({', '.join(used)}) need a baseline: record one with "
            '`replay baseline SUITE -o FILE` and set baseline = "FILE" in the suite\'s '
            "[defaults] (or pass baseline=ScenarioBaseline(...))"
        )
    return Budget(
        max_cost_usd,
        max_tokens,
        max_model_calls,
        max_p95_latency_ms,
        max_cost_increase_pct,
        max_tokens_increase_pct,
        max_model_calls_increase_pct,
        max_p95_latency_increase_pct,
        baseline,
    )
