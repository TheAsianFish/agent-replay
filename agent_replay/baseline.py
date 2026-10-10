"""Cost and latency baselines: what a suite's scenarios cost when they last passed.

A baseline file (``replay baseline SUITE -o FILE``) holds, per scenario, the
median cost, tokens, model calls and p95 model latency over one or more runs.
The ``budget`` check's relative limits (``max_cost_increase_pct`` and friends)
compare a run against it. :func:`measure` is the single definition of those
metrics, shared by the check and the baseline, so both always agree.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_replay.errors import ConfigError
from agent_replay.session import Policy, ReplayMode

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

_RECORD_HINT = "record one with `replay baseline SUITE -o PATH`"


def percentile_nearest_rank(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100); ``None`` for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


class RunMetrics(BaseModel):
    """What one run cost: the quantities ``budget`` limits and baselines record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_usd: float
    """Priced model cost; served steps count at their recorded cost."""
    tokens: int
    """Input, output and cache tokens over every model call."""
    model_calls: int
    p95_latency_ms: float | None
    """Nearest-rank p95 over the model steps that carry a latency; ``None`` if none do."""


def measure(run: RunResult) -> RunMetrics:
    """Measure one run. Served steps carry their recorded cost and latency."""
    model_steps = run.trajectory.model_steps()
    latencies = [s.latency_ms for s in model_steps if s.latency_ms is not None]
    return RunMetrics(
        cost_usd=run.cost_usd,
        tokens=run.tokens,
        model_calls=len(model_steps),
        p95_latency_ms=percentile_nearest_rank(latencies, 95),
    )


class ScenarioBaseline(BaseModel):
    """One scenario's baseline: the per-metric median over ``runs`` runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_usd: float = Field(ge=0)
    tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    runs: int = Field(ge=1)


def _median_int(values: Sequence[int]) -> int:
    # The median of an even count can fall halfway between two integers; round
    # half up so a baseline never understates the usage it describes.
    return math.floor(statistics.median(values) + 0.5)


def aggregate(metrics: Sequence[RunMetrics]) -> ScenarioBaseline:
    """Per-metric median across runs (each metric independently).

    Integer metrics (tokens, model calls) are rounded half up when an even
    number of runs puts the median between two integers; cost is rounded to
    8 decimals like :attr:`RunResult.cost_usd`. ``p95_latency_ms`` is the median
    over the runs that measured a latency, or ``None`` when none did.
    """
    if not metrics:
        raise ValueError("aggregate needs at least one run's metrics")
    latencies = [m.p95_latency_ms for m in metrics if m.p95_latency_ms is not None]
    return ScenarioBaseline(
        cost_usd=round(statistics.median(m.cost_usd for m in metrics), 8),
        tokens=_median_int([m.tokens for m in metrics]),
        model_calls=_median_int([m.model_calls for m in metrics]),
        p95_latency_ms=statistics.median(latencies) if latencies else None,
        runs=len(metrics),
    )


class BaselineFile(BaseModel):
    """A suite's baselines, as written by ``replay baseline``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    suite: str
    """The suite file the baseline was measured from."""
    created_at: datetime
    git_sha: str | None = None
    mode: ReplayMode
    """The suite-wide mode (``--mode``, else ``[defaults]``); a scenario may override it."""
    policy: Policy
    scenarios: dict[str, ScenarioBaseline]

    @classmethod
    def load(cls, path: str | Path) -> BaselineFile:
        """Read a baseline file; any problem is a :class:`ConfigError` naming the fix."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"baseline file {path} does not exist; {_RECORD_HINT}") from exc
        except OSError as exc:
            raise ConfigError(
                f"cannot read baseline file {path} ({exc.strerror or exc}); {_RECORD_HINT}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"baseline file {path} is not valid JSON ({exc}); {_RECORD_HINT}"
            ) from exc
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'top level'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ConfigError(
                f"baseline file {path} is invalid ({problems}); {_RECORD_HINT}"
            ) from exc

    def dumps(self) -> str:
        """Stable JSON: 2-space indent, scenarios sorted by name, trailing newline."""
        data = self.model_dump(mode="json")
        data["scenarios"] = dict(sorted(data["scenarios"].items()))
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    def dump(self, path: str | Path) -> None:
        """Write the file (see :meth:`dumps`), creating parent directories. Raises ``OSError``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps(), encoding="utf-8")
