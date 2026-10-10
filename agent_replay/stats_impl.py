"""Selecting the statistics implementation: ``REPLAY_STATS_IMPL=baseline|core``.

Both :mod:`agent_replay.stats` (``core``) and :mod:`agent_replay.stats_baseline`
(``baseline``) expose the same surface, described here as the :class:`StatsImpl`
protocol; :func:`load` returns one of the two modules. ``baseline`` is the
default until ``core`` is selected explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Final, Protocol

from agent_replay import stats, stats_baseline
from agent_replay.errors import ConfigError
from agent_replay.stats import Decision, Weights

IMPLS: Final = ("baseline", "core")
ENV_VAR: Final = "REPLAY_STATS_IMPL"


class Gate(Protocol):
    """A sequential pass/fail gate over repeated runs of one scenario."""

    @property
    def runs(self) -> int: ...

    @property
    def passes(self) -> int: ...

    @property
    def decision(self) -> Decision: ...

    def update(self, passed: bool) -> Decision: ...


class GateFactory(Protocol):
    """The ``SequentialGate`` constructor."""

    def __call__(
        self,
        p_fail: float,
        p_pass: float,
        *,
        alpha: float = ...,
        beta: float = ...,
        max_runs: int = ...,
    ) -> Gate: ...


class StatsImpl(Protocol):
    """The public surface of :mod:`agent_replay.stats` and :mod:`agent_replay.stats_baseline`."""

    @property
    def SequentialGate(self) -> GateFactory: ...

    def cohens_kappa(
        self, a: Sequence[int], b: Sequence[int], *, weights: Weights = ...
    ) -> float: ...

    def bootstrap_ci(
        self,
        stat: Callable[[Sequence[int], Sequence[int]], float],
        a: Sequence[int],
        b: Sequence[int],
        *,
        n_resamples: int = ...,
        alpha: float = ...,
        seed: int = ...,
    ) -> tuple[float, float]: ...

    def wilson_interval(
        self, successes: int, trials: int, *, confidence: float = ...
    ) -> tuple[float, float]: ...


# Type-checked proof that both modules satisfy the protocol: mypy --strict
# rejects these assignments if either module's surface drifts from StatsImpl.
_MODULES: Final[dict[str, StatsImpl]] = {"baseline": stats_baseline, "core": stats}


def load(name: str | None = None) -> StatsImpl:
    """The statistics implementation called ``name``.

    ``name`` defaults to the ``REPLAY_STATS_IMPL`` environment variable, then
    to ``"baseline"``. ``"core"`` is :mod:`agent_replay.stats`; ``"baseline"``
    is :mod:`agent_replay.stats_baseline`.

    Raises:
        ConfigError: ``name`` is not one of :data:`IMPLS`.
    """
    chosen = name if name is not None else os.environ.get(ENV_VAR) or "baseline"
    try:
        return _MODULES[chosen]
    except KeyError:
        valid = ", ".join(IMPLS)
        raise ConfigError(
            f"unknown statistics implementation {chosen!r}; set {ENV_VAR} "
            f"(or pass name=) to one of: {valid}"
        ) from None
