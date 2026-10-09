"""Naive statistics behind the same interface as :mod:`agent_replay.stats`.

These are the obvious first answers that the methods in :mod:`agent_replay.stats`
are measured against:

* :func:`cohens_kappa` returns *raw agreement*, which is not chance-corrected;
* :func:`bootstrap_ci` is a normal-approximation bootstrap interval;
* :func:`wilson_interval` returns the Wald interval;
* :class:`SequentialGate` always runs ``max_runs`` times and takes a majority.

Validation errors and edge-case conventions match :mod:`agent_replay.stats`
wherever they apply. Standard library only.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from statistics import NormalDist

from agent_replay.stats import Decision, Weights

__all__ = [
    "Decision",
    "SequentialGate",
    "Weights",
    "bootstrap_ci",
    "cohens_kappa",
    "wilson_interval",
]

_WEIGHTS = (None, "linear", "quadratic")


def _check_pairs(a: Sequence[int], b: Sequence[int]) -> None:
    if len(a) == 0:
        raise ValueError("need at least one labelled item; pass non-empty label sequences")
    if len(a) != len(b):
        raise ValueError(
            f"the raters labelled different numbers of items ({len(a)} vs {len(b)}); "
            "pass one label per item from each rater, in the same order"
        )


def cohens_kappa(a: Sequence[int], b: Sequence[int], *, weights: Weights = None) -> float:
    """Raw agreement between two raters: the baseline kappa is compared against, not kappa.

    Unweighted, the fraction of items on which ``a`` and ``b`` agree. With
    ``"linear"`` or ``"quadratic"`` weights, ``1 - mean(d)`` where
    ``d = |a_k - b_k| / (hi - lo)`` (squared for ``"quadratic"``) and ``lo``,
    ``hi`` are the smallest and largest label over both raters; 1.0 when
    ``hi == lo``. Nothing is subtracted for agreement expected by chance, so two
    raters who always answer the same majority label score close to 1.

    Raises:
        ValueError: ``a`` is empty, ``a`` and ``b`` differ in length, or
            ``weights`` is not ``None``, ``"linear"`` or ``"quadratic"``.
    """
    if weights not in _WEIGHTS:
        raise ValueError(f"unknown weights {weights!r}; use None, 'linear' or 'quadratic'")
    _check_pairs(a, b)
    if weights is None:
        return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    lo = min(*a, *b)
    hi = max(*a, *b)
    if hi == lo:
        return 1.0
    power = 1 if weights == "linear" else 2
    total = sum((abs(x - y) / (hi - lo)) ** power for x, y in zip(a, b, strict=True))
    return 1.0 - total / len(a)


def _resampled_stats(
    stat: Callable[[Sequence[int], Sequence[int]], float],
    a: Sequence[int],
    b: Sequence[int],
    n_resamples: int,
    seed: int,
) -> list[float]:
    """``stat`` on ``n_resamples`` paired resamples, NaN results dropped."""
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling, not cryptography
    n = len(a)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = stat([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(value):
            values.append(value)
    return values


def bootstrap_ci(
    stat: Callable[[Sequence[int], Sequence[int]], float],
    a: Sequence[int],
    b: Sequence[int],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Normal-approximation bootstrap interval: ``stat(a, b) ± z * sd``.

    Resamples exactly like :func:`agent_replay.stats.bootstrap_ci` (paired
    indices drawn with replacement from ``random.Random(seed)``, NaN results
    dropped); ``sd`` is the sample standard deviation of the resampled
    statistics and ``z = NormalDist().inv_cdf(1 - alpha / 2)``. The interval is
    symmetric and not clamped. ``(nan, nan)`` when fewer than two resamples
    give a finite statistic, or when ``stat(a, b)`` itself is NaN.

    Raises:
        ValueError: ``a`` is empty, ``a`` and ``b`` differ in length,
            ``n_resamples < 1``, or ``alpha`` is not strictly between 0 and 1.
    """
    _check_pairs(a, b)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be strictly between 0 and 1 (e.g. 0.05), got {alpha}")
    values = _resampled_stats(stat, a, b, n_resamples, seed)
    if len(values) < 2:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    centre = stat(a, b)
    half = z * statistics.stdev(values)
    return (centre - half, centre + half)


def _check_counts(successes: int, trials: int, confidence: float) -> None:
    if successes < 0 or trials < 0:
        raise ValueError(f"counts must be non-negative, got successes={successes}, trials={trials}")
    if successes > trials:
        raise ValueError(
            f"successes ({successes}) cannot exceed trials ({trials}); "
            "pass the number of passing runs and the number of runs"
        )
    if not 0 < confidence < 1:
        raise ValueError(
            f"confidence must be strictly between 0 and 1 (e.g. 0.95), got {confidence}"
        )


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """The Wald interval ``p ± z * sqrt(p * (1 - p) / n)``, clamped to ``[0, 1]``.

    ``p = successes / trials``, ``z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)``.
    It collapses to a single point at ``successes == 0`` and
    ``successes == trials``, which is why the Wilson interval replaces it.
    ``trials == 0`` returns ``(0.0, 1.0)``. Matches statsmodels'
    ``proportion_confint(..., method="normal")`` clipped to ``[0, 1]``.

    Raises:
        ValueError: a count is negative, ``successes > trials``, or
            ``confidence`` is not strictly between 0 and 1.
    """
    _check_counts(successes, trials, confidence)
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    half = z * math.sqrt(p * (1 - p) / trials)
    return (max(0.0, p - half), min(1.0, p + half))


class SequentialGate:
    """A fixed-k majority gate: always ``max_runs`` runs, then a majority vote.

    :meth:`update` returns ``"continue"`` until it has seen ``max_runs``
    outcomes, then ``"pass"`` if ``passes / runs >= (p_fail + p_pass) / 2`` and
    ``"fail"`` otherwise. ``alpha`` and ``beta`` are validated but unused.

    Raises:
        ValueError: unless ``0 < p_fail < p_pass < 1``, ``0 < alpha < 1``,
            ``0 < beta < 1`` and ``max_runs >= 1``.
    """

    def __init__(
        self,
        p_fail: float,
        p_pass: float,
        *,
        alpha: float = 0.05,
        beta: float = 0.05,
        max_runs: int = 20,
    ) -> None:
        if not 0 < p_fail < p_pass < 1:
            raise ValueError(
                f"need 0 < p_fail < p_pass < 1, got p_fail={p_fail}, p_pass={p_pass}; "
                "p_fail is the pass rate of a failing scenario, p_pass of a healthy one"
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be strictly between 0 and 1 (e.g. 0.05), got {alpha}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be strictly between 0 and 1 (e.g. 0.05), got {beta}")
        if max_runs < 1:
            raise ValueError(f"max_runs must be at least 1, got {max_runs}")
        self._threshold = (p_fail + p_pass) / 2
        self._max_runs = max_runs
        self._runs = 0
        self._passes = 0
        self._decision: Decision = "continue"

    @property
    def runs(self) -> int:
        """Number of outcomes passed to :meth:`update` so far."""
        return self._runs

    @property
    def passes(self) -> int:
        """Number of those outcomes that were passes."""
        return self._passes

    @property
    def decision(self) -> Decision:
        """The latest decision: ``"continue"`` until ``max_runs`` outcomes are in."""
        return self._decision

    def update(self, passed: bool) -> Decision:
        """Record one run's outcome and return the (possibly final) decision.

        Raises:
            RuntimeError: the gate has already decided ``"pass"`` or
                ``"fail"``; create a new :class:`SequentialGate` for a new
                sequence of runs.
        """
        if self._decision != "continue":
            raise RuntimeError(
                f"this gate already decided {self._decision!r} after {self._runs} runs; "
                "create a new SequentialGate for a new sequence of runs"
            )
        self._runs += 1
        self._passes += bool(passed)
        if self._runs >= self._max_runs:
            rate = self._passes / self._runs
            self._decision = "pass" if rate >= self._threshold else "fail"
        return self._decision
