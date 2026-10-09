"""Statistics every headline number depends on.

Three things in agent-replay are only as good as the statistics underneath:

* judge-human agreement (:func:`cohens_kappa`, with :func:`bootstrap_ci` for
  its confidence interval), which decides whether the LLM judge may gate CI;
* intervals on pass rates (:func:`wilson_interval`), reported for every
  scenario that is run more than once;
* the sequential CI gate (:class:`SequentialGate`), which decides after as few
  runs as possible whether a flaky scenario is healthy or failing.

This module holds the chance-corrected and exact methods and uses the standard
library only (``math``, ``random``, ``statistics.NormalDist``, ``typing``).
Every ``ValueError`` message names the offending argument (``n_resamples``,
``alpha``, ``successes``, ``p_fail`` and so on), or says ``empty`` for empty
label sequences and ``length`` for a length mismatch; the tests match on that
word, never on the rest of the wording.
:mod:`agent_replay.stats_baseline` is the naive reference behind the same
interface (raw agreement, normal-approximation intervals, a fixed-k majority
gate) that this module is measured against; :mod:`agent_replay.stats_impl`
selects one of the two (``REPLAY_STATS_IMPL=baseline|core``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

Weights = Literal[None, "linear", "quadratic"]
"""Disagreement weights for :func:`cohens_kappa`."""

Decision = Literal["continue", "pass", "fail"]
"""What :meth:`SequentialGate.update` decided after the latest run."""

_NOT_IMPLEMENTED = (
    "agent_replay.stats.{name} is not implemented yet; set REPLAY_STATS_IMPL=baseline"
)


def cohens_kappa(a: Sequence[int], b: Sequence[int], *, weights: Weights = None) -> float:
    """Cohen's kappa between two raters' integer labels for the same items.

    Let ``lo`` and ``hi`` be the smallest and largest label over *both* raters
    and ``K = hi - lo + 1``. Over the label values ``i, j`` in ``lo..hi``:

    * ``O[i][j]`` is the observed proportion of items that rater *a* labelled
      ``i`` and rater *b* labelled ``j``;
    * ``E[i][j] = (row_i) * (col_j)`` is the proportion expected by chance, the
      product of *a*'s marginal proportion of ``i`` and *b*'s of ``j``;
    * ``w[i][j]`` is the disagreement weight: ``0 if i == j else 1`` when
      ``weights`` is ``None``, ``|i - j|`` for ``"linear"`` and ``(i - j) ** 2``
      for ``"quadratic"``. Distances are measured on the label *values*, so on
      an ordinal 1..5 scale a level that neither rater used still counts
      towards the distance between its neighbours.

    The result is ``1 - sum(w * O) / sum(w * E)``: 1.0 for perfect agreement,
    0.0 for agreement at chance level, negative for systematic disagreement.
    This equals scikit-learn's
    ``cohen_kappa_score(a, b, weights=weights, labels=list(range(lo, hi + 1)))``.

    When chance disagreement is zero (``sum(w * E) == 0``, for example when
    both raters always use the same single label) kappa is undefined and the
    result is ``math.nan``, as in scikit-learn.

    Raises:
        ValueError: ``a`` is empty, ``a`` and ``b`` differ in length, or
            ``weights`` is not ``None``, ``"linear"`` or ``"quadratic"``.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="cohens_kappa"))


def bootstrap_ci(
    stat: Callable[[Sequence[int], Sequence[int]], float],
    a: Sequence[int],
    b: Sequence[int],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired percentile bootstrap confidence interval for ``stat(a, b)``.

    With ``n = len(a)`` and ``rng = random.Random(seed)``, draw ``n_resamples``
    resamples; each is ``n`` indices ``rng.randrange(n)``, drawn one after the
    other, and the *same* indices select from ``a`` and from ``b`` (the pairs
    stay together). Drawing them that way, and in that order, is part of the
    specification: :mod:`agent_replay.stats_baseline` draws the same indices,
    so the two intervals differ only in how the interval is formed.
    ``stat`` is computed on every resample; NaN results
    (for example kappa on a resample where it is undefined) are dropped, and
    the ``m`` remaining values are sorted. The interval is::

        lo = sorted_values[floor(alpha / 2 * m)]
        hi = sorted_values[ceil((1 - alpha / 2) * m) - 1]

    The result is deterministic for a given ``seed``. If every resample is
    NaN the result is ``(nan, nan)``.

    Raises:
        ValueError: ``a`` is empty, ``a`` and ``b`` differ in length,
            ``n_resamples < 1``, or ``alpha`` is not strictly between 0 and 1.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="bootstrap_ci"))


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``successes / trials``.

    With ``p = successes / trials``, ``n = trials`` and
    ``z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)``::

        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half   = z / (1 + z**2 / n) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        (lo, hi) = (centre - half, centre + half)

    Both bounds are clamped to ``[0, 1]``; the lower bound is exactly ``0.0``
    when ``successes == 0`` and the upper bound exactly ``1.0`` when
    ``successes == trials``. With ``trials == 0`` there is no information and
    the result is ``(0.0, 1.0)``. Matches statsmodels'
    ``proportion_confint(successes, trials, alpha=1 - confidence, method="wilson")``.

    Raises:
        ValueError: a count is negative, ``successes > trials``, or
            ``confidence`` is not strictly between 0 and 1.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="wilson_interval"))


class SequentialGate:
    """Wald's sequential probability ratio test (SPRT) on Bernoulli run outcomes.

    The gate watches repeated runs of one scenario and decides as early as the
    evidence allows whether it is healthy. H0: the scenario's pass probability
    is ``p_fail`` (it is failing); H1: it is ``p_pass`` (it is healthy).

    The log-likelihood ratio (LLR) starts at 0. Each :meth:`update` adds
    ``ln(p_pass / p_fail)`` for a pass and ``ln((1 - p_pass) / (1 - p_fail))``
    for a fail, then compares it with the boundaries::

        A = ln((1 - beta) / alpha)   # LLR >= A  -> "pass"
        B = ln(beta / (1 - alpha))   # LLR <= B  -> "fail"

    Both boundaries are inclusive. The test is truncated at ``max_runs``: if
    the ``max_runs``-th update reaches neither boundary, the decision is
    ``"pass"`` if ``LLR > 0`` and ``"fail"`` otherwise. Before that, a run that
    reaches neither boundary returns ``"continue"``.

    ``alpha`` is the probability of passing a scenario whose true rate is
    ``p_fail``; ``beta`` is the probability of failing one whose true rate is
    ``p_pass`` (both before truncation).

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
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="SequentialGate"))

    @property
    def runs(self) -> int:
        """Number of outcomes passed to :meth:`update` so far."""
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="SequentialGate.runs"))

    @property
    def passes(self) -> int:
        """Number of those outcomes that were passes."""
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="SequentialGate.passes"))

    @property
    def decision(self) -> Decision:
        """The latest decision: ``"continue"`` until a boundary or ``max_runs`` is reached."""
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="SequentialGate.decision"))

    def update(self, passed: bool) -> Decision:
        """Record one run's outcome and return the (possibly final) decision.

        Raises:
            RuntimeError: the gate has already decided ``"pass"`` or
                ``"fail"``; create a new :class:`SequentialGate` for a new
                sequence of runs.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="SequentialGate.update"))
