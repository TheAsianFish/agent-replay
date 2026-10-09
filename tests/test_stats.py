"""Tests for the statistics implementations: core (``stats``) and ``stats_baseline``.

Shared tests run against both implementations through the ``impl`` fixture.
Tests of core behaviour carry ``CORE_XFAIL``: while ``agent_replay.stats``
raises ``NotImplementedError`` they xfail; once it is implemented they XPASS
when correct and fail for real when wrong (any other exception or a failed
assertion is not covered by ``raises=``).
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import accuracy_score, cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from agent_replay import stats, stats_baseline, stats_impl
from agent_replay.errors import ConfigError

CORE_XFAIL = pytest.mark.xfail(
    raises=NotImplementedError, strict=False, reason="agent_replay.stats is not implemented yet"
)

SETTINGS = settings(derandomize=True, deadline=None, max_examples=100)

Z95 = 1.959963984540054  # NormalDist().inv_cdf(0.975)


@pytest.fixture(
    params=[
        pytest.param(stats_baseline, id="baseline"),
        pytest.param(stats, id="core", marks=CORE_XFAIL),
    ]
)
def impl(request: pytest.FixtureRequest) -> ModuleType:
    module: ModuleType = request.param
    return module


def table(counts: dict[tuple[int, int], int]) -> tuple[list[int], list[int]]:
    """Expand a contingency table {(label_a, label_b): count} into two label lists."""
    a: list[int] = []
    b: list[int] = []
    for (x, y), n in counts.items():
        a += [x] * n
        b += [y] * n
    return a, b


# The classic 50-item yes(1)/no(0) example: 20 both yes, 5 a-yes b-no,
# 10 a-no b-yes, 15 both no.
CLASSIC_A, CLASSIC_B = table({(1, 1): 20, (1, 0): 5, (0, 1): 10, (0, 0): 15})

# Three categories, pairs (0,0) (0,1) (1,1) (1,2) (2,2) (2,1).
THREE_A = [0, 0, 1, 1, 2, 2]
THREE_B = [0, 1, 1, 2, 2, 1]


def agreement(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def labels_range(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return list(range(min(*a, *b), max(*a, *b) + 1))


def oracle_kappa(a: Sequence[int], b: Sequence[int], weights: str | None) -> float:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, weights=weights, labels=labels_range(a, b)))


def oracle_interval(
    successes: int, trials: int, confidence: float, method: str
) -> tuple[float, float]:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        lo, hi = proportion_confint(successes, trials, alpha=1 - confidence, method=method)
    return (min(max(float(lo), 0.0), 1.0), min(max(float(hi), 0.0), 1.0))


pairs = st.lists(st.tuples(st.integers(0, 4), st.integers(0, 4)), min_size=1, max_size=60)
counts = st.integers(0, 500).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n)))
confidences = st.sampled_from([0.5, 0.8, 0.9, 0.95, 0.99])


# --- cohens_kappa: hand-computed --------------------------------------------------------


@CORE_XFAIL
def test_kappa_classic_two_by_two() -> None:
    # po = (20 + 15) / 50 = 0.7; a says yes 25/50 = 0.5, b says yes 30/50 = 0.6;
    # pe = 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    assert stats.cohens_kappa(CLASSIC_A, CLASSIC_B) == pytest.approx(0.4)


@CORE_XFAIL
def test_kappa_three_categories_weighted() -> None:
    # Marginals: a = (1/3, 1/3, 1/3), b = (1/6, 1/2, 1/3); E[i][j] = a_i * b_j.
    # Unweighted: po = 3/6, pe = 1/3 -> (1/2 - 1/3) / (2/3) = 0.25.
    # Linear: sum(w*O) = 3 * 1 / 6 = 1/2; column distance sums sum_i |i-j| = 3, 2, 3,
    #   sum(w*E) = 1/3 * (3/6 + 2/2 + 3/3) = 5/6 -> 1 - (1/2) / (5/6) = 0.4.
    # Quadratic: sum(w*O) = 1/2; sum_i (i-j)^2 = 5, 2, 5,
    #   sum(w*E) = 1/3 * (5/6 + 2/2 + 5/3) = 7/6 -> 1 - (1/2) / (7/6) = 4/7.
    assert stats.cohens_kappa(THREE_A, THREE_B) == pytest.approx(0.25)
    assert stats.cohens_kappa(THREE_A, THREE_B, weights="linear") == pytest.approx(0.4)
    assert stats.cohens_kappa(THREE_A, THREE_B, weights="quadratic") == pytest.approx(4 / 7)


@CORE_XFAIL
def test_kappa_unobserved_middle_level_counts_as_distance() -> None:
    # a = [1, 2, 4], b = [2, 4, 4] on the scale 1..5; level 3 is never used.
    # Linear, distances on values: sum(w*O) = (1 + 2 + 0) / 3 = 1.
    # Marginals a = (1/3 at 1, 2, 4), b = (1/3 at 2, 2/3 at 4):
    #   sum(w*E) = 1/3 * [(1/3 * 1 + 2/3 * 3) + (2/3 * 2) + (1/3 * 2)] = 13/9
    #   -> 1 - 1 / (13/9) = 4/13. (On ranks 0, 1, 2 instead it would be 0.25.)
    a, b = [1, 2, 4], [2, 4, 4]
    assert stats.cohens_kappa(a, b, weights="linear") == pytest.approx(4 / 13)
    assert stats.cohens_kappa(a, b, weights="linear") == pytest.approx(oracle_kappa(a, b, "linear"))


@CORE_XFAIL
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_kappa_perfect_agreement_is_one(weights: stats.Weights) -> None:
    assert stats.cohens_kappa([0, 1, 2, 2, 1], [0, 1, 2, 2, 1], weights=weights) == 1.0


@CORE_XFAIL
def test_kappa_systematic_disagreement_is_negative() -> None:
    # po = 0; marginals 1/2 each, pe = 1/2 -> (0 - 1/2) / (1/2) = -1
    assert stats.cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)


@CORE_XFAIL
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_kappa_constant_raters_is_nan(weights: stats.Weights) -> None:
    assert math.isnan(stats.cohens_kappa([2, 2, 2], [2, 2, 2], weights=weights))


def test_baseline_agreement_hand_computed() -> None:
    assert stats_baseline.cohens_kappa(CLASSIC_A, CLASSIC_B) == pytest.approx(0.7)
    # d = |a - b| / (2 - 0): three pairs at 1/2 -> linear 1 - 1.5/6 = 0.75,
    # quadratic 1 - 3 * 0.25 / 6 = 0.875
    assert stats_baseline.cohens_kappa(THREE_A, THREE_B, weights="linear") == pytest.approx(0.75)
    assert stats_baseline.cohens_kappa(THREE_A, THREE_B, weights="quadratic") == pytest.approx(
        0.875
    )


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_baseline_constant_raters_agree_fully(weights: stats.Weights) -> None:
    assert stats_baseline.cohens_kappa([2, 2, 2], [2, 2, 2], weights=weights) == 1.0


# --- cohens_kappa: errors ---------------------------------------------------------------


def test_kappa_rejects_empty(impl: ModuleType) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        impl.cohens_kappa([], [])


def test_kappa_rejects_unequal_lengths(impl: ModuleType) -> None:
    with pytest.raises(ValueError, match="same order"):
        impl.cohens_kappa([0, 1], [0])


def test_kappa_rejects_unknown_weights(impl: ModuleType) -> None:
    with pytest.raises(ValueError, match="quadratic"):
        impl.cohens_kappa([0, 1], [0, 1], weights="cubic")


# --- cohens_kappa: properties -----------------------------------------------------------


@CORE_XFAIL
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_kappa_matches_sklearn(weights: stats.Weights) -> None:
    stats.cohens_kappa([0, 1], [0, 1], weights=weights)  # fail fast while unimplemented

    @SETTINGS
    @given(data=pairs)
    def check(data: list[tuple[int, int]]) -> None:
        a = [x for x, _ in data]
        b = [y for _, y in data]
        expected = oracle_kappa(a, b, weights)
        got = stats.cohens_kappa(a, b, weights=weights)
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-9)

    check()


@SETTINGS
@given(data=pairs)
def test_baseline_agreement_matches_accuracy_score(data: list[tuple[int, int]]) -> None:
    a = [x for x, _ in data]
    b = [y for _, y in data]
    assert stats_baseline.cohens_kappa(a, b) == pytest.approx(float(accuracy_score(a, b)))


@SETTINGS
@given(data=pairs, weights=st.sampled_from(["linear", "quadratic"]))
def test_baseline_weighted_agreement_in_unit_interval(
    data: list[tuple[int, int]], weights: stats.Weights
) -> None:
    a = [x for x, _ in data]
    b = [y for _, y in data]
    assert 0.0 <= stats_baseline.cohens_kappa(a, b, weights=weights) <= 1.0


# --- bootstrap_ci -----------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed(impl: ModuleType) -> None:
    a = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    b = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    first = impl.bootstrap_ci(agreement, a, b, n_resamples=300, seed=7)
    assert impl.bootstrap_ci(agreement, a, b, n_resamples=300, seed=7) == first
    lo, hi = first
    assert lo <= agreement(a, b) <= hi


def test_bootstrap_constant_statistic(impl: ModuleType) -> None:
    assert impl.bootstrap_ci(lambda _a, _b: 0.5, [0, 1, 2], [1, 1, 2], n_resamples=50) == (
        0.5,
        0.5,
    )


def test_bootstrap_all_nan_resamples(impl: ModuleType) -> None:
    lo, hi = impl.bootstrap_ci(lambda _a, _b: math.nan, [0, 1], [0, 1], n_resamples=20)
    assert math.isnan(lo)
    assert math.isnan(hi)


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "match"),
    [
        ([], [], {}, "non-empty"),
        ([0, 1], [0], {}, "same order"),
        ([0, 1], [0, 1], {"n_resamples": 0}, "n_resamples"),
        ([0, 1], [0, 1], {"alpha": 0.0}, "alpha"),
        ([0, 1], [0, 1], {"alpha": 1.0}, "alpha"),
    ],
)
def test_bootstrap_rejects_invalid_input(
    impl: ModuleType, a: list[int], b: list[int], kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        impl.bootstrap_ci(agreement, a, b, **kwargs)


def test_bootstrap_lo_le_hi(impl: ModuleType) -> None:
    impl.bootstrap_ci(agreement, [0], [0], n_resamples=1)  # fail fast while unimplemented

    @SETTINGS
    @given(data=pairs, seed=st.integers(0, 2**32), alpha=st.sampled_from([0.01, 0.05, 0.2]))
    def check(data: list[tuple[int, int]], seed: int, alpha: float) -> None:
        a = [x for x, _ in data]
        b = [y for _, y in data]
        lo, hi = impl.bootstrap_ci(agreement, a, b, n_resamples=40, alpha=alpha, seed=seed)
        assert lo <= hi
        again = impl.bootstrap_ci(agreement, a, b, n_resamples=40, alpha=alpha, seed=seed)
        assert again == (lo, hi)

    check()


@CORE_XFAIL
def test_core_bootstrap_percentile_indices() -> None:
    # A statistic that returns 1, 2, ..., 10 on successive calls: with m = 10 and
    # alpha = 0.2, lo = sorted[floor(0.1 * 10)] = sorted[1] = 2 and
    # hi = sorted[ceil(0.9 * 10) - 1] = sorted[8] = 9.
    calls = iter(range(1, 11))
    assert stats.bootstrap_ci(lambda _a, _b: next(calls), [0], [0], n_resamples=10, alpha=0.2) == (
        2,
        9,
    )


@CORE_XFAIL
def test_core_bootstrap_percentile_indices_floor_and_ceil_not_round() -> None:
    # alpha = 0.3 puts both cut points on a half: lo = sorted[floor(1.5)] = sorted[1] = 2
    # and hi = sorted[ceil(8.5) - 1] = sorted[8] = 9. Rounding instead gives (3, 8).
    calls = iter(range(1, 11))
    assert stats.bootstrap_ci(lambda _a, _b: next(calls), [0], [0], n_resamples=10, alpha=0.3) == (
        2,
        9,
    )


@CORE_XFAIL
def test_core_bootstrap_drops_nan_resamples() -> None:
    # Every other resample is NaN; the five finite values 1..5 remain, so with
    # alpha = 0.4: lo = sorted[floor(0.2 * 5)] = 2, hi = sorted[ceil(0.8 * 5) - 1] = 4.
    values = iter([1, math.nan, 2, math.nan, 3, math.nan, 4, math.nan, 5, math.nan])
    assert stats.bootstrap_ci(lambda _a, _b: next(values), [0], [0], n_resamples=10, alpha=0.4) == (
        2,
        4,
    )


@CORE_XFAIL
def test_core_bootstrap_coverage_of_agreement_rate() -> None:
    # 100 simulated datasets of 40 paired yes/no labels whose true agreement rate is
    # 0.8; a 95% percentile interval should cover 0.8 in roughly 95% of them.
    rng = random.Random(20261009)
    true_rate = 0.8
    covered = 0
    datasets = 100
    for i in range(datasets):
        a = [rng.randrange(2) for _ in range(40)]
        b = [x if rng.random() < true_rate else 1 - x for x in a]
        lo, hi = stats.bootstrap_ci(agreement, a, b, n_resamples=400, seed=i)
        covered += lo <= true_rate <= hi
    assert 0.85 <= covered / datasets <= 1.0


# --- wilson_interval --------------------------------------------------------------------


@CORE_XFAIL
def test_wilson_hand_computed() -> None:
    # z = 1.959964, z^2 = 3.841459.
    # (0, 10): lo = 0; hi = (z^2/10) / (1 + z^2/10) = 0.384146 / 1.384146 = 0.277533.
    assert stats.wilson_interval(0, 10) == (0.0, pytest.approx(0.277533, abs=1e-6))
    # (10, 10): the mirror image, (1 - 0.277533, 1).
    assert stats.wilson_interval(10, 10) == (pytest.approx(0.722467, abs=1e-6), 1.0)
    # (5, 10): centre 0.5; half = z / 1.384146 * sqrt(0.025 + z^2/400)
    #   = 1.415999 * sqrt(0.0346036) = 0.263407 -> (0.236593, 0.763407).
    lo, hi = stats.wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.236593, abs=1e-6), pytest.approx(0.763407, abs=1e-6))
    # (81, 263): p = 0.307985, z^2/n = 0.014606; centre = 0.315288 / 1.014606 = 0.310749;
    # half = 1.931749 * sqrt(0.00081039 + 0.00001388) = 0.055461 -> (0.255288, 0.366210).
    lo, hi = stats.wilson_interval(81, 263)
    assert (lo, hi) == (pytest.approx(0.255288, abs=1e-6), pytest.approx(0.366210, abs=1e-6))


def test_baseline_wald_hand_computed() -> None:
    # (5, 10): 0.5 +- 1.959964 * sqrt(0.25 / 10) = 0.5 +- 0.309898
    lo, hi = stats_baseline.wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.190102, abs=1e-6), pytest.approx(0.809898, abs=1e-6))
    # At k = 0 and k = n the Wald interval collapses to a point.
    assert stats_baseline.wilson_interval(0, 10) == (0.0, 0.0)
    assert stats_baseline.wilson_interval(10, 10) == (1.0, 1.0)


def test_interval_endpoints_exact(impl: ModuleType) -> None:
    assert impl.wilson_interval(0, 7)[0] == 0.0
    assert impl.wilson_interval(7, 7)[1] == 1.0


def test_interval_no_trials_is_uninformative(impl: ModuleType) -> None:
    assert impl.wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("successes", "trials", "confidence", "match"),
    [
        (-1, 10, 0.95, "non-negative"),
        (0, -1, 0.95, "non-negative"),
        (11, 10, 0.95, "cannot exceed"),
        (5, 10, 0.0, "confidence"),
        (5, 10, 1.0, "confidence"),
        (5, 10, 1.5, "confidence"),
    ],
)
def test_interval_rejects_invalid_input(
    impl: ModuleType, successes: int, trials: int, confidence: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        impl.wilson_interval(successes, trials, confidence=confidence)


@CORE_XFAIL
def test_wilson_matches_statsmodels() -> None:
    stats.wilson_interval(1, 2)  # fail fast while unimplemented

    @SETTINGS
    @given(count=counts, confidence=confidences)
    def check(count: tuple[int, int], confidence: float) -> None:
        successes, trials = count
        if trials == 0:
            return
        expected = oracle_interval(successes, trials, confidence, "wilson")
        got = stats.wilson_interval(successes, trials, confidence=confidence)
        assert got == pytest.approx(expected, abs=1e-9)

    check()


@SETTINGS
@given(count=counts, confidence=confidences)
def test_baseline_wald_matches_statsmodels_normal(
    count: tuple[int, int], confidence: float
) -> None:
    successes, trials = count
    if trials == 0:
        return
    expected = oracle_interval(successes, trials, confidence, "normal")
    got = stats_baseline.wilson_interval(successes, trials, confidence=confidence)
    assert got == pytest.approx(expected, abs=1e-9)


def test_interval_is_ordered_within_unit(impl: ModuleType) -> None:
    impl.wilson_interval(1, 2)  # fail fast while unimplemented

    @SETTINGS
    @given(count=counts, confidence=confidences)
    def check(count: tuple[int, int], confidence: float) -> None:
        lo, hi = impl.wilson_interval(*count, confidence=confidence)
        assert 0.0 <= lo <= hi <= 1.0

    check()


# --- SequentialGate ---------------------------------------------------------------------

# p_fail = 0.5, p_pass = 0.9, alpha = beta = 0.05:
#   a pass adds ln(0.9 / 0.5) = 0.587787, a fail adds ln(0.1 / 0.5) = -1.609438,
#   A = ln(0.95 / 0.05) = 2.944439, B = ln(0.05 / 0.95) = -2.944439.


@CORE_XFAIL
def test_sprt_six_passes_pass_on_the_sixth() -> None:
    # 5 passes: 2.938936 < A; 6 passes: 3.526723 >= A.
    gate = stats.SequentialGate(0.5, 0.9)
    assert [gate.update(True) for _ in range(5)] == ["continue"] * 5
    assert gate.update(True) == "pass"
    assert (gate.runs, gate.passes, gate.decision) == (6, 6, "pass")


@CORE_XFAIL
def test_sprt_two_fails_fail_on_the_second() -> None:
    # 1 fail: -1.609438 > B; 2 fails: -3.218876 <= B.
    gate = stats.SequentialGate(0.5, 0.9)
    assert gate.update(False) == "continue"
    assert gate.update(False) == "fail"
    assert (gate.runs, gate.passes, gate.decision) == (2, 0, "fail")


@CORE_XFAIL
def test_sprt_truncates_at_max_runs() -> None:
    # pass, fail, pass: LLR = 2 * 0.587787 - 1.609438 = -0.433864, no boundary -> "fail".
    gate = stats.SequentialGate(0.5, 0.9, max_runs=3)
    assert [gate.update(o) for o in (True, False, True)] == ["continue", "continue", "fail"]
    # three passes: LLR = 1.763360 > 0, no boundary -> "pass".
    gate = stats.SequentialGate(0.5, 0.9, max_runs=3)
    assert [gate.update(True) for _ in range(3)] == ["continue", "continue", "pass"]


@CORE_XFAIL
def test_sprt_boundaries_are_inclusive() -> None:
    # p_fail = 0.25, p_pass = 0.5, alpha = beta = 0.2: a pass adds ln 2 and
    # A = ln(0.8 / 0.2) = ln 4, so two passes land exactly on A.
    gate = stats.SequentialGate(0.25, 0.5, alpha=0.2, beta=0.2)
    assert gate.update(True) == "continue"
    assert gate.update(True) == "pass"
    # p_fail = 0.5, p_pass = 0.75: a fail adds ln(0.25 / 0.5) = -ln 2 and
    # B = ln(0.2 / 0.8) = -ln 4, so two fails land exactly on B.
    gate = stats.SequentialGate(0.5, 0.75, alpha=0.2, beta=0.2)
    assert gate.update(False) == "continue"
    assert gate.update(False) == "fail"


@CORE_XFAIL
def test_sprt_starts_undecided() -> None:
    gate = stats.SequentialGate(0.5, 0.9)
    assert (gate.runs, gate.passes, gate.decision) == (0, 0, "continue")


def test_baseline_gate_always_runs_max_runs() -> None:
    for outcomes in ([True] * 10, [False] * 10, [True, False] * 5):
        gate = stats_baseline.SequentialGate(0.5, 0.9, max_runs=10)
        decisions = [gate.update(o) for o in outcomes]
        assert decisions[:-1] == ["continue"] * 9
        assert decisions[-1] != "continue"
        assert gate.runs == 10


def test_baseline_gate_majority_threshold() -> None:
    # threshold (0.6 + 0.9) / 2 = 0.75: 15/20 passes, 14/20 fails.
    for passes, expected in ((15, "pass"), (14, "fail"), (20, "pass"), (0, "fail")):
        gate = stats_baseline.SequentialGate(0.6, 0.9, max_runs=20)
        for i in range(20):
            gate.update(i < passes)
        assert (gate.passes, gate.decision) == (passes, expected)


def test_gate_update_after_decision_raises(impl: ModuleType) -> None:
    gate = impl.SequentialGate(0.5, 0.9, max_runs=1)
    assert gate.update(False) == "fail"
    with pytest.raises(RuntimeError, match="create a new SequentialGate"):
        gate.update(True)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((0.9, 0.5), {}, "p_fail < p_pass"),
        ((0.5, 0.5), {}, "p_fail < p_pass"),
        ((0.0, 0.5), {}, "p_fail < p_pass"),
        ((0.5, 1.0), {}, "p_fail < p_pass"),
        ((0.5, 0.9), {"alpha": 0.0}, "alpha"),
        ((0.5, 0.9), {"alpha": 1.0}, "alpha"),
        ((0.5, 0.9), {"beta": 0.0}, "beta"),
        ((0.5, 0.9), {"beta": 1.0}, "beta"),
        ((0.5, 0.9), {"max_runs": 0}, "max_runs"),
    ],
)
def test_gate_rejects_invalid_parameters(
    impl: ModuleType, args: tuple[float, float], kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        impl.SequentialGate(*args, **kwargs)


def test_gate_decides_by_max_runs(impl: ModuleType) -> None:
    impl.SequentialGate(0.6, 0.9)  # fail fast while unimplemented

    @SETTINGS
    @given(
        outcomes=st.lists(st.booleans(), min_size=20, max_size=20),
        max_runs=st.integers(1, 20),
    )
    def check(outcomes: list[bool], max_runs: int) -> None:
        gate = impl.SequentialGate(0.6, 0.9, max_runs=max_runs)
        for outcome in outcomes:
            if gate.update(outcome) != "continue":
                break
        assert gate.decision in ("pass", "fail")
        assert 1 <= gate.runs <= max_runs
        assert gate.passes == sum(outcomes[: gate.runs])

    check()


# --- stats_impl -------------------------------------------------------------------------


def test_load_defaults_to_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPLAY_STATS_IMPL", raising=False)
    assert stats_impl.load() is stats_baseline


def test_load_reads_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_STATS_IMPL", "core")
    assert stats_impl.load() is stats
    monkeypatch.setenv("REPLAY_STATS_IMPL", "")
    assert stats_impl.load() is stats_baseline


def test_load_by_name_overrides_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_STATS_IMPL", "core")
    assert stats_impl.load("baseline") is stats_baseline
    assert stats_impl.load("core") is stats


def test_load_rejects_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_STATS_IMPL", "fancy")
    with pytest.raises(ConfigError, match=r"'fancy'.*REPLAY_STATS_IMPL.*baseline, core"):
        stats_impl.load()
