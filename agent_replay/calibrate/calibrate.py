"""``replay calibrate``: measure how well the judge agrees with reference labels.

For every golden item with a reference label the judge is asked ``repeats``
times (samples ``0..repeats-1``, each cached on its own, see
:func:`agent_replay.judge.cache.cache_key`). Sample 0 is *the* verdict, the one
a ``judge`` check would use; it is compared with the reference per criterion
(and on :data:`~agent_replay.calibrate.golden.OVERALL`, "every criterion
passed", when the reference has whole-run labels). The other samples measure
self-consistency. Three probes look for known judge biases. Labels are encoded
``1`` = pass, ``0`` = fail for the statistics.

Every number comes from :func:`agent_replay.stats_impl.load`, and the report
records which implementation made it: under ``baseline`` the kappa columns hold
raw agreement, and such a report never gates (see
:func:`agent_replay.judge.calibration.report_blocker`).

Which items count where:

* kappa, agreement, confusion and the majority baseline for a criterion: the
  items with a reference label for it whose sample-0 verdict is known; ``n``
  is their count.
* self-consistency for a criterion: the same items, less those where some
  sample failed. Using the kappa items keeps both numbers about one
  population; the alternative (every item, labelled or not) would mix in
  items the report says nothing else about.
* probes: every labelled item (flips need no labels), except where a probe
  says otherwise.
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal

import structlog

from agent_replay.calibrate.golden import (
    OVERALL,
    GoldenItem,
    GoldenSet,
    portable,
    reference_labels,
)
from agent_replay.errors import ConfigError, JudgeCacheMiss, JudgeError
from agent_replay.judge.cache import JudgeCache, cache_key
from agent_replay.judge.calibration import (
    CalibrationReport,
    Confusion,
    CriterionCalibration,
    FlipStats,
    GroupStats,
    ProbeResult,
    SelfConsistency,
)
from agent_replay.judge.providers import JudgeProvider
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import JudgeInput
from agent_replay.judge.verdict import PROMPT_VERSION, JudgeResponse, system_prompt
from agent_replay.pricing import prices
from agent_replay.stats_impl import ENV_VAR as STATS_ENV_VAR
from agent_replay.stats_impl import StatsImpl
from agent_replay.stats_impl import load as load_stats

log = structlog.get_logger("agent_replay")

Reference = Literal["human", "objective"]
ProbeName = Literal["position", "verbosity", "self_preference"]
PROBES: Final[tuple[ProbeName, ...]] = ("position", "verbosity", "self_preference")

VERBOSITY_PADDING: Final = (
    "\n\nThank you for reaching out, and for your patience while I looked into this. "
    "I hope this reply is clear and helpful. If anything else comes up, please do not "
    "hesitate to get in touch; I am always happy to help. Have a wonderful day!"
)
"""Appended to the agent's final output by the verbosity probe: polite, longer and
empty. It states no fact and claims no action, so no criterion's answer should change
because of it; a judge that passes more runs with it is rewarding length or tone."""

OUTPUT_TOKENS_PER_CALL: Final = 289
"""Output tokens assumed per judge call in :func:`estimate`: the median ``output_tokens``
of the 21 committed retail verdicts in ``examples/retail_agent/judge-cache/``
(claude-opus-5-5, effort low, 3 criteria; range 227 to 691). ``make token-rates``
(``eval/token_rates.py``) re-derives this and :data:`CHARS_PER_TOKEN`."""

CHARS_PER_TOKEN: Final = 2.3
"""Characters per input token for :func:`estimate`. Measured on the same 21 committed
verdicts: rendered transcript characters over ``input_tokens`` ranged 2.24 to 2.46
(median 2.32; JSON tool results tokenise densely), and the retail system prompt's 5472
characters were read from the prompt cache as 2243 tokens (2.44, the verdict schema
included). The common 4-characters-per-token rule would undercount by about 40%."""

MIN_CACHED_PREFIX_TOKENS: Final = 512
"""The shortest prompt prefix claude-opus-5-5 caches. A system prompt estimated below it is
priced by :func:`estimate` as uncached input on every call (and the estimate says so);
the same minimum is assumed for every priced model."""

_FREE_PROVIDERS: Final = ("fake", "ollama")


# --- the plan: which verdicts a calibration needs ---------------------------------------


@dataclass(frozen=True)
class _Job:
    """One verdict to obtain: ``kind`` is ``main`` (samples of the real rubric and
    transcript) or a probe name."""

    kind: str
    item: str
    rubric: Rubric
    judge_input: JudgeInput
    sample: int


@dataclass(frozen=True)
class _Plan:
    rubric: Rubric
    reference: Reference
    labeller: str | None
    labels: dict[str, dict[str, bool]]
    """``{criterion: {item_id: passed}}``, rubric criteria and :data:`OVERALL` only."""
    items: tuple[GoldenItem, ...]
    """Items with at least one reference label, in golden-set order."""
    repeats: int
    probes: tuple[ProbeName, ...]
    jobs: tuple[_Job, ...]
    skipped: dict[str, str]
    """Probes that make no calls, with the reason."""

    @property
    def criteria(self) -> list[str]:
        """Labelled criteria in rubric order, then :data:`OVERALL` if labelled."""
        ids = [c for c in self.rubric.ids if c in self.labels]
        return ids + ([OVERALL] if OVERALL in self.labels else [])


def reversed_rubric(rubric: Rubric) -> Rubric:
    """``rubric`` with its criteria in reverse order: same ids and text, a new fingerprint
    (so its verdicts are cached apart)."""
    return rubric.model_copy(update={"criteria": tuple(reversed(rubric.criteria))})


def padded(judge_input: JudgeInput) -> JudgeInput | None:
    """The transcript with :data:`VERBOSITY_PADDING` after a string final output; ``None``
    when the output is not a string (there is no reply to pad)."""
    if not isinstance(judge_input.output, str):
        return None
    return judge_input.model_copy(update={"output": judge_input.output + VERBOSITY_PADDING})


def judge_model(judge_id: str) -> str:
    """The model in a judge id: the id less its provider prefix and any trailing
    ``key=value`` settings. ``anthropic:claude-opus-5-5:effort=low`` gives
    ``claude-opus-5-5``, ``ollama:qwen3:8b`` gives ``qwen3:8b``; an id without a colon is
    its own model.

    The self-preference probe treats the exact model id as the "family": a judge is
    compared with runs of the very model it is. Grouping related models (one vendor's
    sizes, say) would need a family table that goes stale with every release."""
    head, _, rest = judge_id.partition(":")
    if not rest:
        return head
    parts = rest.split(":")
    while len(parts) > 1 and "=" in parts[-1]:
        parts.pop()
    return ":".join(parts)


def _resolve_probes(probes: Sequence[str]) -> tuple[ProbeName, ...]:
    chosen: list[ProbeName] = []
    for name in probes:
        normal = name.replace("-", "_")
        match = next((p for p in PROBES if p == normal), None)
        if match is None:
            raise ConfigError(
                f"unknown probe {name!r}; use {', '.join(p.replace('_', '-') for p in PROBES)}"
            )
        if match not in chosen:
            chosen.append(match)
    return tuple(p for p in PROBES if p in chosen)


def _plan(
    golden: GoldenSet,
    rubric: Rubric,
    *,
    reference: Reference,
    labeller: str | None,
    repeats: int,
    probes: Sequence[str],
) -> _Plan:
    if repeats < 1:
        raise ConfigError(f"repeats must be at least 1, got {repeats}")
    chosen = _resolve_probes(probes)
    if labeller is None:
        names = golden.labellers(reference)
        labeller = names[0] if len(names) == 1 else None
    found = reference_labels(golden, rubric, reference=reference, labeller=labeller)
    labels = {c: v for c, v in found.items() if c in rubric.ids or c == OVERALL}
    if not labels:
        how = (
            "label it with `replay label`"
            if reference == "human"
            else "add runs with objective labels with `replay golden add`"
        )
        who = f" from labeller {labeller!r}" if labeller else ""
        raise ConfigError(
            f"the golden set {str(golden.directory)!r} has no current {reference} labels{who} "
            f"for rubric {rubric.name!r}; {how}"
        )
    labelled = {i for v in labels.values() for i in v}
    items = tuple(i for i in golden.items if i.id in labelled)

    jobs = [_Job("main", i.id, rubric, i.transcript, s) for i in items for s in range(repeats)]
    skipped: dict[str, str] = {}
    if "position" in chosen:
        flipped = reversed_rubric(rubric)
        if flipped.fingerprint == rubric.fingerprint:
            skipped["position"] = "the rubric has one criterion, so its order cannot change"
        else:
            jobs += [_Job("position", i.id, flipped, i.transcript, 0) for i in items]
    if "verbosity" in chosen:
        verbose = [(i, padded(i.transcript)) for i in items]
        jobs += [_Job("verbosity", i.id, rubric, t, 0) for i, t in verbose if t is not None]
        if all(t is None for _, t in verbose):
            skipped["verbosity"] = "no labelled item has a text final output to pad"
    return _Plan(rubric, reference, labeller, labels, items, repeats, chosen, tuple(jobs), skipped)


# --- the estimate --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """What a calibration would cost before it runs."""

    calls: int
    """Verdicts the provider would be asked for."""
    cached: int
    """Verdicts already in the cache (free)."""
    est_cost_usd: float | None
    """``None`` when the judge's model has no price (add it with ``REPLAY_PRICES_FILE``);
    0 for local and fake judges."""
    notes: tuple[str, ...] = ()
    """Assumptions behind ``est_cost_usd`` worth printing with it."""

    @property
    def total(self) -> int:
        return self.calls + self.cached


def _is_cached(cache: JudgeCache | None, judge_id: str, job: _Job) -> bool:
    return (
        cache is not None
        and cache.lookup(judge_id, job.rubric, job.judge_input, sample=job.sample) is not None
    )


def _cost(judge_id: str, jobs: Sequence[_Job]) -> tuple[float | None, tuple[str, ...]]:
    """The estimated cost of asking the provider for ``jobs``, with notes on how it was
    priced. The system prompt (instructions and rubric) is a cache write on the first call
    per rubric and a cache read after, unless it is shorter than
    :data:`MIN_CACHED_PREFIX_TOKENS` (then it is uncached input on every call); the
    transcript is uncached input; output is :data:`OUTPUT_TOKENS_PER_CALL`."""
    if judge_id.split(":", 1)[0] in _FREE_PROVIDERS:
        return 0.0, ()
    price = prices().get(judge_model(judge_id))
    if price is None:
        return None, ()
    total = 0.0
    seen: set[str] = set()
    short: dict[str, float] = {}
    for job in jobs:
        system = len(system_prompt(job.rubric)) / CHARS_PER_TOKEN
        if system < MIN_CACHED_PREFIX_TOKENS:
            short[job.rubric.fingerprint] = system
            total += system * price.input
        elif job.rubric.fingerprint in seen:
            total += system * price.cache_read
        else:
            seen.add(job.rubric.fingerprint)
            total += system * price.cache_write
        total += len(job.judge_input.render()) / CHARS_PER_TOKEN * price.input
        total += OUTPUT_TOKENS_PER_CALL * price.output
    notes: tuple[str, ...] = ()
    if short:
        tokens = round(max(short.values()))
        notes = (
            f"the system prompt (about {tokens} tokens) is under the "
            f"{MIN_CACHED_PREFIX_TOKENS}-token minimum the prompt cache needs, so it is "
            "priced as uncached input on every call",
        )
    return round(total / 1_000_000, 4), notes


def estimate(
    golden: GoldenSet,
    rubric: Rubric,
    provider: JudgeProvider,
    cache: JudgeCache | None,
    *,
    repeats: int = 3,
    probes: Sequence[str] = PROBES,
    reference: Reference = "human",
    labeller: str | None = None,
) -> Estimate:
    """How many provider calls :func:`calibrate` with these arguments would make, and
    roughly what they would cost. Reads the cache; never calls the provider."""
    plan = _plan(
        golden, rubric, reference=reference, labeller=labeller, repeats=repeats, probes=probes
    )
    missing = [j for j in plan.jobs if not _is_cached(cache, provider.judge_id, j)]
    cost, notes = _cost(provider.judge_id, missing)
    return Estimate(
        calls=len(missing),
        cached=len(plan.jobs) - len(missing),
        est_cost_usd=cost,
        notes=notes,
    )


# --- running the judge ----------------------------------------------------------------------


Verdicts = dict[str, bool]
"""``{criterion: passed}`` for one verdict, :data:`OVERALL` included."""


@dataclass
class _Tally:
    calls: int = 0
    cached: int = 0
    errors: int = 0
    cost: float | None = 0.0
    newest: datetime | None = None

    def saw(self, when: datetime) -> None:
        if self.newest is None or when > self.newest:
            self.newest = when


_GIVE_UP_AFTER: Final = 3
"""Consecutive failures, from the first call on, after which calibration stops: a judge
that never answers (no credentials, a wrong model name) would otherwise be asked about
every item only to produce an empty report. Later failures cost one item each."""


def _verdicts(response: JudgeResponse) -> Verdicts:
    out = {v.id: v.passed for v in response.verdicts}
    out[OVERALL] = all(out.values())
    return out


def _ask(
    job: _Job, provider: JudgeProvider, cache: JudgeCache | None, tally: _Tally
) -> Verdicts | None:
    """The verdict for ``job`` from the cache or the provider; ``None`` (logged and
    counted) when the provider fails. An offline miss propagates."""
    if cache is not None:
        entry = cache.lookup(provider.judge_id, job.rubric, job.judge_input, sample=job.sample)
        if entry is not None:
            tally.cached += 1
            tally.saw(entry.created_at)
            return _verdicts(entry.response())
        if cache.offline:
            raise _offline_miss(cache, provider, 1, 1)
    tally.calls += 1
    try:
        response = provider.evaluate(job.rubric, job.judge_input)
    except Exception as exc:  # a custom provider's bug costs one item, not the report
        reason = str(exc) if isinstance(exc, JudgeError) else f"{type(exc).__name__}: {exc}"
        log.warning(
            "calibrate.judge_error", item=job.item, kind=job.kind, sample=job.sample, error=reason
        )
        tally.errors += 1
        if tally.errors == tally.calls == _GIVE_UP_AFTER:
            raise JudgeError(
                f"the first {_GIVE_UP_AFTER} judge calls all failed, so calibration stopped "
                f"(last error: {reason}); fix the judge and rerun (cached verdicts are kept)"
            ) from exc
        return None
    if tally.cost is not None:
        tally.cost = None if response.cost_usd is None else tally.cost + response.cost_usd
    if cache is not None:
        entry = cache.put(
            _key(provider, job),
            provider.judge_id,
            job.rubric,
            job.judge_input,
            response,
            sample=job.sample,
        )
        tally.saw(entry.created_at)
    return _verdicts(response)


def _key(provider: JudgeProvider, job: _Job) -> str:
    return cache_key(provider.judge_id, job.rubric, job.judge_input, job.sample)


def _offline_miss(
    cache: JudgeCache, provider: JudgeProvider, missing: int, total: int
) -> JudgeCacheMiss:
    return JudgeCacheMiss(
        f"{missing} of the {total} verdicts this calibration needs are not in the judge cache "
        f"{str(cache.directory)!r} (judge {provider.judge_id}) and offline mode is on; "
        "calibrating offline needs every verdict cached (each labelled item once per repeat, "
        "plus once per probe). Point --cache-dir at the cache that holds them, lower "
        "--repeats or drop probes to match it, or run with the provider available (without "
        "--offline and REPLAY_JUDGE_OFFLINE)"
    )


# --- statistics -------------------------------------------------------------------------------


def _finite(value: float) -> float | None:
    """NaN (undefined) to ``None``; every other value clamped to kappa's range ``[-1, 1]``.
    That includes the CI bounds: the baseline's normal-approximation bootstrap interval
    can overshoot the range, and the report's ``method`` says the interval is clamped."""
    if math.isnan(value):
        return None
    return max(-1.0, min(1.0, value))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class _Stats:
    impl: StatsImpl
    name: str
    n_resamples: int
    seed: int

    def kappa(self, a: Sequence[int], b: Sequence[int]) -> float | None:
        return _finite(self.impl.cohens_kappa(a, b))

    def kappa_ci(self, a: Sequence[int], b: Sequence[int]) -> tuple[float | None, float | None]:
        low, high = self.impl.bootstrap_ci(
            self.impl.cohens_kappa, a, b, n_resamples=self.n_resamples, seed=self.seed
        )
        return _finite(low), _finite(high)

    def flips(self, original: Sequence[bool], perturbed: Sequence[bool]) -> FlipStats:
        n = len(original)
        to_pass = sum(1 for o, p in zip(original, perturbed, strict=True) if not o and p)
        to_fail = sum(1 for o, p in zip(original, perturbed, strict=True) if o and not p)
        flips = to_pass + to_fail
        if n == 0:
            return FlipStats(n=0, flips=0, to_pass=0, to_fail=0, rate=None)
        low, high = self.impl.wilson_interval(flips, n)
        return FlipStats(
            n=n,
            flips=flips,
            to_pass=to_pass,
            to_fail=to_fail,
            rate=flips / n,
            ci_low=_unit(low),
            ci_high=_unit(high),
        )


def _load_stats(name: str | None, n_resamples: int, seed: int) -> _Stats:
    chosen = name if name is not None else load_stats_name()
    impl = load_stats(chosen)
    try:  # fail before any judge call is paid for
        impl.cohens_kappa([1, 0], [1, 0])
        impl.wilson_interval(1, 2)
    except NotImplementedError as exc:
        raise ConfigError(
            f"the {chosen!r} statistics implementation is not written yet ({exc}); "
            f"calibrate with {STATS_ENV_VAR}=baseline (its reports never gate)"
        ) from exc
    if n_resamples < 1:
        raise ConfigError(f"n_resamples must be at least 1, got {n_resamples}")
    return _Stats(impl, chosen, n_resamples, seed)


def load_stats_name() -> str:
    """The statistics implementation in effect: ``REPLAY_STATS_IMPL``, else ``baseline``."""
    return os.environ.get(STATS_ENV_VAR) or "baseline"


def _criterion(
    stats: _Stats,
    criterion: str,
    items: Sequence[str],
    labels: dict[str, bool],
    runs: Sequence[dict[str, Verdicts | None]],
) -> CriterionCalibration | None:
    """The calibration of one criterion over ``items`` (labelled, judged at sample 0)."""
    first = runs[0]
    judged = [i for i in items if i in labels and first.get(i) is not None]
    if not judged:
        return None
    judge = [int(_get(first, i)[criterion]) for i in judged]
    ref = [int(labels[i]) for i in judged]
    n = len(judged)
    kappa = stats.kappa(judge, ref)
    low, high = stats.kappa_ci(judge, ref) if kappa is not None else (None, None)
    passes = sum(ref)
    majority: Literal["pass", "fail"] = "pass" if passes >= n - passes else "fail"
    return CriterionCalibration(
        kappa=kappa,
        ci_low=low,
        ci_high=high,
        agreement=sum(j == r for j, r in zip(judge, ref, strict=True)) / n,
        n=n,
        confusion=Confusion(
            both_pass=sum(1 for j, r in zip(judge, ref, strict=True) if j and r),
            judge_pass_reference_fail=sum(
                1 for j, r in zip(judge, ref, strict=True) if j and not r
            ),
            judge_fail_reference_pass=sum(
                1 for j, r in zip(judge, ref, strict=True) if not j and r
            ),
            both_fail=sum(1 for j, r in zip(judge, ref, strict=True) if not j and not r),
        ),
        majority_label=majority,
        majority_agreement=max(passes, n - passes) / n,
        self_consistency=_self_consistency(stats, criterion, judged, runs),
    )


def _get(run: dict[str, Verdicts | None], item: str) -> Verdicts:
    verdicts = run[item]
    assert verdicts is not None, "callers filter out unanswered items"
    return verdicts


def _self_consistency(
    stats: _Stats, criterion: str, items: Sequence[str], runs: Sequence[dict[str, Verdicts | None]]
) -> SelfConsistency | None:
    if len(runs) < 2:
        return None
    answered = [i for i in items if all(r.get(i) is not None for r in runs)]
    if not answered:
        return None
    series = [[int(_get(r, i)[criterion]) for i in answered] for r in runs]
    unanimous = sum(1 for col in zip(*series, strict=True) if len(set(col)) == 1)
    pairwise = [stats.kappa(a, b) for a, b in itertools.combinations(series, 2)]
    defined = [k for k in pairwise if k is not None]
    return SelfConsistency(
        runs=len(runs),
        n=len(answered),
        unanimous=unanimous / len(answered),
        mean_pairwise_kappa=_finite(sum(defined) / len(defined)) if defined else None,
    )


# --- probes ---------------------------------------------------------------------------------


_POSITION = (
    "Criterion order: the rubric with its criteria in reverse order (same ids and text), "
    "each item judged once; a flip is a verdict that changed. The judge grades one run at a "
    "time, so there is no answer A/B order to swap; criterion order is the position effect "
    "that applies (pairwise position bias is measured on LLMBar)."
)
_VERBOSITY = (
    "Verbosity: a fixed polite paragraph that adds no facts or claims appended to the final "
    "output, each item judged once; to_pass flips (fail became pass) are the bias."
)
_SELF_PREFERENCE = (
    "Self-preference: judge leniency on runs whose agent model is the judge's own model "
    "({model}) against runs by other models; a higher false-pass rate on same_model runs "
    "is the bias."
)


def _flip_probe(
    stats: _Stats,
    probe: Literal["position", "verbosity"],
    description: str,
    criteria: Sequence[str],
    original: dict[str, Verdicts | None],
    perturbed: dict[str, Verdicts | None],
) -> ProbeResult:
    both = [i for i, v in perturbed.items() if v is not None and original.get(i) is not None]
    if not both:
        return ProbeResult(
            probe=probe, description=description, skipped="no item was judged both ways"
        )
    flips = {
        c: stats.flips([_get(original, i)[c] for i in both], [_get(perturbed, i)[c] for i in both])
        for c in criteria
    }
    return ProbeResult(probe=probe, description=description, flips=flips)


def _group(
    criterion: str, items: Sequence[str], labels: dict[str, bool], first: dict[str, Verdicts | None]
) -> GroupStats:
    rows = [
        (_get(first, i)[criterion], labels[i])
        for i in items
        if i in labels and first.get(i) is not None
    ]
    n = len(rows)
    failed = [j for j, r in rows if not r]
    return GroupStats(
        n=n,
        judge_pass_rate=sum(j for j, _ in rows) / n if n else None,
        reference_pass_rate=sum(r for _, r in rows) / n if n else None,
        false_pass_rate=sum(failed) / len(failed) if failed else None,
        reference_fail_n=len(failed),
    )


def _self_preference(
    plan: _Plan, judge_id: str, criteria: Sequence[str], first: dict[str, Verdicts | None]
) -> ProbeResult:
    model = judge_model(judge_id)
    description = _SELF_PREFERENCE.format(model=model)
    known = [i for i in plan.items if i.agent_models]
    if not known:
        return ProbeResult(
            probe="self_preference",
            description=description,
            skipped="no labelled item records the model that drove its agent",
        )
    same = [i.id for i in known if model in i.agent_models]
    other = [i.id for i in known if model not in i.agent_models]
    if not same or not other:
        return ProbeResult(
            probe="self_preference",
            description=description,
            skipped=(
                f"no labelled item was produced by {model}"
                if not same
                else f"every labelled item with a known agent model was produced by {model}"
            ),
        )
    groups = {
        c: {
            "same_model": _group(c, same, plan.labels[c], first),
            "other_model": _group(c, other, plan.labels[c], first),
        }
        for c in criteria
    }
    return ProbeResult(probe="self_preference", description=description, groups=groups)


# --- the report -------------------------------------------------------------------------------


def _method(stats: _Stats, plan: _Plan, errors: int) -> str:
    resamples = f"{stats.n_resamples} resamples, seed {stats.seed}"
    if stats.name == "baseline":
        measure = (
            "raw agreement in the kappa column (REPLAY_STATS_IMPL=baseline; set it to core for "
            f"Cohen's kappa), with a normal-approximation bootstrap CI ({resamples}, "
            "clamped to [-1, 1])"
        )
        pairwise = "mean pairwise raw agreement"
        interval = "Wald"
    else:
        measure = f"Cohen's kappa (unweighted), 95% bootstrap CI ({resamples}, clamped to [-1, 1])"
        pairwise = "mean pairwise Cohen's kappa"
        interval = "Wilson" if stats.name == "core" else f"{stats.name} wilson_interval"
    text = (
        f"The judge's first verdict per item against {plan.reference} labels"
        f"{f' by {plan.labeller}' if plan.labeller else ''} (1 = pass, 0 = fail): {measure}"
    )
    if plan.repeats > 1:
        text += f"; self-consistency over {plan.repeats} verdicts per item: {pairwise}"
    if plan.probes:
        text += f"; probe flip rates with {interval} 95% intervals"
    if errors:
        text += f"; {errors} judge call(s) failed and their items were left out where needed"
    return text + "."


def calibrate(
    golden: GoldenSet,
    rubric: Rubric,
    provider: JudgeProvider,
    *,
    cache: JudgeCache | None,
    reference: Reference = "human",
    labeller: str | None = None,
    repeats: int = 3,
    probes: Sequence[str] = PROBES,
    stats: str | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CalibrationReport:
    """Judge the golden set and measure the judge against its ``reference`` labels.

    ``labeller`` picks one labeller when the reference has several (a
    :class:`ConfigError` otherwise). ``stats`` names the statistics implementation
    (default ``REPLAY_STATS_IMPL``, then ``baseline``). A provider failure on an
    item drops that item where its verdict is needed and is counted in the
    report's ``method``; an offline cache miss raises :class:`JudgeCacheMiss` before
    any call is made.

    ``golden_set`` is the directory relative to the working directory when it is inside
    it, so a relative and an absolute GOLDEN_DIR give the same report.

    ``created_at`` is the newest verdict used (a fresh one is stamped by the
    cache's clock), so a report recomputed from the same cache is byte-identical;
    without a cache it is ``clock()``.
    """
    plan = _plan(
        golden, rubric, reference=reference, labeller=labeller, repeats=repeats, probes=probes
    )
    impl = _load_stats(stats, n_resamples, seed)
    if cache is not None and cache.offline:
        missing = sum(1 for j in plan.jobs if not _is_cached(cache, provider.judge_id, j))
        if missing:
            raise _offline_miss(cache, provider, missing, len(plan.jobs))

    tally = _Tally()
    results: dict[tuple[str, int], dict[str, Verdicts | None]] = {}
    for job in plan.jobs:
        results.setdefault((job.kind, job.sample), {})[job.item] = _ask(job, provider, cache, tally)
    runs = [results.get(("main", s), {}) for s in range(plan.repeats)]
    first = runs[0]
    criteria = plan.criteria
    item_ids = [i.id for i in plan.items]

    report_criteria: dict[str, CriterionCalibration] = {}
    for criterion in criteria:
        calibration = _criterion(impl, criterion, item_ids, plan.labels[criterion], runs)
        if calibration is None:
            log.warning("calibrate.criterion_unjudged", criterion=criterion)
            continue
        report_criteria[criterion] = calibration
    flip_criteria = [*rubric.ids, OVERALL]
    probe_results: list[ProbeResult] = []
    for probe in plan.probes:
        if probe in plan.skipped:
            description = _POSITION if probe == "position" else _VERBOSITY
            probe_results.append(
                ProbeResult(probe=probe, description=description, skipped=plan.skipped[probe])
            )
        elif probe == "position":
            probe_results.append(
                _flip_probe(
                    impl, probe, _POSITION, flip_criteria, first, results.get((probe, 0), {})
                )
            )
        elif probe == "verbosity":
            probe_results.append(
                _flip_probe(
                    impl, probe, _VERBOSITY, flip_criteria, first, results.get((probe, 0), {})
                )
            )
        else:
            probe_results.append(
                _self_preference(plan, provider.judge_id, list(report_criteria), first)
            )

    log.info(
        "calibrate.done",
        rubric=rubric.name,
        judge_id=provider.judge_id,
        items=len(plan.items),
        calls=tally.calls,
        cached=tally.cached,
        errors=tally.errors,
    )
    return CalibrationReport(
        rubric=rubric.name,
        rubric_fingerprint=rubric.fingerprint,
        judge_id=provider.judge_id,
        prompt_version=PROMPT_VERSION,
        created_at=tally.newest if tally.newest is not None else clock(),
        n_items=len(plan.items),
        method=_method(impl, plan, tally.errors),
        criteria=report_criteria,
        reference=plan.reference,
        labeller=plan.labeller,
        stats_impl=impl.name,
        golden_set=portable(golden.directory),
        golden_fingerprint=golden.fingerprint,
        repeats=plan.repeats,
        probes=tuple(probe_results),
        judge_calls=tally.calls,
        cached_calls=tally.cached,
        cost_usd=None if tally.cost is None else round(tally.cost, 8),
    )
