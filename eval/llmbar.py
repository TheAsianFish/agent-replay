"""Is the judge robust? Its accuracy and position bias on LLMBar.

LLMBar (Zeng et al., "Evaluating Large Language Models at Evaluating Instruction
Following", ICLR 2024, https://github.com/princeton-nlp/LLMBar, MIT) has 419
instances: an instruction, two outputs, and which one follows the instruction.
``Natural`` (100) are pairs from existing preference data, cleaned up;
the four ``Adversarial`` subsets (``Neighbor`` 134, ``GPTInst`` 92, ``GPTOut``
47, ``Manual`` 46) pair a correct output with one that answers a nearby
instruction or looks better on the surface. The data is vendored at a pinned
commit in ``eval/data/llmbar/`` (``eval/llmbar_fetch.py`` re-downloads and
verifies it).

Every instance is judged twice, by the pairwise judge in
:mod:`agent_replay.judge.pairwise`: once in the original order (Output (a) is
``output_1``) and once swapped. Reported per subset, for Adversarial (the four
pooled) and overall:

- accuracy in the original order, in the swapped order, and averaged over both
  orders (each order a trial; LLMBar's headline "average accuracy");
- consistent-correct: the fraction of instances judged correctly in *both* orders;
- position-bias rate: the fraction of instances whose chosen *output* changes
  when the order is swapped (one minus LLMBar's "positional agreement"), and,
  among those flips, the fraction that went to whichever output came first.

Accuracies count only answered trials; instances with a judge error in either
order are left out of consistent-correct and position bias, and every error is
counted and reported. Intervals come from ``agent_replay.stats_impl`` (Wald
under the default ``baseline`` implementation, Wilson under ``core``); the
averaged accuracy treats an instance's two orders as independent trials, which
they are not, so its interval is too narrow.

Verdicts are cached on disk (``--cache-dir``), so an interrupted run resumes and
a finished one re-runs for free; ``--offline`` makes a cache miss an error
instead of a call. ``--dry-run`` prints the number of calls still needed and
their estimated cost, then exits. With ``--limit N`` each subset is a seeded
random sample of N instances (``--seed``), kept in file order.

Usage::

    uv run python eval/llmbar.py --dry-run
    uv run python eval/llmbar.py --out eval/results/llmbar.json      # needs ANTHROPIC_API_KEY
    uv run python eval/llmbar.py --provider fake-longer --limit 20   # offline baseline

Exit status: 0 when every trial was answered, 1 when some had judge errors, 2 on
a configuration, data or budget problem (nothing judged after a budget stop is lost:
re-running resumes from the cache).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from agent_replay import stats_impl
from agent_replay.errors import ConfigError, JudgeError
from agent_replay.flake import interval_name
from agent_replay.judge.pairwise import (
    PAIRWISE_INSTRUCTIONS,
    PAIRWISE_PROMPT_VERSION,
    FakePairwiseJudge,
    PairwiseCache,
    PairwiseDecision,
    PairwiseJudge,
    PairwiseOutput,
    make_pairwise_judge,
    render_pair,
)
from agent_replay.pricing import prices
from eval.llmbar_fetch import COMMIT, REPOSITORY, FetchError, parse_records

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "eval" / "data" / "llmbar"
DEFAULT_CACHE = ROOT / "eval" / "llmbar-cache"
SUBSETS = (
    "Natural",
    "Adversarial/Neighbor",
    "Adversarial/GPTInst",
    "Adversarial/GPTOut",
    "Adversarial/Manual",
)
ADVERSARIAL = tuple(s for s in SUBSETS if s.startswith("Adversarial/"))
CONFIDENCE = 0.95
CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_ESTIMATE = 600
"""Assumed output tokens per call for the cost estimate: the brief reasoning and the JSON
verdict plus the model's own (always on, effort ``low``) thinking. The real run reports
the real cost."""

Order = Literal["original", "swapped"]
ORDERS: tuple[Order, Order] = ("original", "swapped")


class BudgetExceeded(Exception):
    """The run spent more than ``--max-cost-usd``."""


@dataclass(frozen=True)
class Instance:
    subset: str
    index: int
    instruction: str
    output_1: str
    output_2: str
    label: int
    """1 or 2: the output that follows the instruction."""

    def pair(self, order: Order) -> tuple[str, str]:
        """``(output_a, output_b)`` as shown to the judge."""
        if order == "original":
            return self.output_1, self.output_2
        return self.output_2, self.output_1


def chosen_output(order: Order, winner: str) -> int:
    """Which dataset output (1 or 2) the judge picked when it answered ``winner`` in ``order``."""
    first = winner == "a"
    if order == "original":
        return 1 if first else 2
    return 2 if first else 1


@dataclass(frozen=True)
class Trial:
    """One judge call: an instance in one order."""

    instance: Instance
    order: Order

    @property
    def texts(self) -> tuple[str, str, str]:
        a, b = self.instance.pair(self.order)
        return self.instance.instruction, a, b


@dataclass(frozen=True)
class TrialResult:
    chosen: int | None
    """The dataset output the judge picked; ``None`` on a judge error."""
    error: str | None = None
    cached: bool = False
    cost_usd: float | None = None


# --- data ---------------------------------------------------------------------------


def load(data_dir: Path = DATA_DIR, subsets: Sequence[str] = SUBSETS) -> list[Instance]:
    """Every instance of ``subsets`` from ``<data_dir>/<subset>/dataset.json``, in file order.

    Raises:
        ConfigError: a file is missing or not a list of LLMBar records.
    """
    instances: list[Instance] = []
    for subset in subsets:
        path = data_dir / subset / "dataset.json"
        try:
            records = parse_records(path.read_bytes(), str(path))
        except OSError as exc:
            raise ConfigError(
                f"cannot read {path} ({exc.strerror or exc}); restore it with "
                "`make data-llmbar` or pass --data"
            ) from exc
        except FetchError as exc:
            raise ConfigError(f"{exc}; restore the file with `make data-llmbar`") from exc
        instances += [
            Instance(
                subset=subset,
                index=i,
                instruction=str(r["input"]),
                output_1=str(r["output_1"]),
                output_2=str(r["output_2"]),
                label=int(r["label"]),
            )
            for i, r in enumerate(records)
        ]
    return instances


def select(instances: Sequence[Instance], limit: int | None, seed: int = 0) -> list[Instance]:
    """At most ``limit`` instances per subset: a random sample seeded by ``seed`` and the
    subset name (so adding a subset does not change another's sample), kept in file order.
    A sample rather than the first N because the upstream files are not shuffled."""
    if limit is None:
        return list(instances)
    chosen: list[Instance] = []
    for subset in dict.fromkeys(i.subset for i in instances):
        members = [i for i in instances if i.subset == subset]
        if len(members) > limit:
            rng = random.Random(f"llmbar:{seed}:{subset}")  # noqa: S311 - sampling, not crypto
            members = sorted(rng.sample(members, limit), key=lambda i: i.index)
        chosen += members
    return chosen


def trials(instances: Iterable[Instance]) -> list[Trial]:
    return [Trial(i, order) for i in instances for order in ORDERS]


# --- judging --------------------------------------------------------------------------


def _judge_one(judge: PairwiseJudge, cache: PairwiseCache, trial: Trial) -> TrialResult:
    try:
        verdict, hit = cache.verdict(judge, *trial.texts)
    except JudgeError as exc:  # includes JudgeCacheMiss in offline mode
        return TrialResult(chosen=None, error=str(exc))
    return TrialResult(
        chosen=chosen_output(trial.order, verdict.winner), cached=hit, cost_usd=verdict.cost_usd
    )


def run(
    instances: Sequence[Instance],
    judge: PairwiseJudge,
    cache: PairwiseCache,
    *,
    concurrency: int = 1,
    max_cost_usd: float | None = None,
) -> dict[Trial, TrialResult]:
    """Judge every instance in both orders. Results come back keyed by trial, so the report
    does not depend on which call finished first.

    Up to ``concurrency`` calls run at once, in batches; after each batch the money spent on
    new (uncached) verdicts is checked against ``max_cost_usd``.

    Raises:
        BudgetExceeded: the spend passed ``max_cost_usd``; everything judged so far is cached.
    """
    if concurrency < 1:
        raise ConfigError(f"--concurrency must be at least 1, got {concurrency}")
    todo = trials(instances)
    results: dict[Trial, TrialResult] = {}
    spent = 0.0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(todo), concurrency):
            batch = todo[start : start + concurrency]
            for trial, result in zip(
                batch, pool.map(lambda t: _judge_one(judge, cache, t), batch), strict=True
            ):
                results[trial] = result
                if not result.cached:
                    spent += result.cost_usd or 0.0
            if max_cost_usd is not None and spent > max_cost_usd:
                raise BudgetExceeded(
                    f"spent ${spent:.4f}, more than --max-cost-usd {max_cost_usd}, after "
                    f"{len(results)} of {len(todo)} calls; the verdicts so far are cached, so "
                    "re-running with a higher limit resumes where this stopped"
                )
    return results


# --- cost estimate ------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    calls: int
    cached: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    """``None`` when the judge's price is unknown."""


def _schema_chars() -> int:
    return len(json.dumps(PairwiseOutput.model_json_schema()))


def estimate(
    instances: Sequence[Instance],
    judge: PairwiseJudge,
    cache: PairwiseCache,
    *,
    output_tokens: int = OUTPUT_TOKENS_ESTIMATE,
) -> Estimate:
    """The calls still needed after cache hits and what they would cost.

    Input tokens are approximated as characters / 4 over the instructions, the verdict
    schema and the rendered pair; output tokens as ``output_tokens`` per call. Prices
    come from :mod:`agent_replay.pricing` for the judge's ``model``; prompt-cache
    discounts on the shared instructions are ignored, so this errs high on input. A
    fake judge costs nothing; any other judge without a priced ``model`` is unknown.
    """
    overhead = len(PAIRWISE_INSTRUCTIONS) + _schema_chars()
    calls = cached = input_tokens = 0
    for trial in trials(instances):
        if cache.get(judge.judge_id, *trial.texts) is not None:
            cached += 1
            continue
        calls += 1
        input_tokens += math.ceil((overhead + len(render_pair(*trial.texts))) / CHARS_PER_TOKEN)
    out = calls * output_tokens
    cost: float | None
    if isinstance(judge, FakePairwiseJudge) or calls == 0:
        cost = 0.0
    else:
        model = getattr(judge, "model", None)
        price = prices().get(model) if isinstance(model, str) else None
        cost = (
            None
            if price is None
            else round((input_tokens * price.input + out * price.output) / 1_000_000, 4)
        )
    return Estimate(calls, cached, input_tokens, out, cost)


def describe_estimate(est: Estimate, judge_id: str) -> str:
    cost = (
        "unknown (the model is not in the price table)"
        if est.cost_usd is None
        else f"${est.cost_usd:.2f}"
    )
    return (
        f"{est.calls} judge calls needed ({est.cached} cached) with {judge_id}: "
        f"~{est.input_tokens} input + ~{est.output_tokens} output tokens, estimated cost {cost}"
    )


# --- metrics ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Proportion:
    """``successes / trials`` with its interval from the selected statistics implementation."""

    successes: int
    trials: int
    interval: tuple[float, float]

    @classmethod
    def of(cls, successes: int, trials_: int, stats_name: str) -> Proportion:
        impl = stats_impl.load(stats_name)
        lo, hi = impl.wilson_interval(successes, trials_, confidence=CONFIDENCE)
        return cls(successes, trials_, (lo, hi))

    @property
    def rate(self) -> float | None:
        return self.successes / self.trials if self.trials else None

    def json(self) -> dict[str, JsonValue]:
        rate = self.rate
        return {
            "successes": self.successes,
            "trials": self.trials,
            "rate": None if rate is None else round(rate, 4),
            "interval": [round(self.interval[0], 4), round(self.interval[1], 4)],
        }

    def cell(self) -> str:
        rate = self.rate
        if rate is None:
            return "—"
        return f"{rate:.1%} [{self.interval[0]:.1%}, {self.interval[1]:.1%}]"


@dataclass(frozen=True)
class Row:
    """One instance's outcome in both orders; ``chosen_*`` is the dataset output picked (1 or
    2), ``None`` when that trial ended in a judge error."""

    subset: str
    index: int
    label: int
    chosen_original: int | None
    chosen_swapped: int | None
    error_original: str | None = None
    error_swapped: str | None = None
    cost_usd: float = 0.0

    def json(self) -> dict[str, JsonValue]:
        return {
            "subset": self.subset,
            "index": self.index,
            "label": self.label,
            "chosen_original": self.chosen_original,
            "chosen_swapped": self.chosen_swapped,
            "error_original": self.error_original,
            "error_swapped": self.error_swapped,
            "cost_usd": round(self.cost_usd, 8),
        }


@dataclass(frozen=True)
class Metrics:
    group: str
    instances: int
    errors: int
    accuracy_original: Proportion
    accuracy_swapped: Proportion
    accuracy_average: Proportion
    consistent_correct: Proportion
    position_bias: Proportion
    flips_to_first: Proportion

    def json(self) -> dict[str, JsonValue]:
        return {
            "group": self.group,
            "instances": self.instances,
            "errors": self.errors,
            "accuracy_original": self.accuracy_original.json(),
            "accuracy_swapped": self.accuracy_swapped.json(),
            "accuracy_average": self.accuracy_average.json(),
            "consistent_correct": self.consistent_correct.json(),
            "position_bias": self.position_bias.json(),
            "flips_to_first": self.flips_to_first.json(),
        }


def summarise(group: str, rows: Sequence[Row], stats_name: str = "baseline") -> Metrics:
    """The metrics over ``rows``. Accuracies count answered trials; consistent-correct and
    position bias count instances answered in both orders. Intervals come from the
    ``stats_name`` implementation of :mod:`agent_replay.stats_impl`."""
    p = functools.partial(Proportion.of, stats_name=stats_name)
    original = [r for r in rows if r.chosen_original is not None]
    swapped = [r for r in rows if r.chosen_swapped is not None]
    complete = [r for r in original if r.chosen_swapped is not None]
    right_original = sum(1 for r in original if r.chosen_original == r.label)
    right_swapped = sum(1 for r in swapped if r.chosen_swapped == r.label)
    both_right = sum(1 for r in complete if r.chosen_original == r.label == r.chosen_swapped)
    flips = [r for r in complete if r.chosen_original != r.chosen_swapped]
    # Output 1 is first in the original order and output 2 first in the swapped one, so a
    # flip towards the first position picks output 1, then output 2.
    to_first = sum(1 for r in flips if r.chosen_original == 1)
    return Metrics(
        group=group,
        instances=len(rows),
        errors=2 * len(rows) - len(original) - len(swapped),
        accuracy_original=p(right_original, len(original)),
        accuracy_swapped=p(right_swapped, len(swapped)),
        accuracy_average=p(right_original + right_swapped, len(original) + len(swapped)),
        consistent_correct=p(both_right, len(complete)),
        position_bias=p(len(flips), len(complete)),
        flips_to_first=p(to_first, len(flips)),
    )


def rows_of(instances: Sequence[Instance], results: dict[Trial, TrialResult]) -> list[Row]:
    rows: list[Row] = []
    for inst in instances:
        original = results[Trial(inst, "original")]
        swapped = results[Trial(inst, "swapped")]
        rows.append(
            Row(
                subset=inst.subset,
                index=inst.index,
                label=inst.label,
                chosen_original=original.chosen,
                chosen_swapped=swapped.chosen,
                error_original=original.error,
                error_swapped=swapped.error,
                cost_usd=(original.cost_usd or 0.0) + (swapped.cost_usd or 0.0),
            )
        )
    return rows


def groups(rows: Sequence[Row], stats_name: str = "baseline") -> list[Metrics]:
    """Each subset present, in dataset order, then Adversarial (pooled) and Overall."""
    present = list(dict.fromkeys(r.subset for r in rows))
    out = [summarise(s, [r for r in rows if r.subset == s], stats_name) for s in present]
    adversarial = [r for r in rows if r.subset in ADVERSARIAL]
    if adversarial:
        out.append(summarise("Adversarial (pooled)", adversarial, stats_name))
    out.append(summarise("Overall", rows, stats_name))
    return out


def error_counts(rows: Sequence[Row]) -> dict[str, int]:
    counts = Counter(e for r in rows for e in (r.error_original, r.error_swapped) if e is not None)
    return dict(sorted(counts.items()))


def report(
    rows: Sequence[Row],
    *,
    judge_id: str,
    stats_name: str,
    limit: int | None,
    seed: int,
    timestamp: str | None = None,
) -> dict[str, JsonValue]:
    """The result document. Byte-stable under ``json.dumps(..., sort_keys=True)``: it holds
    no run-dependent values (cache hits, wall time) and a timestamp only when given one."""
    doc: dict[str, JsonValue] = {
        "dataset": {
            "name": "LLMBar",
            "citation": "Zeng et al., Evaluating Large Language Models at Evaluating "
            "Instruction Following, ICLR 2024",
            "source": f"https://github.com/{REPOSITORY}",
            "commit": COMMIT,
        },
        "judge_id": judge_id,
        "prompt_version": PAIRWISE_PROMPT_VERSION,
        "stats_impl": stats_name,
        "interval": f"{CONFIDENCE:.0%} {interval_name(stats_name)}",
        "selection": {"limit_per_subset": limit, "seed": seed if limit is not None else None},
        "groups": [g.json() for g in groups(rows, stats_name)],
        "errors": dict(error_counts(rows)),
        "cost_usd": round(sum(r.cost_usd for r in rows), 6),
        "instances": [r.json() for r in rows],
    }
    if timestamp is not None:
        doc["generated_at"] = timestamp
    return doc


def markdown(rows: Sequence[Row], *, judge_id: str, stats_name: str) -> str:
    lines = [
        f"LLMBar ({REPOSITORY}@{COMMIT[:12]}), judge `{judge_id}`, pairwise prompt "
        f"v{PAIRWISE_PROMPT_VERSION}. Intervals: {CONFIDENCE:.0%} {interval_name(stats_name)} "
        f"(`REPLAY_STATS_IMPL={stats_name}`); the averaged accuracy's interval treats the two "
        "orders of an instance as independent trials, so it is too narrow.",
        "",
        "| Group | n | Acc. original | Acc. swapped | Acc. average | Consistent-correct "
        "| Position bias | Flips to first | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in groups(rows, stats_name):
        lines.append(
            f"| {g.group} | {g.instances} | {g.accuracy_original.cell()} "
            f"| {g.accuracy_swapped.cell()} | {g.accuracy_average.cell()} "
            f"| {g.consistent_correct.cell()} | {g.position_bias.cell()} "
            f"| {g.flips_to_first.successes}/{g.flips_to_first.trials} | {g.errors} |"
        )
    errors = error_counts(rows)
    if errors:
        lines += ["", "Judge errors (trials):"]
        lines += [f"- {count} x {message}" for message, count in errors.items()]
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------------


def _longer(_i: str, a: str, b: str) -> PairwiseDecision:
    return "a" if len(a) >= len(b) else "b"


def _shorter(_i: str, a: str, b: str) -> PairwiseDecision:
    return "a" if len(a) <= len(b) else "b"


FAKES: dict[str, Callable[[str, str, str], PairwiseDecision]] = {
    "fake-first": lambda _i, _a, _b: "a",
    "fake-second": lambda _i, _a, _b: "b",
    "fake-longer": _longer,
    "fake-shorter": _shorter,
}
"""Offline reference judges: the two extreme position biases and the two length heuristics
(the adversarial subsets are built so that surface quality misleads)."""


def make_judge(provider: str, model: str | None, effort: str | None) -> PairwiseJudge:
    if provider in FAKES:
        if model is not None or effort is not None:
            raise ConfigError(f"--model and --effort apply to the anthropic judge, not {provider}")
        return FakePairwiseJudge(FAKES[provider], name=provider.removeprefix("fake-"))
    return make_pairwise_judge(provider, model=model, effort=effort)


def _subsets(chosen: Sequence[str] | None) -> list[str]:
    if not chosen:
        return list(SUBSETS)
    out: list[str] = []
    for name in chosen:
        expanded = ADVERSARIAL if name == "Adversarial" else (name,)
        out += [s for s in expanded if s not in out]
    return [s for s in SUBSETS if s in out]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="the LLMBar data directory")
    parser.add_argument(
        "--subset",
        action="append",
        choices=[*SUBSETS, "Adversarial"],
        help="judge only this subset (repeatable; Adversarial = all four); default all",
    )
    parser.add_argument("--limit", type=int, help="at most N instances per subset (seeded sample)")
    parser.add_argument("--seed", type=int, default=0, help="the --limit sample's seed")
    parser.add_argument(
        "--provider",
        default="anthropic",
        help=f"anthropic (default), {', '.join(FAKES)}, or module:factory",
    )
    parser.add_argument("--model", help="the anthropic judge model (default claude-opus-5-5)")
    parser.add_argument("--effort", help="the anthropic judge effort (default low)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true", help="judge from the cache only")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the calls needed and their cost, then exit"
    )
    parser.add_argument("--max-cost-usd", type=float, help="refuse to spend more than this")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel judge calls")
    parser.add_argument(
        "--output-tokens", type=int, default=OUTPUT_TOKENS_ESTIMATE, help="per call, for estimates"
    )
    parser.add_argument("--out", type=Path, help="write the JSON result here")
    parser.add_argument("--timestamp", help="recorded as generated_at in the JSON (optional)")
    return parser


def _stats_name() -> str:
    """The ``REPLAY_STATS_IMPL`` implementation, checked before any judge call: an unknown
    or not yet implemented one would otherwise only fail after the calls were paid for."""
    impl = stats_impl.load()
    name = next(n for n in stats_impl.IMPLS if stats_impl.load(n) is impl)
    try:
        impl.wilson_interval(1, 2, confidence=CONFIDENCE)
    except NotImplementedError as exc:
        raise ConfigError(
            f"the {name!r} statistics implementation has no wilson_interval yet ({exc}); "
            f"unset {stats_impl.ENV_VAR} or set it to baseline"
        ) from exc
    return name


def _budget_problem(est: Estimate, max_cost_usd: float | None) -> str | None:
    """Why the planned calls may not start under ``--max-cost-usd``, if they may not."""
    if max_cost_usd is None or est.calls == 0:
        return None
    if est.cost_usd is None:
        return (
            "cannot check --max-cost-usd: the judge's price is unknown; add the model with "
            "REPLAY_PRICES_FILE or drop --max-cost-usd"
        )
    if est.cost_usd > max_cost_usd:
        return (
            f"the estimate ${est.cost_usd:.2f} is over --max-cost-usd {max_cost_usd}; raise it, "
            "or judge fewer instances with --limit / --subset"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ConfigError(f"--limit must be at least 1, got {args.limit}")
        judge = make_judge(args.provider, args.model, args.effort)
        stats_name = _stats_name()
        instances = select(load(args.data, _subsets(args.subset)), args.limit, args.seed)
        cache = PairwiseCache(args.cache_dir, offline=True if args.offline else None)
        # The estimate is printed before any paid call is made.
        est = estimate(instances, judge, cache, output_tokens=args.output_tokens)
        print(describe_estimate(est, judge.judge_id), file=sys.stderr)
        if args.dry_run:
            return 0
        problem = None if cache.offline else _budget_problem(est, args.max_cost_usd)
        if problem is not None:
            raise ConfigError(problem)
        results = run(
            instances,
            judge,
            cache,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost_usd,
        )
    except (BudgetExceeded, ConfigError) as exc:
        print(f"llmbar: {exc}", file=sys.stderr)
        return 2
    rows = rows_of(instances, results)
    print(markdown(rows, judge_id=judge.judge_id, stats_name=stats_name))
    new = [r for r in results.values() if not r.cached and r.error is None]
    print(
        f"\n{len(new)} new verdicts (${sum(r.cost_usd or 0.0 for r in new):.4f}), "
        f"{sum(1 for r in results.values() if r.cached)} from the cache",
        file=sys.stderr,
    )
    if args.out is not None:
        doc = report(
            rows,
            judge_id=judge.judge_id,
            stats_name=stats_name,
            limit=args.limit,
            seed=args.seed,
            timestamp=args.timestamp,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}", file=sys.stderr)
    return 1 if error_counts(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
