"""Tests for ``replay calibrate``: the sampled judge cache, the calibration engine, its
probes and estimate, the command, and calibration reports gating ``judge`` checks.

Offline and deterministic: judges are fakes deciding from the transcript, golden sets
are built in ``tmp_path`` from :class:`JudgeInput` objects, and clocks are fixed.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sklearn.metrics import cohen_kappa_score
from typer.testing import CliRunner

import agent_replay.calibrate.calibrate as calibrate_module
from agent_replay import RunResult, Trajectory, stats_baseline, stats_impl
from agent_replay.calibrate import commands_calibrate
from agent_replay.calibrate.calibrate import (
    CHARS_PER_TOKEN,
    MIN_CACHED_PREFIX_TOKENS,
    OUTPUT_TOKENS_PER_CALL,
    VERBOSITY_PADDING,
    calibrate,
    estimate,
    judge_model,
    padded,
    reversed_rubric,
)
from agent_replay.calibrate.golden import (
    OVERALL,
    GoldenItem,
    GoldenSet,
    GoldenSource,
    Label,
)
from agent_replay.cli import app
from agent_replay.errors import ConfigError, JudgeCacheMiss, JudgeError
from agent_replay.judge import (
    CacheEntry,
    CalibrationReport,
    FakeJudge,
    JudgeCache,
    JudgeInput,
    Rubric,
    gating,
    judge,
)
from agent_replay.judge.cache import cache_key
from agent_replay.judge.calibration import criterion_gates, report_blocker
from agent_replay.judge.providers import Decision
from agent_replay.judge.verdict import JudgeResponse, system_prompt
from agent_replay.pricing import PRICES
from agent_replay.store import Store

ROOT = Path(__file__).parents[1]
RETAIL = ROOT / "examples" / "retail_agent"
T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
OPUS = "anthropic:claude-opus-5-5:effort=low"

RUBRIC = Rubric.model_validate(
    {
        "name": "refunds",
        "version": 1,
        "criteria": [
            {"id": "resolved", "question": "Was the request resolved?"},
            {"id": "policy", "question": "Was the policy followed?"},
        ],
    }
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "REPLAY_STATS_IMPL",
        "REPLAY_JUDGE_OFFLINE",
        "REPLAY_JUDGE_CACHE",
        "REPLAY_JUDGE_PROVIDER",
        "REPLAY_JUDGE_MIN_KAPPA",
        "REPLAY_PRICES_FILE",
        "REPLAY_DB_URL",
    ):
        monkeypatch.delenv(name, raising=False)


# --- helpers -----------------------------------------------------------------------------


def transcript(task: str, output: Any = "Done.") -> JudgeInput:
    return JudgeInput(task=task, output=output)


def make_golden(
    directory: Path,
    tasks: Sequence[str],
    labels: Mapping[str, Sequence[int]],
    *,
    source: str = "human",
    labeller: str = "ann",
    agent_models: Sequence[str | None] | None = None,
    rubric: Rubric = RUBRIC,
) -> GoldenSet:
    """A golden set of one item per task, labelled ``labels[criterion][k]`` (1 = pass) on
    item ``k``; ``-1`` leaves that item unlabelled for that criterion."""
    golden = GoldenSet.load(directory, must_exist=False)
    models = agent_models or [None] * len(tasks)
    items = [
        GoldenItem.from_transcript(transcript(t), GoldenSource(agent_model=m))
        for t, m in zip(tasks, models, strict=True)
    ]
    golden, _ = golden.add_items(items)
    fingerprints = {c.id: c.fingerprint for c in rubric.criteria}
    rows = [
        Label(
            item=items[k].id,
            criterion=criterion,
            criterion_fingerprint=fingerprints.get(criterion),
            label="pass" if value else "fail",
            source=source,  # type: ignore[arg-type]
            labeller=labeller,
            labelled_at=T0,
        )
        for criterion, values in labels.items()
        for k, value in enumerate(values)
        if value >= 0
    ]
    return golden.append_labels(rows)


def table_judge(table: Mapping[str, Mapping[str, bool]], name: str = "table") -> FakeJudge:
    """A fake judge answering ``table[task]`` (every criterion passes for unknown tasks)."""

    def decide(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        row = table.get(judge_input.task, {})
        return {c: row.get(c, True) for c in rubric.ids}

    return FakeJudge(decide, name=name)


class Renamed:
    """A provider with another judge id (to look like a priced or same-model judge)."""

    def __init__(self, inner: FakeJudge, judge_id: str) -> None:
        self.inner = inner
        self._judge_id = judge_id

    @property
    def judge_id(self) -> str:
        return self._judge_id

    def evaluate(self, rubric: Rubric, judge_input: JudgeInput, /) -> JudgeResponse:
        return self.inner.evaluate(rubric, judge_input)


def cache_at(tmp_path: Path, *, offline: bool = False) -> JudgeCache:
    return JudgeCache(tmp_path / "cache", offline=offline, clock=lambda: T0)


TASKS = [f"task {k}" for k in range(10)]
REF_RESOLVED = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
JUDGE_RESOLVED = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
REF_POLICY = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
JUDGE_POLICY = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def designed_judge() -> FakeJudge:
    return table_judge(
        {
            t: {"resolved": bool(r), "policy": bool(p)}
            for t, r, p in zip(TASKS, JUDGE_RESOLVED, JUDGE_POLICY, strict=True)
        }
    )


# --- the sampled cache key -----------------------------------------------------------------


def test_sample_zero_keeps_the_committed_cache_keys_and_other_samples_differ():
    rubric = Rubric.load(RETAIL / "rubric.yaml")
    path = next(iter(sorted((RETAIL / "judge-cache").glob("*/*.json"))))
    entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
    as_input = cast(JudgeInput, SimpleNamespace(fingerprint=entry.input_fingerprint))

    assert entry.sample == 0  # written before samples existed, still valid
    assert cache_key(entry.judge_id, rubric, as_input) == entry.key
    assert cache_key(entry.judge_id, rubric, as_input, 0) == entry.key
    keys = {cache_key(entry.judge_id, rubric, as_input, s) for s in range(4)}
    assert len(keys) == 4
    found = JudgeCache(RETAIL / "judge-cache", offline=True).lookup(
        entry.judge_id, rubric, as_input
    )
    assert found is not None
    assert found.key == entry.key
    with pytest.raises(ValueError, match="sample"):
        cache_key(entry.judge_id, rubric, as_input, -1)


def test_samples_are_cached_apart(tmp_path: Path):
    cache = cache_at(tmp_path)
    provider = FakeJudge()
    t = transcript("task")

    _, hit0 = cache.verdict(provider, RUBRIC, t)
    _, hit1 = cache.verdict(provider, RUBRIC, t, sample=1)
    _, again = cache.verdict(provider, RUBRIC, t, sample=1)

    assert (hit0, hit1, again) == (False, False, True)
    assert provider.calls == 2
    entry = cache.lookup(provider.judge_id, RUBRIC, t, sample=1)
    assert entry is not None
    assert entry.sample == 1
    # an entry copied under another sample's key counts as corrupt
    key0, key2 = (cache_key(provider.judge_id, RUBRIC, t, s) for s in (0, 2))
    cache.path(key2).parent.mkdir(parents=True, exist_ok=True)
    cache.path(key2).write_text(
        cache.path(key0).read_text(encoding="utf-8").replace(key0, key2), encoding="utf-8"
    )
    assert cache.lookup(provider.judge_id, RUBRIC, t, sample=2) is None


# --- per-criterion metrics -----------------------------------------------------------------


def test_criterion_metrics_match_an_independent_computation(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED, "policy": REF_POLICY})

    report = calibrate(
        golden, RUBRIC, designed_judge(), cache=cache_at(tmp_path), repeats=1, probes=()
    )

    impl = stats_impl.load("baseline")
    c = report.criteria["resolved"]
    assert c.n == 10
    assert c.kappa == pytest.approx(impl.cohens_kappa(JUDGE_RESOLVED, REF_RESOLVED))
    low, high = impl.bootstrap_ci(impl.cohens_kappa, JUDGE_RESOLVED, REF_RESOLVED)
    assert (c.ci_low, c.ci_high) == (pytest.approx(max(-1, low)), pytest.approx(min(1, high)))
    assert c.agreement == pytest.approx(0.8)
    assert c.confusion is not None
    assert c.confusion.model_dump() == {
        "both_pass": 5,
        "judge_pass_reference_fail": 1,
        "judge_fail_reference_pass": 1,
        "both_fail": 3,
    }
    assert (c.majority_label, c.majority_agreement) == ("pass", pytest.approx(0.6))
    assert c.self_consistency is None  # one verdict per item
    assert report.criteria["policy"].agreement == 1.0
    assert report.criteria["policy"].majority_agreement == pytest.approx(0.5)
    assert report.criteria["policy"].majority_label == "pass"  # a tie goes to pass
    assert OVERALL not in report.criteria  # no whole-run labels
    assert report.stats_impl == "baseline"
    assert "raw agreement in the kappa column" in report.method
    assert "clamped to [-1, 1]" in report.method
    assert report.reference == "human"
    assert report.labeller == "ann"
    assert report.n_items == 10
    assert report.repeats == 1
    assert report.golden_fingerprint == golden.fingerprint
    assert (report.judge_calls, report.cached_calls, report.cost_usd) == (10, 0, 0.0)
    assert report.created_at == T0  # the newest verdict used, stamped by the cache's clock
    assert report_blocker(report) is not None  # baseline numbers never gate


def test_core_kappa_matches_scikit_learn_and_undefined_kappa_is_none(
    tmp_path: Path, core_shim: None
):
    # policy: every reference label passes and the judge passes all: kappa undefined
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED, "policy": [1] * 10})
    provider = table_judge(
        {t: {"resolved": bool(r)} for t, r in zip(TASKS, JUDGE_RESOLVED, strict=True)}
    )

    report = calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=(), clock=lambda: T0)

    assert report.stats_impl == "core"
    assert report.criteria["resolved"].kappa == pytest.approx(
        cohen_kappa_score(JUDGE_RESOLVED, REF_RESOLVED)
    )
    assert report.criteria["resolved"].kappa == pytest.approx(0.28 / 0.48)
    policy = report.criteria["policy"]
    assert (policy.kappa, policy.ci_low, policy.ci_high) == (None, None, None)
    assert policy.agreement == 1.0
    assert criterion_gates(report, 0.0)["policy"] is False
    gates = gating(report, RUBRIC, provider.judge_id, 0.0)
    assert gates["policy"].gating is False
    assert "undefined" in gates["policy"].reason
    assert report.created_at == T0  # no cache: the clock
    assert (
        "Cohen's kappa (unweighted), 95% bootstrap CI (2000 resamples, seed 0, clamped to "
        "[-1, 1])" in report.method
    )


def test_the_confusion_matrix_tells_false_passes_from_false_fails(tmp_path: Path):
    ref = [0, 0, 1, 1, 0]
    judged = [1, 1, 0, 1, 0]  # two false passes (tasks 0, 1), one false fail (task 2)
    golden = make_golden(tmp_path / "g", TASKS[:5], {"resolved": ref})
    provider = table_judge(
        {t: {"resolved": bool(j)} for t, j in zip(TASKS[:5], judged, strict=True)}
    )

    report = calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=())

    c = report.criteria["resolved"]
    assert c.confusion is not None
    assert c.confusion.model_dump() == {
        "both_pass": 1,
        "judge_pass_reference_fail": 2,
        "judge_fail_reference_pass": 1,
        "both_fail": 1,
    }
    assert (c.majority_label, c.majority_agreement) == ("fail", pytest.approx(0.6))


def test_self_consistency_counts_unanimous_items_and_pairwise_kappa(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED})
    seen: dict[str, int] = {}
    unstable = {"task 0", "task 7"}

    def decide(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        sample = seen.get(judge_input.task, 0)  # calls arrive in sample order
        seen[judge_input.task] = sample + 1
        base = JUDGE_RESOLVED[TASKS.index(judge_input.task)] == 1
        flip = judge_input.task in unstable and sample == 1
        return {"resolved": base != flip, "policy": True}

    report = calibrate(golden, RUBRIC, FakeJudge(decide), cache=cache_at(tmp_path), probes=())

    sc = report.criteria["resolved"].self_consistency
    assert sc is not None
    assert (sc.runs, sc.n, sc.unanimous) == (3, 10, pytest.approx(0.8))
    runs = [list(JUDGE_RESOLVED) for _ in range(3)]
    runs[1][0], runs[1][7] = 0, 1
    impl = stats_impl.load("baseline")
    pairs = [impl.cohens_kappa(a, b) for a, b in itertools.combinations(runs, 2)]
    assert sc.mean_pairwise_kappa == pytest.approx(sum(pairs) / len(pairs))
    # sample 0 is the verdict compared with the reference
    assert report.criteria["resolved"].agreement == pytest.approx(0.8)
    assert report.judge_calls == 30
    # a second run is served from the cache, sample by sample
    again = calibrate(golden, RUBRIC, FakeJudge(decide), cache=cache_at(tmp_path), probes=())
    assert (again.judge_calls, again.cached_calls) == (0, 30)
    assert again.model_dump_json() == report.model_dump_json().replace(
        '"judge_calls":30,"cached_calls":0', '"judge_calls":0,"cached_calls":30'
    )


def test_self_consistency_leaves_out_items_a_later_sample_failed_on(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED})
    seen: dict[str, int] = {}

    def later_sample_fails(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        sample = seen.get(judge_input.task, 0)
        seen[judge_input.task] = sample + 1
        if judge_input.task == "task 3" and sample == 2:
            raise JudgeError("overloaded")
        return {"resolved": JUDGE_RESOLVED[TASKS.index(judge_input.task)] == 1, "policy": True}

    report = calibrate(golden, RUBRIC, FakeJudge(later_sample_fails), cache=None, probes=())

    c = report.criteria["resolved"]
    assert c.n == 10  # sample 0 answered on every item
    assert c.self_consistency is not None
    assert (c.self_consistency.runs, c.self_consistency.n) == (3, 9)
    assert "1 judge call(s) failed" in report.method


# --- probes ------------------------------------------------------------------------------


def probe(report: CalibrationReport, name: str) -> Any:
    return next(p for p in report.probes if p.probe == name)


def test_the_position_probe_catches_an_order_sensitive_judge(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED, "policy": REF_POLICY})
    base = {
        t: {"resolved": bool(r), "policy": bool(p)}
        for t, r, p in zip(TASKS, JUDGE_RESOLVED, JUDGE_POLICY, strict=True)
    }

    def first_criterion_passes(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        first = rubric.criteria[0].id
        return {c: c == first or base[judge_input.task][c] for c in rubric.ids}

    report = calibrate(
        golden, RUBRIC, FakeJudge(first_criterion_passes), cache=None, repeats=1,
        probes=("position",),
    )  # fmt: skip

    flips = probe(report, "position").flips
    # original order: resolved always passes; reversed: policy always passes instead
    assert flips["resolved"].model_dump(include={"n", "flips", "to_pass", "to_fail"}) == {
        "n": 10, "flips": 4, "to_pass": 0, "to_fail": 4,
    }  # fmt: skip
    assert (flips["policy"].to_pass, flips["policy"].to_fail) == (5, 0)
    assert flips["resolved"].rate == pytest.approx(0.4)
    low, high = stats_baseline.wilson_interval(4, 10)
    assert (flips["resolved"].ci_low, flips["resolved"].ci_high) == (low, high)
    assert OVERALL in flips
    assert "Wald" in report.method

    steady = calibrate(
        golden, RUBRIC, designed_judge(), cache=None, repeats=1, probes=("position",)
    )
    assert all(f.flips == 0 for f in probe(steady, "position").flips.values())
    assert reversed_rubric(RUBRIC).ids == ("policy", "resolved")
    assert reversed_rubric(RUBRIC).fingerprint != RUBRIC.fingerprint


def test_position_is_skipped_for_a_one_criterion_rubric(tmp_path: Path):
    single = Rubric.model_validate(
        {"name": "one", "version": 1, "criteria": [{"id": "resolved", "question": "Q?"}]}
    )
    golden = make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]}, rubric=single)

    report = calibrate(golden, single, FakeJudge(), cache=None, repeats=1, probes=("position",))

    assert probe(report, "position").skipped == (
        "the rubric has one criterion, so its order cannot change"
    )


def test_the_verbosity_probe_catches_a_length_lenient_judge(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED})

    def lenient(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        base = JUDGE_RESOLVED[TASKS.index(judge_input.task)] == 1
        long = isinstance(judge_input.output, str) and len(judge_input.output) > 100
        return {"resolved": base or long, "policy": True}

    report = calibrate(
        golden, RUBRIC, FakeJudge(lenient), cache=None, repeats=1, probes=("verbosity",)
    )

    flips = probe(report, "verbosity").flips
    assert (flips["resolved"].to_pass, flips["resolved"].to_fail) == (4, 0)
    assert flips["policy"].flips == 0
    padded_input = padded(transcript("t"))
    assert padded_input is not None
    assert padded_input.output == "Done." + VERBOSITY_PADDING
    assert padded(transcript("t", output={"status": "ok"})) is None

    fair = calibrate(golden, RUBRIC, designed_judge(), cache=None, repeats=1, probes=("verbosity",))
    assert probe(fair, "verbosity").flips["resolved"].flips == 0


def test_the_verbosity_probe_is_skipped_without_a_text_output(tmp_path: Path):
    golden = GoldenSet.load(tmp_path / "g", must_exist=False)
    items = [
        GoldenItem.from_transcript(transcript(t, output={"status": "ok"}), GoldenSource())
        for t in TASKS[:2]
    ]
    golden, _ = golden.add_items(items)
    fp = RUBRIC.criteria[0].fingerprint
    golden = golden.append_labels(
        [
            Label(
                item=i.id, criterion="resolved", criterion_fingerprint=fp, label="pass",
                source="human", labeller="ann", labelled_at=T0,
            )
            for i in items
        ]
    )  # fmt: skip
    provider = FakeJudge()

    report = calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=("verbosity",))

    assert probe(report, "verbosity").skipped == "no labelled item has a text final output to pad"
    assert provider.calls == 2  # the main verdicts only


def test_an_unknown_probe_is_a_config_error(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    provider = FakeJudge()

    with pytest.raises(
        ConfigError, match=r"unknown probe 'length'; use position, verbosity, self-preference"
    ):
        calibrate(golden, RUBRIC, provider, cache=None, probes=("position", "length"))
    assert provider.calls == 0


def test_the_self_preference_probe_groups_runs_by_the_judges_model(tmp_path: Path):
    models = ["claude-opus-5-5"] * 5 + ["other-model"] * 4 + [None]
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED}, agent_models=models)
    # lenient on its own model's runs (tasks 0-4), strict elsewhere
    own = {t: {"resolved": True} for t in TASKS[:5]}
    others = {t: {"resolved": bool(r)} for t, r in zip(TASKS[5:], REF_RESOLVED[5:], strict=True)}
    provider = Renamed(table_judge(own | others), OPUS)

    report = calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=("self-preference",))

    result = probe(report, "self_preference")
    assert result.skipped is None
    assert "claude-opus-5-5" in result.description
    same, other = result.groups["resolved"]["same_model"], result.groups["resolved"]["other_model"]
    assert (same.n, same.judge_pass_rate, same.reference_pass_rate) == (5, 1.0, 1.0)
    assert (same.reference_fail_n, same.false_pass_rate) == (0, None)
    assert (other.n, other.reference_fail_n, other.false_pass_rate) == (4, 3, 0.0)
    assert other.reference_pass_rate == pytest.approx(0.25)

    unknown = make_golden(tmp_path / "u", TASKS[:2], {"resolved": [1, 0]})
    skipped = calibrate(
        unknown, RUBRIC, provider, cache=None, repeats=1, probes=("self_preference",)
    )
    assert "records the model" in (probe(skipped, "self_preference").skipped or "")
    only_own = make_golden(
        tmp_path / "o", TASKS[:2], {"resolved": [1, 0]}, agent_models=["claude-opus-5-5"] * 2
    )
    skipped = calibrate(
        only_own, RUBRIC, provider, cache=None, repeats=1, probes=("self_preference",)
    )
    assert "every labelled item" in (probe(skipped, "self_preference").skipped or "")


def test_judge_model_parses_judge_ids():
    assert judge_model(OPUS) == "claude-opus-5-5"
    assert judge_model("ollama:qwen3:8b") == "qwen3:8b"
    assert judge_model("fake:default") == "default"
    assert judge_model("custom") == "custom"


# --- references and labels -----------------------------------------------------------------


def test_an_objective_reference_compares_whole_runs_and_never_gates(tmp_path: Path):
    overall = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    golden = make_golden(
        tmp_path / "g", TASKS, {OVERALL: overall}, source="objective", labeller="checks"
    )

    report = calibrate(
        golden, RUBRIC, designed_judge(), cache=None, repeats=1, probes=(),
        reference="objective",
    )  # fmt: skip

    assert list(report.criteria) == [OVERALL]
    judged = [int(r and p) for r, p in zip(JUDGE_RESOLVED, JUDGE_POLICY, strict=True)]
    agreement = sum(j == o for j, o in zip(judged, overall, strict=True)) / 10
    assert report.criteria[OVERALL].agreement == pytest.approx(agreement)
    assert report.reference == "objective"
    assert report.labeller == "checks"
    assert "objective" in (report_blocker(report) or "")
    with pytest.raises(ConfigError, match="replay label"):
        calibrate(golden, RUBRIC, FakeJudge(), cache=None, reference="human")
    empty = make_golden(tmp_path / "e", TASKS[:1], {})
    with pytest.raises(ConfigError, match="replay golden add"):
        calibrate(empty, RUBRIC, FakeJudge(), cache=None, reference="objective")


def test_several_labellers_need_a_choice(tmp_path: Path):
    make_golden(tmp_path / "g", TASKS[:4], {"resolved": [1, 0, 1, 0]}, labeller="ann")
    golden = make_golden(tmp_path / "g", TASKS[:4], {"resolved": [1, 1, 1, 0]}, labeller="bo")

    with pytest.raises(ConfigError, match=r"ann, bo.*--labeller"):
        calibrate(golden, RUBRIC, FakeJudge(), cache=None)
    report = calibrate(golden, RUBRIC, FakeJudge(), cache=None, labeller="bo", probes=())
    assert report.labeller == "bo"
    assert report.criteria["resolved"].confusion is not None
    assert report.criteria["resolved"].confusion.both_pass == 3


def test_stale_labels_are_left_out(tmp_path: Path):
    old = Rubric.model_validate(
        {
            **RUBRIC.model_dump(),
            "criteria": [
                {"id": "resolved", "question": "Was it resolved, in the old wording?"},
                {"id": "policy", "question": "Was the policy followed?"},
            ],
        }
    )
    make_golden(tmp_path / "g", TASKS[:4], {"resolved": [1, 0, 1, 0]}, rubric=old)
    golden = make_golden(tmp_path / "g", TASKS[4:6], {"resolved": [1, 0], "policy": [1, 1]})

    report = calibrate(golden, RUBRIC, FakeJudge(), cache=None, repeats=1, probes=())

    assert report.criteria["resolved"].n == 2
    assert report.n_items == 2
    assert golden.stale_labels(RUBRIC) == 4


# --- errors ------------------------------------------------------------------------------


def test_an_offline_cache_miss_fails_before_any_call(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:3], {"resolved": [1, 0, 1]})
    provider = FakeJudge()

    with pytest.raises(JudgeCacheMiss, match=r"9 of the 9 verdicts.*calibrating offline"):
        calibrate(golden, RUBRIC, provider, cache=cache_at(tmp_path, offline=True), probes=())
    assert provider.calls == 0

    calibrate(golden, RUBRIC, provider, cache=cache_at(tmp_path), probes=())
    offline = calibrate(golden, RUBRIC, provider, cache=cache_at(tmp_path, offline=True), probes=())
    assert (offline.judge_calls, offline.cached_calls) == (0, 9)


def test_a_judge_error_drops_the_item_not_the_report(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:4], {"resolved": [1, 0, 1, 0]})

    def flaky(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        if judge_input.task == "task 1":
            raise JudgeError("the judge refused")
        return dict.fromkeys(rubric.ids, True)

    report = calibrate(golden, RUBRIC, FakeJudge(flaky), cache=None, repeats=2, probes=())

    assert report.criteria["resolved"].n == 3
    assert "2 judge call(s) failed" in report.method


def test_a_judge_that_never_answers_stops_the_calibration_early(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED})

    def broken(_rubric: Rubric, _judge_input: JudgeInput) -> dict[str, Decision]:
        raise JudgeError("no credentials")

    provider = FakeJudge(broken)
    with pytest.raises(JudgeError, match=r"first 3 judge calls all failed.*no credentials"):
        calibrate(golden, RUBRIC, provider, cache=None)
    assert provider.calls == 3


def test_failures_after_a_first_answer_never_stop_the_calibration(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED})
    failing = {"task 1", "task 2", "task 3", "task 6"}  # three in a row, then one more

    def sometimes(rubric: Rubric, judge_input: JudgeInput) -> dict[str, Decision]:
        if judge_input.task in failing:
            raise JudgeError("overloaded")
        return dict.fromkeys(rubric.ids, True)

    provider = FakeJudge(sometimes)
    report = calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=())

    assert provider.calls == 10
    assert report.criteria["resolved"].n == 6
    assert "4 judge call(s) failed" in report.method


def test_an_unwritten_core_stats_module_is_a_config_error(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    provider = FakeJudge()
    with pytest.raises(ConfigError, match="REPLAY_STATS_IMPL=baseline"):
        calibrate(golden, RUBRIC, provider, cache=None, stats="core")
    assert provider.calls == 0


# --- the estimate --------------------------------------------------------------------------


def test_the_estimate_counts_calls_and_cached_verdicts(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:4], {"resolved": [1, 0, 1, 0]})
    provider = FakeJudge()
    cache = cache_at(tmp_path)

    first = estimate(golden, RUBRIC, provider, cache, repeats=2)
    assert (first.calls, first.cached, first.est_cost_usd) == (16, 0, 0.0)  # 4x2 + 4 + 4
    calibrate(golden, RUBRIC, provider, cache=cache, repeats=1, probes=())
    later = estimate(golden, RUBRIC, provider, cache, repeats=2, probes=())
    assert (later.calls, later.cached) == (4, 4)
    assert provider.calls == 4  # estimating never calls the judge
    assert estimate(golden, RUBRIC, provider, None, repeats=1, probes=()).calls == 4


def test_the_estimate_prices_the_judge_model(tmp_path: Path):
    golden = make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    priced = Renamed(FakeJudge(), OPUS)

    est = estimate(golden, RUBRIC, priced, None, repeats=2, probes=())

    price = PRICES["claude-opus-5-5"]
    system = len(system_prompt(RUBRIC)) / CHARS_PER_TOKEN
    inputs = [len(transcript(t).render()) / CHARS_PER_TOKEN for t in TASKS[:2] for _ in range(2)]
    expected = (
        system * price.cache_write
        + 3 * system * price.cache_read
        + sum(inputs) * price.input
        + 4 * OUTPUT_TOKENS_PER_CALL * price.output
    ) / 1_000_000
    assert est.est_cost_usd == pytest.approx(expected, abs=1e-4)
    assert est.notes == ()  # the real system prompt is long enough to cache
    unpriced = estimate(golden, RUBRIC, Renamed(FakeJudge(), "custom:mystery"), None)
    assert unpriced.est_cost_usd is None


def test_a_system_prompt_too_short_to_cache_is_priced_as_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_judge: Callable[[Any], None]
):
    golden = make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    monkeypatch.setattr(calibrate_module, "system_prompt", lambda _rubric: "x" * 230)
    priced = Renamed(FakeJudge(), OPUS)

    est = estimate(golden, RUBRIC, priced, None, repeats=2, probes=())

    price = PRICES["claude-opus-5-5"]
    inputs = [len(transcript(t).render()) / CHARS_PER_TOKEN for t in TASKS[:2] for _ in range(2)]
    expected = (
        4 * 100 * price.input  # 230 chars: 100 tokens, under the cacheable minimum, every call
        + sum(inputs) * price.input
        + 4 * OUTPUT_TOKENS_PER_CALL * price.output
    ) / 1_000_000
    assert est.est_cost_usd == pytest.approx(expected, abs=1e-4)
    assert len(est.notes) == 1
    assert f"under the {MIN_CACHED_PREFIX_TOKENS}-token minimum" in est.notes[0]
    assert "uncached input on every call" in est.notes[0]

    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")
    cli_judge(priced)
    result = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--dry-run",
        "--cache-dir", str(tmp_path / "cache"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "note: the system prompt (about 100 tokens) is under the 512-token" in result.output


# --- the command -------------------------------------------------------------------------


def invoke(*args: str, env: dict[str, str] | None = None):
    return runner.invoke(app, list(args), env={"COLUMNS": "1000", **(env or {})})


@pytest.fixture
def cli_judge(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    """Make ``replay calibrate`` use the given provider whatever --provider says."""

    def use(provider: Any) -> None:
        monkeypatch.setattr(commands_calibrate, "make_provider", lambda *a, **k: provider)

    return use


def test_a_dry_run_prints_the_estimate_and_judges_nothing(tmp_path: Path):
    make_golden(tmp_path / "g", TASKS[:3], {"resolved": [1, 0, 1]})
    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")  # JSON is YAML

    result = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--provider", "fake",
        "--cache-dir", str(tmp_path / "cache"), "--dry-run", "-o", str(tmp_path / "r.json"),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "estimate: 15 judge call(s) to make, 0 cached (of 15 verdicts)" in result.output
    assert "estimated cost $0.0000" in result.output
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "r.json").exists()


def test_the_cost_cap_aborts_before_judging(tmp_path: Path, cli_judge: Callable[[Any], None]):
    make_golden(tmp_path / "g", TASKS[:3], {"resolved": [1, 0, 1]})
    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")
    inner = FakeJudge()
    cli_judge(Renamed(inner, OPUS))

    result = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric),
        "--cache-dir", str(tmp_path / "cache"), "--max-cost-usd", "0.001", "--no-store",
    )  # fmt: skip

    assert result.exit_code == 2, result.output
    assert "above --max-cost-usd 0.001" in result.output
    assert inner.calls == 0

    both = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--probe", "position",
        "--no-probes",
    )  # fmt: skip
    assert both.exit_code == 2
    assert "contradict" in both.output


def test_the_cost_cap_needs_a_priced_model(tmp_path: Path, cli_judge: Callable[[Any], None]):
    make_golden(tmp_path / "g", TASKS[:3], {"resolved": [1, 0, 1]})
    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")
    inner = FakeJudge()
    cli_judge(Renamed(inner, "custom:mystery"))

    result = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric),
        "--cache-dir", str(tmp_path / "cache"), "--max-cost-usd", "1", "--no-store",
    )  # fmt: skip

    assert result.exit_code == 2, result.output
    assert "cannot estimate the cost of judge custom:mystery" in result.output
    assert "REPLAY_PRICES_FILE or drop --max-cost-usd" in result.output
    assert inner.calls == 0


def test_the_report_names_the_golden_set_the_same_from_any_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    monkeypatch.chdir(tmp_path)

    def report(directory: Path) -> CalibrationReport:
        golden = GoldenSet.load(directory)
        return calibrate(
            golden, RUBRIC, FakeJudge(), cache=None, repeats=1, probes=(), clock=lambda: T0
        )

    relative, absolute = report(Path("g")), report(tmp_path / "g")
    assert relative.golden_set == "g"
    assert absolute.model_dump_json() == relative.model_dump_json()
    monkeypatch.chdir(tmp_path / "g")
    assert report(tmp_path / "g").golden_set == "."
    outside = tmp_path / "g" / "inner"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert report(tmp_path / "g").golden_set == (tmp_path / "g").as_posix()


def test_the_command_writes_and_stores_the_report(tmp_path: Path, cli_judge: Callable[[Any], None]):
    make_golden(tmp_path / "g", TASKS, {"resolved": REF_RESOLVED, "policy": REF_POLICY})
    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")
    cli_judge(designed_judge())
    db = f"sqlite:///{tmp_path / 'replay.db'}"
    out = tmp_path / "out" / "report.json"

    result = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--repeats", "2",
        "--cache-dir", str(tmp_path / "cache"), "-o", str(out), "--db", db,
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "estimate: 40 judge call(s) to make" in result.output
    assert "resolved" in result.output
    assert "80%" in result.output
    assert "probe position: flips" in result.output
    assert "probe self-preference: skipped" in result.output
    assert "REPLAY_STATS_IMPL=baseline" in result.output  # why nothing gates
    report = CalibrationReport.load(out)
    assert report.judge_calls == 40
    assert report.repeats == 2
    with Store.open(db) as store:
        assert [c.n_items for c in store.list_calibrations()] == [10]

    # the second run is served from the cache and writes the same bytes
    first = out.read_text(encoding="utf-8")
    again = invoke(
        "calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--repeats", "2",
        "--cache-dir", str(tmp_path / "cache"), "-o", str(out), "--no-store", "--offline",
    )  # fmt: skip
    assert again.exit_code == 0, again.output
    assert "0 judge call(s) to make, 40 cached" in again.output
    assert (
        json.loads(out.read_text(encoding="utf-8"))["created_at"] == json.loads(first)["created_at"]
    )


def test_the_command_exits_2_without_labels_or_cached_verdicts(tmp_path: Path):
    make_golden(tmp_path / "g", TASKS[:2], {"resolved": [1, 0]})
    rubric = tmp_path / "rubric.json"
    rubric.write_text(RUBRIC.model_dump_json(), encoding="utf-8")
    base = ["calibrate", str(tmp_path / "g"), "--rubric", str(rubric), "--provider", "fake"]

    objective = invoke(*base, "--reference", "objective", "--no-store")
    assert objective.exit_code == 2
    assert "replay golden add" in objective.output
    offline = invoke(*base, "--cache-dir", str(tmp_path / "c"), "--offline", "--no-store")
    assert offline.exit_code == 2
    assert "calibrating offline" in offline.output


# --- calibration reports gating judge checks ------------------------------------------------


GATE_TASKS = [f"case {k}" for k in range(20)]
GATE_REF = [1, 0] * 10
# resolved: the judge always agrees; policy: it agrees on half the items, by chance
GATE_TABLE = {
    t: {"resolved": bool(r), "policy": bool((k // 2) % 2)}
    for k, (t, r) in enumerate(zip(GATE_TASKS, GATE_REF, strict=True))
} | {
    "fails resolved": {"resolved": False, "policy": True},
    "fails policy": {"resolved": True, "policy": False},
}


def _check_run(report: CalibrationReport, task: str) -> Any:
    provider = table_judge(GATE_TABLE, name="gate")
    check = judge(RUBRIC, provider=provider, cache=None, calibration=report, min_kappa=0.6)
    run = RunResult(mode="full", task=task, output="Done.", trajectory=Trajectory(steps=()))
    return check(run)


def _gate_report(tmp_path: Path, **kwargs: Any) -> CalibrationReport:
    golden = make_golden(tmp_path / "g", GATE_TASKS, {"resolved": GATE_REF, "policy": GATE_REF})
    provider = table_judge(GATE_TABLE, name="gate")
    return calibrate(golden, RUBRIC, provider, cache=None, repeats=1, probes=(), **kwargs)


def test_a_core_human_report_gates_the_calibrated_criterion_only(tmp_path: Path, core_shim: None):
    report = _gate_report(tmp_path)

    assert report.stats_impl == "core"
    assert report_blocker(report) is None
    assert report.criteria["resolved"].ci_low == 1.0
    policy = report.criteria["policy"]
    assert policy.kappa is not None
    assert policy.kappa < 0.6
    assert criterion_gates(report, 0.6) == {"resolved": True, "policy": False}

    blocked = _check_run(report, "fails resolved")
    assert (blocked.passed, blocked.advisory) == (False, False)
    advisory = _check_run(report, "fails policy")
    assert (advisory.passed, advisory.advisory) == (False, True)
    assert "gating: resolved; advisory: policy" in advisory.message


def test_baseline_and_objective_reports_never_gate(tmp_path: Path, core_shim: None):
    baseline = _gate_report(tmp_path, stats="baseline")
    assert baseline.stats_impl == "baseline"
    assert baseline.criteria["resolved"].ci_low == 1.0  # the same numbers ...
    assert criterion_gates(baseline, 0.6) == {"resolved": False, "policy": False}
    result = _check_run(baseline, "fails resolved")
    assert (result.passed, result.advisory) == (False, True)  # ... yet only advisory

    golden = make_golden(
        tmp_path / "o", GATE_TASKS, {OVERALL: GATE_REF}, source="objective", labeller="checks"
    )
    objective = calibrate(
        golden, RUBRIC, table_judge(GATE_TABLE, name="gate"), cache=None, repeats=1,
        probes=(), reference="objective",
    )  # fmt: skip
    assert objective.stats_impl == "core"
    result = _check_run(objective, "fails resolved")
    assert (result.passed, result.advisory) == (False, True)
