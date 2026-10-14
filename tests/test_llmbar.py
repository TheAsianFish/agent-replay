"""Tests for the LLMBar harness (``eval/llmbar.py``) and the vendored data.

The metric tests use small synthetic datasets written here, not LLMBar data: every text
says "synthetic". Only the last tests read the vendored files, to check they parse and
match the counts and hashes the README records.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_replay import stats, stats_baseline
from agent_replay.errors import JudgeError
from agent_replay.judge.pairwise import (
    PAIRWISE_INSTRUCTIONS,
    FakePairwiseJudge,
    PairwiseCache,
    PairwiseDecision,
    PairwiseVerdict,
    render_pair,
)
from eval import llmbar, llmbar_fetch
from eval.llmbar import (
    SUBSETS,
    BudgetExceeded,
    Instance,
    Row,
    chosen_output,
    estimate,
    groups,
    load,
    markdown,
    rows_of,
    run,
    select,
    summarise,
)

# --- synthetic data ----------

Spec = Sequence[int]  # the labels of a synthetic subset, one per instance


def write_data(root: Path, labels: dict[str, Spec]) -> Path:
    """A data directory in LLMBar's layout; the correct output always says "right"."""
    for subset in SUBSETS:
        records = []
        for i, label in enumerate(labels.get(subset, ())):
            right = f"synthetic right answer {subset} {i}"
            wrong = f"synthetic wrong answer {subset} {i}, longer"
            records.append(
                {
                    "input": f"synthetic instruction {subset} {i}",
                    "output_1": right if label == 1 else wrong,
                    "output_2": wrong if label == 1 else right,
                    "label": label,
                }
            )
        path = root / subset / "dataset.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")
    return root


BALANCED: dict[str, Spec] = {
    "Natural": (1, 2, 1, 2),
    "Adversarial/Neighbor": (2, 1),
    "Adversarial/GPTInst": (1, 2),
    "Adversarial/GPTOut": (2, 1),
    "Adversarial/Manual": (1, 2),
}
N_BALANCED = 12


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return write_data(tmp_path / "data", BALANCED)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("REPLAY_STATS_IMPL", "REPLAY_JUDGE_OFFLINE", "REPLAY_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def oracle(_i: str, a: str, _b: str) -> PairwiseDecision:
    return "a" if "right" in a else "b"


def first(_i: str, _a: str, _b: str) -> PairwiseDecision:
    return "a"


def second(_i: str, _a: str, _b: str) -> PairwiseDecision:
    return "b"


def judged(data: Path, tmp_path: Path, judge: FakePairwiseJudge, **kwargs: object) -> list[Row]:
    instances = load(data)
    cache = PairwiseCache(tmp_path / "cache", offline=False)
    return rows_of(instances, run(instances, judge, cache, **kwargs))  # type: ignore[arg-type]


def overall(rows: Sequence[Row]):
    return groups(rows)[-1]


# --- mapping ----------


def test_the_swapped_order_maps_positions_back_to_outputs():
    inst = Instance("Natural", 0, "synthetic", "one", "two", 2)

    assert inst.pair("original") == ("one", "two")
    assert inst.pair("swapped") == ("two", "one")
    assert chosen_output("original", "a") == 1
    assert chosen_output("original", "b") == 2
    assert chosen_output("swapped", "a") == 2
    assert chosen_output("swapped", "b") == 1


# --- metrics ----------


def test_summarise_hand_checked():
    rows = [
        Row("Natural", 0, 1, 1, 1),  # right both times
        Row("Natural", 1, 2, 2, 2),  # right both times
        Row("Natural", 2, 1, 1, 2),  # flip, first position both times; right then wrong
        Row("Natural", 3, 2, 1, 2),  # flip, first position both times; wrong then right
        Row("Natural", 4, 1, 2, 1),  # flip, second position both times; wrong then right
        Row("Natural", 5, 2, 1, 1),  # consistently wrong
        Row("Natural", 6, 1, None, 1, error_original="synthetic error"),
    ]

    m = summarise("Natural", rows)

    assert (m.instances, m.errors) == (7, 1)
    assert (m.accuracy_original.successes, m.accuracy_original.trials) == (3, 6)
    assert (m.accuracy_swapped.successes, m.accuracy_swapped.trials) == (5, 7)
    assert (m.accuracy_average.successes, m.accuracy_average.trials) == (8, 13)
    assert (m.consistent_correct.successes, m.consistent_correct.trials) == (2, 6)
    assert (m.position_bias.successes, m.position_bias.trials) == (3, 6)
    assert (m.flips_to_first.successes, m.flips_to_first.trials) == (2, 3)
    # The interval is the selected implementation's (baseline: Wald), not computed here.
    assert m.accuracy_original.interval == stats_baseline.wilson_interval(3, 6)


def test_an_always_first_judge_is_at_chance_and_fully_position_biased(data, tmp_path):
    m = overall(judged(data, tmp_path, FakePairwiseJudge(first)))

    assert m.instances == N_BALANCED
    assert m.accuracy_original.rate == 0.5  # label 1 in half the instances
    assert m.accuracy_swapped.rate == 0.5
    assert m.accuracy_average.rate == 0.5
    assert m.consistent_correct.rate == 0.0
    assert m.position_bias.rate == 1.0
    assert m.flips_to_first.rate == 1.0
    assert m.errors == 0


def test_an_always_second_judge_flips_towards_the_second_position(data, tmp_path):
    m = overall(judged(data, tmp_path, FakePairwiseJudge(second)))

    assert (m.accuracy_average.rate, m.position_bias.rate) == (0.5, 1.0)
    assert m.flips_to_first.rate == 0.0


def test_an_oracle_judge_is_always_right_and_never_flips(data, tmp_path):
    rows = judged(data, tmp_path, FakePairwiseJudge(oracle))

    for m in groups(rows):
        assert m.accuracy_original.rate == m.accuracy_swapped.rate == 1.0
        assert m.consistent_correct.rate == 1.0
        assert m.position_bias.rate == 0.0
        assert m.flips_to_first.trials == 0
    names = [m.group for m in groups(rows)]
    assert names == [*SUBSETS, "Adversarial (pooled)", "Overall"]
    assert [m.instances for m in groups(rows)][-2:] == [8, 12]


def test_a_length_judge_is_right_only_where_the_right_output_is_longer(tmp_path):
    # In the synthetic data the wrong output is always the longer one.
    data = write_data(tmp_path / "data", {"Natural": (1, 2, 2)})
    instances = load(data, ["Natural"])
    judge = FakePairwiseJudge(llmbar.FAKES["fake-shorter"], name="shorter")
    rows = rows_of(instances, run(instances, judge, PairwiseCache(tmp_path / "c", offline=False)))

    m = summarise("Natural", rows)
    assert (m.accuracy_average.rate, m.position_bias.rate) == (1.0, 0.0)
    longer = FakePairwiseJudge(llmbar.FAKES["fake-longer"], name="longer")
    rows = rows_of(instances, run(instances, longer, PairwiseCache(tmp_path / "d", offline=False)))
    assert summarise("Natural", rows).accuracy_average.rate == 0.0


def test_judge_errors_are_counted_not_fatal(data, tmp_path):
    def flaky(instruction: str, a: str, b: str) -> PairwiseDecision:
        if instruction == "synthetic instruction Natural 1":
            raise JudgeError("synthetic judge failure")
        return oracle(instruction, a, b)

    rows = judged(data, tmp_path, FakePairwiseJudge(flaky, name="flaky"))

    natural = summarise("Natural", [r for r in rows if r.subset == "Natural"])
    assert (natural.instances, natural.errors) == (4, 2)
    assert natural.accuracy_average.trials == 6
    assert natural.position_bias.trials == 3
    assert llmbar.error_counts(rows) == {"synthetic judge failure": 2}
    text = markdown(rows, judge_id="fake:flaky", stats_name="baseline")
    assert "- 2 x synthetic judge failure" in text


def _broken(instruction: str, _a: str, _b: str) -> PairwiseDecision:
    raise JudgeError(f"synthetic: no credentials for {instruction}")


def broken_judge() -> FakePairwiseJudge:
    """A judge that never answers, as one without credentials or with a wrong model name."""
    return FakePairwiseJudge(_broken, name="broken")


def test_a_judge_that_never_answers_stops_the_run_after_the_first_calls(data, tmp_path):
    instances = load(data)
    judge = broken_judge()

    with pytest.raises(JudgeError, match="first 3 judge calls all failed"):
        run(instances, judge, PairwiseCache(tmp_path / "cache", offline=False))

    assert judge.calls == llmbar.GIVE_UP_AFTER  # not one per trial


def test_a_judge_that_never_answers_exits_2_without_writing_results(data, tmp_path, capsys):
    out_path = tmp_path / "result.json"
    code, _, err = main_out(
        capsys,
        *("--data", str(data), "--cache-dir", str(tmp_path / "cache"), "--out", str(out_path)),
        *("--provider", "tests.test_llmbar:broken_judge"),
    )

    assert code == 2
    assert "judge calls all failed, so the run stopped" in err
    assert "synthetic: no credentials" in err
    assert not out_path.exists()


def test_concurrency_does_not_change_the_result(data, tmp_path):
    def mixed(instruction: str, a: str, b: str) -> PairwiseDecision:
        return first(instruction, a, b) if "Natural" in instruction else oracle(instruction, a, b)

    one = judged(data, tmp_path / "one", FakePairwiseJudge(mixed), concurrency=1)
    many = judged(data, tmp_path / "many", FakePairwiseJudge(mixed), concurrency=5)

    assert one == many


# --- selection ----------


def test_limit_takes_a_seeded_sample_per_subset_in_file_order(tmp_path):
    data = write_data(tmp_path / "data", {"Natural": (1,) * 10, "Adversarial/Manual": (2, 1)})
    instances = load(data)

    picked = select(instances, 3, seed=7)
    again = select(instances, 3, seed=7)
    other = select(instances, 3, seed=8)

    assert picked == again
    natural = [i.index for i in picked if i.subset == "Natural"]
    assert len(natural) == 3
    assert natural == sorted(natural)
    assert [i.index for i in picked if i.subset == "Adversarial/Manual"] == [0, 1]
    assert select(instances, None) == instances
    assert natural == [0, 8, 9]  # pinned: the sample must not drift between runs
    assert [i.index for i in other if i.subset == "Natural"] == [1, 2, 8]


# --- cache, dry run, budget ----------


def test_a_second_run_is_served_from_the_cache(data, tmp_path):
    instances = load(data)
    cache = PairwiseCache(tmp_path / "cache", offline=False)
    judge = FakePairwiseJudge(oracle)

    run(instances, judge, cache)
    results = run(instances, judge, cache)

    assert judge.calls == 2 * N_BALANCED
    assert all(r.cached for r in results.values())


def main_out(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = llmbar.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_dry_run_counts_calls_left_after_cache_hits(data, tmp_path, capsys):
    cache = str(tmp_path / "cache")
    common = ("--data", str(data), "--cache-dir", cache, "--provider", "fake-first")

    code, _, err = main_out(capsys, *common, "--dry-run")
    assert code == 0
    assert f"{2 * N_BALANCED} judge calls needed (0 cached)" in err

    assert main_out(capsys, *common, "--subset", "Natural")[0] == 0
    code, out, err = main_out(capsys, *common, "--dry-run")
    assert code == 0
    assert out == ""  # a dry run judges nothing
    assert f"{2 * (N_BALANCED - 4)} judge calls needed (8 cached)" in err
    assert "estimated cost $0.00" in err


def test_the_estimate_prices_input_by_characters_and_output_per_call(tmp_path):
    data = write_data(tmp_path / "data", {"Natural": (1,)})
    instances = load(data, ["Natural"])
    judge = llmbar.make_judge("anthropic", None, None)
    cache = PairwiseCache(tmp_path / "cache", offline=False)

    est = estimate(instances, judge, cache, output_tokens=100)

    schema = len(json.dumps(llmbar.PairwiseOutput.model_json_schema()))
    inst = instances[0]
    expected_in = sum(
        math.ceil(
            (len(PAIRWISE_INSTRUCTIONS) + schema + len(render_pair(inst.instruction, a, b))) / 4
        )
        for a, b in (inst.pair("original"), inst.pair("swapped"))
    )
    assert (est.calls, est.cached, est.input_tokens, est.output_tokens) == (2, 0, expected_in, 200)
    assert est.cost_usd == round((expected_in * 4.0 + 200 * 20.0) / 1_000_000, 4)


def test_max_cost_aborts_before_any_call(data, tmp_path, capsys):
    code, out, err = main_out(
        capsys,
        *("--data", str(data), "--cache-dir", str(tmp_path / "c")),
        *("--provider", "anthropic", "--max-cost-usd", "0.0001"),
    )

    assert code == 2
    assert out == ""
    assert "judge calls needed" in err  # the estimate is printed first
    assert "over --max-cost-usd 0.0001" in err
    assert not (tmp_path / "c").exists()


class PaidJudge:
    """A synthetic judge that charges one cent a call."""

    judge_id = "synthetic:paid"

    def __init__(self) -> None:
        self.calls = 0

    def compare(self, instruction: str, output_a: str, output_b: str, /) -> PairwiseVerdict:
        self.calls += 1
        return PairwiseVerdict(reasoning="synthetic", winner="a", cost_usd=0.01)


def test_spending_past_the_budget_stops_the_run_and_keeps_what_was_judged(data, tmp_path):
    instances = load(data)
    cache = PairwiseCache(tmp_path / "cache", offline=False)
    judge = PaidJudge()

    with pytest.raises(BudgetExceeded, match="re-running with a higher limit resumes"):
        run(instances, judge, cache, concurrency=1, max_cost_usd=0.015)
    assert judge.calls == 2

    results = run(instances, judge, cache)
    assert judge.calls == 2 * N_BALANCED
    assert sum(r.cached for r in results.values()) == 2


def test_the_budget_counts_only_new_verdicts_when_resuming(data, tmp_path):
    instances = load(data)
    cache = PairwiseCache(tmp_path / "cache", offline=False)
    judge = PaidJudge()
    run(instances, judge, cache)  # 24 verdicts at a cent each, all now cached
    assert judge.calls == 2 * N_BALANCED

    results = run(instances, judge, cache, max_cost_usd=0.015)  # well under what was spent

    assert judge.calls == 2 * N_BALANCED  # nothing new was asked for
    assert all(r.cached for r in results.values())


def test_offline_misses_are_counted_errors(data, tmp_path, capsys):
    out_path = tmp_path / "result.json"
    code, out, _ = main_out(
        capsys,
        *("--data", str(data), "--cache-dir", str(tmp_path / "empty")),
        *("--provider", "fake-first", "--offline", "--out", str(out_path)),
    )

    assert code == 1
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    [(message, count)] = doc["errors"].items()
    assert count == 2 * N_BALANCED
    assert "offline mode is on" in message
    assert doc["groups"][-1]["accuracy_average"]["rate"] is None
    assert "Judge errors" in out


# --- output ----------


def test_the_json_is_byte_stable_and_the_table_names_its_interval(data, tmp_path, capsys):
    common = ("--data", str(data), "--cache-dir", str(tmp_path / "c"), "--provider", "fake-longer")
    first_path, second_path = tmp_path / "1.json", tmp_path / "2.json"

    code, out, _ = main_out(capsys, *common, "--out", str(first_path))
    assert code == 0
    assert main_out(capsys, *common, "--out", str(second_path))[0] == 0  # from the cache

    assert first_path.read_bytes() == second_path.read_bytes()
    doc = json.loads(first_path.read_text(encoding="utf-8"))
    assert list(doc) == sorted(doc)
    assert "generated_at" not in doc
    assert doc["interval"] == "95% Wald"
    assert doc["dataset"]["commit"] == llmbar_fetch.COMMIT
    assert len(doc["instances"]) == N_BALANCED
    assert "95% Wald (`REPLAY_STATS_IMPL=baseline`)" in out
    assert "| Overall | 12 | 0.0% " in out  # the wrong output is the longer one throughout

    stamped = tmp_path / "3.json"
    main_out(capsys, *common, "--out", str(stamped), "--timestamp", "2026-10-14T00:00:00Z")
    assert json.loads(stamped.read_text(encoding="utf-8"))["generated_at"] == "2026-10-14T00:00:00Z"


def test_the_interval_follows_the_statistics_implementation(monkeypatch: pytest.MonkeyPatch):
    rows = [Row("Natural", 0, 1, 1, 2)]
    monkeypatch.setattr(stats, "wilson_interval", stats_baseline.wilson_interval)

    assert "95% Wilson (`REPLAY_STATS_IMPL=core`)" in markdown(
        rows, judge_id="j", stats_name="core"
    )


def test_an_unimplemented_statistics_implementation_fails_before_judging(
    data, tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
):
    def missing(*_args: object, **_kwargs: object) -> tuple[float, float]:
        raise NotImplementedError("synthetic: not implemented")

    monkeypatch.setattr(stats, "wilson_interval", missing)
    monkeypatch.setenv("REPLAY_STATS_IMPL", "core")

    code, _, err = main_out(
        capsys, "--data", str(data), "--cache-dir", str(tmp_path / "c"), "--provider", "fake-first"
    )
    assert code == 2
    assert "set it to baseline" in err
    assert not (tmp_path / "c").exists()


def test_configuration_problems_exit_2(data, tmp_path, capsys):
    common = ("--cache-dir", str(tmp_path / "c"), "--provider", "fake-first")

    code, _, err = main_out(capsys, "--data", str(tmp_path / "missing"), *common)
    assert (code, "make data-llmbar" in err) == (2, True)
    code, _, err = main_out(capsys, "--data", str(data), *common, "--model", "m")
    assert (code, "--model and --effort" in err) == (2, True)
    code, _, err = main_out(capsys, "--data", str(data), *common, "--limit", "0")
    assert (code, "--limit" in err) == (2, True)
    bad = data / "Natural" / "dataset.json"
    bad.write_text(json.dumps([{"input": "synthetic", "label": 3}]), encoding="utf-8")
    code, _, err = main_out(capsys, "--data", str(data), *common)
    assert (code, "Natural/dataset.json[0]" in err) == (2, True)


def test_subset_adversarial_means_all_four():
    assert llmbar._subsets(["Adversarial"]) == list(SUBSETS[1:])
    assert llmbar._subsets(["Adversarial/Manual", "Natural"]) == ["Natural", "Adversarial/Manual"]
    assert llmbar._subsets(None) == list(SUBSETS)


# --- the vendored data ----------


def readme_counts() -> dict[str, int]:
    text = (llmbar.DATA_DIR / "README.md").read_text(encoding="utf-8")
    return {
        m["subset"]: int(m["n"])
        for m in re.finditer(r"^\| `(?P<subset>[\w/]+)/dataset\.json` \| (?P<n>\d+) \|", text, re.M)
    }


def test_the_vendored_data_parses_and_matches_the_readme():
    instances = load()

    counts = {s: sum(1 for i in instances if i.subset == s) for s in SUBSETS}
    assert counts == readme_counts()
    assert counts == {s: llmbar_fetch.FILES[f"{s}/dataset.json"][2] for s in SUBSETS}
    assert len(instances) == 419  # LLMBar's published size: 100 Natural + 319 Adversarial
    assert {i.label for i in instances} == {1, 2}
    readme = (llmbar.DATA_DIR / "README.md").read_text(encoding="utf-8")
    assert llmbar_fetch.COMMIT in readme


def test_the_vendored_files_match_the_pinned_hashes():
    counts = llmbar_fetch.verify()

    assert sum(c for c in counts.values() if c is not None) == 419
    assert (
        (llmbar.DATA_DIR / "LICENSE-LLMBar").read_text(encoding="utf-8").startswith("MIT License")
    )


@pytest.mark.parametrize("judge", ["first", "longer", "shorter"])
def test_the_committed_heuristic_floors_reproduce_byte_for_byte(judge, tmp_path, capsys):
    """What ``make eval-llmbar-heuristics`` writes, on the vendored data: the numbers in
    RESULTS.md, including the length-tie rule that decides the 0.5% position bias."""
    out_path = tmp_path / f"{judge}.json"
    code, _, _ = main_out(
        capsys,
        *("--provider", f"fake-{judge}", "--cache-dir", str(tmp_path / "cache")),
        *("--out", str(out_path)),
    )

    assert code == 0
    committed = llmbar.ROOT / "eval" / "results" / f"llmbar_heuristic_{judge}.json"
    assert out_path.read_bytes() == committed.read_bytes()
