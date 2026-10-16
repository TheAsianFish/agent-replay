"""Golden sets: the format, ``replay golden add`` / ``show`` and the ``replay label`` session."""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.calibrate import commands_golden
from agent_replay.calibrate.build import OBJECTIVE_LABELLER, golden_from_suite, objective_label
from agent_replay.calibrate.golden import (
    ITEMS_FILE,
    LABELS_FILE,
    OVERALL,
    GoldenItem,
    GoldenSet,
    GoldenSource,
    Label,
    reference_labels,
)
from agent_replay.cassette import Cassette
from agent_replay.checks.base import CheckResult
from agent_replay.cli import app
from agent_replay.errors import ConfigError
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import JudgeInput, ToolCallRecord
from agent_replay.runner import load_agent, replay

ROOT = Path(__file__).parents[1]
SUITE = Path("examples/retail_agent/replay.toml")
RUBRIC = Path("examples/retail_agent/rubric.yaml")
MUTANT = "examples.retail_agent.mutants:wrong_write_argument"
ENV = {"COLUMNS": "200", "REPLAY_LABELLER": ""}
T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(commands_golden, "now", lambda: T0)


def invoke(*args: str, input: str | None = None):
    return runner.invoke(app, list(args), env=ENV, input=input)


def transcript(task: str = "Cancel order #W1", output: str = "Done.") -> JudgeInput:
    return JudgeInput(
        task=task,
        output=output,
        tool_calls=(ToolCallRecord(name="cancel", input={"order": "#W1"}, result="ok"),),
        model_call_count=2,
        tool_call_count=1,
    )


def human(item: str, criterion: str, label: str, labeller: str, fingerprint: str) -> Label:
    return Label.model_validate(
        {
            "item": item,
            "criterion": criterion,
            "criterion_fingerprint": fingerprint,
            "label": label,
            "source": "human",
            "labeller": labeller,
            "labelled_at": T0,
        }
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fake retail suite with the real agent and one mutant, built once."""
    out = tmp_path_factory.mktemp("golden") / "set"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        for args in ((), ("--variant", "wrong_write_argument", "--agent", MUTANT)):
            result = invoke("golden", "add", str(SUITE), "--out", str(out), *args)
            assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def golden_dir(built: Path, tmp_path: Path) -> Path:
    shutil.copytree(built, tmp_path / "set")
    return tmp_path / "set"


# --- format -------------------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    rubric = Rubric.load(RUBRIC)
    item = GoldenItem.from_transcript(transcript(), GoldenSource(scenario="s", variant="agent"))
    golden, added = GoldenSet.load(tmp_path, must_exist=False).add_items([item])
    label = human(item.id, "resolved", "pass", "ann", rubric.criteria[0].fingerprint)
    golden.append_labels([label])

    loaded = GoldenSet.load(tmp_path)
    assert added == 1
    assert loaded.items == (item,)
    assert loaded.labels == (label,)
    assert loaded.items[0].transcript.fingerprint == transcript().fingerprint


def test_transcripts_with_unicode_line_separators_round_trip(tmp_path: Path) -> None:
    """U+2028 and friends are not escaped in the JSON; a reader that split on them would
    find half a record on the next "line"."""
    odd = JudgeInput(
        task="Line one\u2028line two\u0085three\x0cfour",
        output="Done.\u2029",
        tool_calls=(ToolCallRecord(name="cancel", input={"order": "#W1"}, result="a\u2028b"),),
        model_call_count=1,
        tool_call_count=1,
    )
    item = GoldenItem.from_transcript(odd, GoldenSource(scenario="s", variant="agent"))
    golden, _ = GoldenSet.load(tmp_path, must_exist=False).add_items([item])
    golden.append_labels([human(item.id, "resolved", "pass", "ann", "sha256:x")])

    loaded = GoldenSet.load(tmp_path)

    assert loaded.items == (item,)
    assert [lb.item for lb in loaded.labels] == [item.id]


def test_items_dedupe_by_transcript(tmp_path: Path) -> None:
    a = GoldenItem.from_transcript(transcript(), GoldenSource(scenario="s", variant="agent"))
    b = GoldenItem.from_transcript(transcript(), GoldenSource(scenario="s", variant="mutant"))
    c = GoldenItem.from_transcript(transcript(output="Other."), GoldenSource(variant="agent"))
    golden, added = GoldenSet.load(tmp_path, must_exist=False).add_items([a, b, c])
    golden, again = golden.add_items([b])

    assert (added, again) == (2, 0)
    assert [len(i.sources) for i in GoldenSet.load(tmp_path).items] == [2, 1]


def test_stale_labels_after_editing_a_criterion(tmp_path: Path) -> None:
    rubric = Rubric.load(RUBRIC)
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    golden, _ = GoldenSet.load(tmp_path, must_exist=False).add_items([item])
    golden = golden.append_labels(
        [human(item.id, c.id, "pass", "ann", c.fingerprint) for c in rubric.criteria]
    )
    edited_criterion = rubric.criteria[0].model_copy(update={"question": "Was it resolved?"})
    edited = rubric.model_copy(update={"criteria": (edited_criterion, *rubric.criteria[1:])})

    assert golden.stale_labels(rubric) == 0
    assert golden.stale_labels(edited) == 1
    current = golden.current(source="human", rubric=edited)
    assert {criterion for _, criterion, _, _ in current} == {"policy_followed", "accurate_report"}


def test_reference_labels_refuse_to_mix_labellers(tmp_path: Path) -> None:
    rubric = Rubric.load(RUBRIC)
    fp = rubric.criteria[0].fingerprint
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    golden, _ = GoldenSet.load(tmp_path, must_exist=False).add_items([item])
    golden = golden.append_labels(
        [
            human(item.id, "resolved", "pass", "ann", fp),
            human(item.id, "resolved", "fail", "bo", fp),
        ]
    )

    with pytest.raises(ConfigError, match=r"2 labellers \(ann, bo\); choose one with --labeller"):
        reference_labels(golden, rubric, reference="human")
    assert reference_labels(golden, rubric, reference="human", labeller="bo") == {
        "resolved": {item.id: False}
    }


def test_current_keys_on_the_source(tmp_path: Path) -> None:
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    golden, _ = GoldenSet.load(tmp_path, must_exist=False).add_items([item])
    objective = Label(
        item=item.id,
        criterion=OVERALL,
        label="pass",
        source="objective",
        labeller=OBJECTIVE_LABELLER,
        labelled_at=T0,
    )
    # a human who happens to be called "checks" is not the deterministic checks
    impostor = objective.model_copy(update={"source": "human", "label": "fail"})
    golden = golden.append_labels([objective, impostor])

    current = golden.current()
    assert current == {
        (item.id, OVERALL, "objective", OBJECTIVE_LABELLER): objective,
        (item.id, OVERALL, "human", OBJECTIVE_LABELLER): impostor,
    }


def _write(directory: Path, items: Sequence[str], labels: Sequence[str] = ()) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ITEMS_FILE).write_text("".join(f"{x}\n" for x in items), encoding="utf-8")
    (directory / LABELS_FILE).write_text("".join(f"{x}\n" for x in labels), encoding="utf-8")


def test_load_rejects_a_duplicate_item(tmp_path: Path) -> None:
    line = GoldenItem.from_transcript(transcript(), GoldenSource()).model_dump_json()
    _write(tmp_path, [line, line])
    with pytest.raises(ConfigError, match=r"lists item [0-9a-f]{16} twice; keep one line"):
        GoldenSet.load(tmp_path)


def test_load_rejects_an_id_that_does_not_match_the_transcript(tmp_path: Path) -> None:
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    edited = item.model_copy(update={"transcript": transcript(output="Edited.")})
    _write(tmp_path, [edited.model_dump_json()])
    with pytest.raises(ConfigError, match=rf"item {item.id} .* re-add the run"):
        GoldenSet.load(tmp_path)


def test_load_rejects_labels_for_unknown_items(tmp_path: Path) -> None:
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    stray = human("0123456789abcdef", "resolved", "pass", "ann", "sha256:x")
    _write(tmp_path, [item.model_dump_json()], [stray.model_dump_json()])
    with pytest.raises(
        ConfigError, match=r"labels items that are not in the set \(0123456789abcdef\)"
    ):
        GoldenSet.load(tmp_path)


def test_load_names_the_invalid_line(tmp_path: Path) -> None:
    item = GoldenItem.from_transcript(transcript(), GoldenSource())
    ok = human(item.id, "resolved", "pass", "ann", "sha256:x").model_dump_json()
    bad = ok.replace('"pass"', '"maybe"')
    _write(tmp_path, [item.model_dump_json()], [ok, "", bad])
    with pytest.raises(ConfigError, match=r"labels.jsonl' line 3 is not a valid Label \(label: "):
        GoldenSet.load(tmp_path)


# --- building -----------------------------------------------------------------------


def test_golden_from_suite_labels_the_real_agent_pass() -> None:
    build = golden_from_suite(SUITE)

    assert len(build.items) == 6
    assert not build.skipped
    assert not build.conflicts
    assert all(lb.label == "pass" and lb.criterion == OVERALL for lb in build.labels)
    assert {lb.labeller for lb in build.labels} == {OBJECTIVE_LABELLER}
    source = build.items[0].sources[0]
    assert source.scenario == "cancel_pending_laptop"
    assert source.cassette == "examples/retail_agent/cassettes/cancel_pending_laptop.jsonl"
    assert source.agent_model == "claude-opus-5-5"
    # Rebuilding is byte-stable: labelled_at is the cassette's recording time.
    assert golden_from_suite(SUITE) == build


def test_golden_from_suite_labels_a_mutant_fail() -> None:
    build = golden_from_suite(SUITE, variant="wrong_write_argument", agent=MUTANT, clock=lambda: T0)

    verdicts = {lb.label for lb in build.labels}
    assert "fail" in verdicts
    failing = [lb for lb in build.labels if lb.label == "fail"]
    assert all(lb.note and lb.note.startswith("failed: ") for lb in failing)
    assert all(lb.labelled_at == T0 for lb in build.labels)


def test_judge_checks_are_dropped_and_max_chars_kept(tmp_path: Path) -> None:
    cassette = ROOT / "examples/retail_agent/cassettes/lookup_latest_order_total.jsonl"
    rubric = (ROOT / RUBRIC).as_posix()
    suite = tmp_path / "suite.toml"
    suite.write_text(
        f"""
[defaults]
agent = "examples.retail_agent.agent:run"
checks = [
  {{ type = "judge", rubric = "{rubric}", offline = true, cache_dir = "c", max_chars = 40 }},
]

[[scenario]]
name = "lookup"
cassette = "{cassette.as_posix()}"

[[scenario]]
name = "missing"
cassette = "nope.jsonl"
""",
        encoding="utf-8",
    )
    build = golden_from_suite(suite)

    # Offline with an empty cache, a judge that ran would have broken the scenario.
    assert [s.name for s in build.skipped] == ["missing"]
    assert len(build.items) == 1
    assert build.items[0].transcript.task.endswith("chars]")
    assert build.labels[0].label == "pass"


def test_golden_add_is_idempotent(golden_dir: Path) -> None:
    before = {f: (golden_dir / f).read_bytes() for f in (ITEMS_FILE, LABELS_FILE)}
    result = invoke("golden", "add", str(ROOT / SUITE), "--out", str(golden_dir))

    assert result.exit_code == 0, result.output
    assert "0 new, 0 existing gained a source, 6 unchanged; 0 objective label(s)" in result.output
    assert {f: (golden_dir / f).read_bytes() for f in before} == before


def test_golden_add_with_a_mutant_merges_and_labels(built: Path) -> None:
    golden = GoldenSet.load(built)
    variants = {s.variant for i in golden.items for s in i.sources}
    objective = golden.current(source="objective")

    assert variants == {"agent", "wrong_write_argument"}
    assert len(objective) == len(golden.items)
    assert 6 < len(golden.items) <= 12
    assert {lb.label for lb in objective.values()} == {"pass", "fail"}


def test_the_objective_label_ignores_budget_checks() -> None:
    """A dearer run is not a wrong one: cost is nothing the judge can see in a transcript."""
    cassette = Cassette.load(
        ROOT / "examples/retail_agent/cassettes/lookup_latest_order_total.jsonl"
    )
    finished = replay(cassette, load_agent("examples.retail_agent.agent:run"))
    assert finished.ok
    budget_only = (CheckResult(check="budget", passed=False, message="26% over baseline"),)
    behaviour = (*budget_only, CheckResult(check="tool_sequence", passed=False, message="differs"))

    assert objective_label("0" * 16, finished, budget_only, T0).label == "pass"
    label = objective_label("0" * 16, finished, behaviour, T0)
    assert label.label == "fail"
    assert label.note == "failed: tool_sequence"  # the budget check is not even listed


def test_golden_add_is_idempotent_from_another_working_directory(
    golden_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = (golden_dir / ITEMS_FILE).read_bytes()
    monkeypatch.chdir(ROOT / "examples")

    result = invoke("golden", "add", "retail_agent/replay.toml", "--out", str(golden_dir))

    assert result.exit_code == 0, result.output
    assert "0 new, 0 existing gained a source" in result.output
    assert (golden_dir / ITEMS_FILE).read_bytes() == before
    monkeypatch.chdir("/")
    result = invoke("golden", "add", str(ROOT / SUITE), "--out", str(golden_dir))
    assert result.exit_code == 0, result.output
    assert (golden_dir / ITEMS_FILE).read_bytes() == before


def test_runs_of_one_transcript_with_different_verdicts_are_left_out(tmp_path: Path) -> None:
    cassette = (ROOT / "examples/retail_agent/cassettes/lookup_latest_order_total.jsonl").as_posix()
    suite = tmp_path / "suite.toml"
    suite.write_text(
        "[defaults]\n"
        'agent = "examples.retail_agent.agent:run"\n'
        f'[[scenario]]\nname = "plain"\ncassette = "{cassette}"\n'
        f'[[scenario]]\nname = "stricter"\ncassette = "{cassette}"\n'
        'checks = [ { type = "required_tools", tools = ["cancel_pending_order"] } ]\n',
        encoding="utf-8",
    )
    out = tmp_path / "set"

    build = golden_from_suite(suite)
    result = invoke("golden", "add", str(suite), "--out", str(out))

    assert len(build.conflicts) == 1
    assert build.items == ()
    assert build.labels == ()
    assert result.exit_code == 0, result.output
    assert "got different check verdicts; it was left out" in result.output
    assert GoldenSet.load(out).items == ()
    # Adding the plain scenario alone later is a fresh, consistent item.
    suite.write_text(
        '[defaults]\nagent = "examples.retail_agent.agent:run"\n'
        f'[[scenario]]\nname = "plain"\ncassette = "{cassette}"\n',
        encoding="utf-8",
    )
    assert invoke("golden", "add", str(suite), "--out", str(out)).exit_code == 0
    assert len(GoldenSet.load(out).items) == 1


def test_a_run_that_disagrees_with_the_stored_label_is_not_added_as_a_source(
    golden_dir: Path,
) -> None:
    build = golden_from_suite(SUITE)
    first = build.labels[0]
    golden = GoldenSet.load(golden_dir)
    sources_before = golden.item(first.item).sources
    golden.append_labels([first.model_copy(update={"label": "fail"})])

    result = invoke("golden", "add", str(SUITE), "--out", str(golden_dir), "--variant", "other")

    assert result.exit_code == 0, result.output
    assert "this run was not added as a source" in result.output
    after = GoldenSet.load(golden_dir).item(first.item).sources
    assert after == sources_before
    assert "other" not in {s.variant for s in after}


def test_golden_add_reports_a_verdict_that_differs_from_an_earlier_build(golden_dir: Path) -> None:
    build = golden_from_suite(SUITE)
    first = build.labels[0]
    assert first.label == "pass"
    # an earlier build of this very transcript said fail
    GoldenSet.load(golden_dir).append_labels([first.model_copy(update={"label": "fail"})])
    before = (golden_dir / LABELS_FILE).read_bytes()

    result = invoke("golden", "add", str(SUITE), "--out", str(golden_dir))

    assert result.exit_code == 0, result.output
    assert f"item {first.item}: conflict, this run's checks say pass (agent)" in result.output
    assert "the earlier label stands" in result.output
    assert "0 objective label(s) written" in result.output
    assert (golden_dir / LABELS_FILE).read_bytes() == before
    current = GoldenSet.load(golden_dir).current(source="objective")
    assert current[(first.item, OVERALL, "objective", OBJECTIVE_LABELLER)].label == "fail"


def test_golden_add_reports_a_broken_scenario(tmp_path: Path) -> None:
    suite = tmp_path / "suite.toml"
    suite.write_text(
        '[[scenario]]\nname = "gone"\ncassette = "missing.jsonl"\n'
        'agent = "examples.retail_agent.agent:run"\n',
        encoding="utf-8",
    )
    result = invoke("golden", "add", str(suite), "--out", str(tmp_path / "set"))

    assert result.exit_code == 2
    assert "skipped gone" in result.output
    assert "1 scenario(s) could not run" in result.output


def test_golden_show(golden_dir: Path) -> None:
    rubric = Rubric.load(RUBRIC)
    golden = GoldenSet.load(golden_dir)
    first = golden.items[0].id
    old = rubric.criteria[0].model_copy(update={"question": "An older question?"})
    golden.append_labels(
        [
            human(first, "resolved", "pass", "ann", rubric.criteria[0].fingerprint),
            human(first, "policy_followed", "fail", "ann", old.fingerprint),
        ]
    )
    n = len(golden.items)
    result = invoke("golden", "show", str(golden_dir), "--rubric", str(RUBRIC))

    assert result.exit_code == 0, result.output
    assert f"{n} item(s) from 12 run(s) (agent 6, wrong_write_argument 6)" in result.output
    assert "objective  checks    _overall" in result.output
    assert "human      ann       resolved   1      1     0" in result.output
    assert "1 human label line(s) answer an earlier version" in result.output
    assert f"resolved: {n - 1}/{n} item(s) lack a human label" in result.output
    assert f"policy_followed: {n}/{n} item(s) lack a human label" in result.output


def test_golden_show_needs_a_golden_set(tmp_path: Path) -> None:
    result = invoke("golden", "show", str(tmp_path))
    assert result.exit_code == 2
    assert "replay golden add SUITE --out DIR" in result.output


# --- labelling ----------------------------------------------------------------------


def label(golden_dir: Path, *extra: str, input: str) -> tuple[str, list[Label]]:
    result = invoke(
        "label", str(golden_dir), "--rubric", str(RUBRIC), "--labeller", "ann", *extra, input=input
    )
    assert result.exit_code == 0, result.output
    humans = [lb for lb in GoldenSet.load(golden_dir).labels if lb.source == "human"]
    return result.output, humans


def test_label_needs_a_labeller(golden_dir: Path) -> None:
    result = invoke("label", str(golden_dir), "--rubric", str(RUBRIC))
    assert result.exit_code == 2
    assert "Pass --labeller NAME or set REPLAY_LABELLER" in result.output


def test_label_rejects_an_unknown_criterion(golden_dir: Path) -> None:
    result = invoke(
        "label", str(golden_dir), "--rubric", str(RUBRIC), "--labeller", "ann", "--criterion", "x"
    )
    assert result.exit_code == 2
    assert "choose from resolved, policy_followed, accurate_report" in result.output


def test_label_session_pass_fail_skip_note_back_quit(golden_dir: Path) -> None:
    rubric = Rubric.load(RUBRIC)
    # Item 1: resolved p, policy_followed f, back, re-answer p with a note, accurate_report s.
    # Item 2: resolved: an unknown key, then f, then quit.
    keys = ["p", "f", "b", "n", "checked twice", "p", "s", "?", "f", "q"]
    output, labels = label(golden_dir, "--no-shuffle", input="\n".join(keys) + "\n")

    first, second = sorted(i.id for i in GoldenSet.load(golden_dir).items)[:2]
    got = [(lb.item, lb.criterion, lb.label, lb.note) for lb in labels]
    assert got == [
        (first, "resolved", "pass", None),
        (first, "policy_followed", "fail", None),
        (first, "policy_followed", "pass", "checked twice"),
        (second, "resolved", "fail", None),
    ]
    fingerprints = {c.id: c.fingerprint for c in rubric.criteria}
    assert all(lb.criterion_fingerprint == fingerprints[lb.criterion] for lb in labels)
    assert all(lb.labelled_at == T0 and lb.labeller == "ann" for lb in labels)
    # The later answer wins.
    current = GoldenSet.load(golden_dir).current(source="human", labeller="ann", rubric=rubric)
    assert current[(first, "policy_followed", "human", "ann")].label == "pass"
    n = len(GoldenSet.load(golden_dir).items)
    assert f"item 1/{n}" in output
    assert f"item 2/{n}" in output
    assert "answer with one of" in output
    assert "stopped: 3 answer(s) saved (2 pass, 1 fail), 1 skipped" in output


def test_a_note_survives_going_back(golden_dir: Path) -> None:
    # resolved p; a note on policy_followed; back to resolved and change it to f; then
    # answer policy_followed: the note goes with that answer, not with resolved.
    keys = ["p", "n", "asked the customer", "b", "f", "p", "q"]
    output, labels = label(golden_dir, "--no-shuffle", input="\n".join(keys) + "\n")

    first = sorted(i.id for i in GoldenSet.load(golden_dir).items)[0]
    got = [(lb.item, lb.criterion, lb.label, lb.note) for lb in labels]
    assert got == [
        (first, "resolved", "pass", None),
        (first, "resolved", "fail", None),
        (first, "policy_followed", "pass", "asked the customer"),
    ]
    assert "your note stays with this criterion" in output
    assert "note(s) not saved" not in output


def test_an_unsaved_note_is_counted(golden_dir: Path) -> None:
    output, labels = label(golden_dir, "--no-shuffle", input="n\nunsure\nq\n")

    assert labels == []
    assert "1 note(s) not saved (their questions went unanswered)" in output


def test_skip_after_back_keeps_the_earlier_answer(golden_dir: Path) -> None:
    output, labels = label(golden_dir, "--no-shuffle", input="p\nb\ns\nq\n")

    assert [lb.label for lb in labels] == ["pass"]
    assert "your earlier answer (pass) stands" in output
    assert "stopped: 1 answer(s) saved (1 pass, 0 fail), 0 skipped" in output
    assert "1 skip(s) kept the earlier answer" in output


def test_back_at_the_first_question_stays_there(golden_dir: Path) -> None:
    output, labels = label(golden_dir, "--no-shuffle", input="b\np\nq\n")

    first = sorted(i.id for i in GoldenSet.load(golden_dir).items)[0]
    assert "this is the first question of the session" in output
    assert [(lb.item, lb.criterion, lb.label) for lb in labels] == [(first, "resolved", "pass")]
    assert "stopped: 1 answer(s) saved" in output


def test_label_is_blind(golden_dir: Path) -> None:
    output, _ = label(golden_dir, "--no-shuffle", input="p\n" * 200)

    lowered = output.lower()
    for leak in ("judge", "objective", OVERALL, "failed:", "wrong_write_argument", "checks"):
        assert leak not in lowered
    assert "done:" in output


def test_label_resumes_and_relabels(golden_dir: Path) -> None:
    n = len(GoldenSet.load(golden_dir).items)
    _, first = label(golden_dir, "--criterion", "resolved", "--limit", "2", input="p\np\n")
    output, after = label(golden_dir, "--criterion", "resolved", input="")

    assert len(first) == 2
    # Resuming skips the two labelled items; end of input stops the session cleanly.
    assert f"item 1/{n - 2}" in output
    assert "stopped: 0 answer(s) saved" in output
    assert after == first

    output, _ = label(golden_dir, "--criterion", "resolved", "--relabel", "--limit", "1", input="")
    assert re.search(r"item 1/1\b", output)  # not "item 1/10"


def test_label_asks_again_after_a_criterion_was_edited(golden_dir: Path, tmp_path: Path) -> None:
    n = len(GoldenSet.load(golden_dir).items)
    _, first = label(golden_dir, "--criterion", "resolved", "--limit", "1", input="p\n")
    assert len(first) == 1
    output, _ = label(golden_dir, "--criterion", "resolved", input="")
    assert f"item 1/{n - 1}" in output  # the answered item is not asked again

    text = RUBRIC.read_text(encoding="utf-8")
    edited = tmp_path / "rubric.yaml"
    assert "Did the agent do what the user asked, as far as the policy allows?" in text
    edited.write_text(
        text.replace(
            "Did the agent do what the user asked, as far as the policy allows?",
            "Did the agent do what the user asked, within the policy?",
        ),
        encoding="utf-8",
    )
    result = invoke(
        "label", str(golden_dir), "--rubric", str(edited), "--labeller", "ann",
        "--criterion", "resolved", input="",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert f"item 1/{n}" in result.output  # the edited question is asked for every item again


def test_label_nothing_left(golden_dir: Path) -> None:
    n = len(GoldenSet.load(golden_dir).items)
    _, labels = label(golden_dir, "--criterion", "accurate_report", input="f\n" * n)
    output, again = label(golden_dir, "--criterion", "accurate_report", input="")

    assert len(labels) == n
    assert "nothing to label" in output
    assert again == labels


def test_label_shuffle_is_seeded(golden_dir: Path, tmp_path: Path) -> None:
    other = tmp_path / "copy"
    shutil.copytree(golden_dir, other)
    _, a = label(
        golden_dir, "--criterion", "resolved", "--seed", "7", "--limit", "3", input="p\n" * 3
    )
    _, b = label(other, "--criterion", "resolved", "--seed", "7", "--limit", "3", input="p\n" * 3)

    assert [lb.item for lb in a] == [lb.item for lb in b]


def test_quitting_keeps_earlier_answers(golden_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["p", "f"])

    def fake_input(_prompt: str = "", **_kwargs: object) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(commands_golden.console, "input", fake_input)
    output, labels = label(golden_dir, "--no-shuffle", input="")

    assert [lb.label for lb in labels] == ["pass", "fail"]
    assert "stopped: 2 answer(s) saved" in output
