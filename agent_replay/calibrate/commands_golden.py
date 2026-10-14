"""The ``replay golden`` and ``replay label`` commands (registered on the main app by
:mod:`agent_replay.cli`).

``replay golden add`` replays a suite into a golden set with objective labels,
``replay golden show`` summarises one, and ``replay label`` is the blind
labelling session that collects human labels, one rubric criterion at a time.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import structlog
import typer
from pydantic import JsonValue
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_replay._cli import EXIT_USAGE, console, fail
from agent_replay.calibrate.build import DEFAULT_VARIANT, OBJECTIVE_LABELLER, golden_from_suite
from agent_replay.calibrate.golden import GoldenItem, GoldenSet, Label, Verdict
from agent_replay.errors import ReplayError
from agent_replay.judge.rubric import Criterion, Rubric
from agent_replay.judge.transcript import JudgeInput

log = structlog.get_logger("agent_replay")

LABELLER_ENV = "REPLAY_LABELLER"


def now() -> datetime:
    """When a human label is given; tests replace this to stay off the wall clock."""
    return datetime.now(UTC)


class Mode(StrEnum):
    full = "full"
    model_live = "model-live"
    tools_live = "tools-live"


golden_app = typer.Typer(
    help="Build and inspect golden sets: judge transcripts with reference labels.",
    no_args_is_help=True,
)

_DIR = typer.Argument(metavar="DIR", help="Golden set directory (items.jsonl, labels.jsonl).")


@golden_app.command("add")
def add_cmd(
    suite_path: Annotated[
        Path, typer.Argument(metavar="SUITE", help="Suite file (replay.toml) to replay.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Golden set directory to create or extend.")],
    variant: Annotated[
        str, typer.Option(help="Name recorded in each item's source (a mutant's name, say).")
    ] = DEFAULT_VARIANT,
    agent: Annotated[
        str | None,
        typer.Option(help="Replay every scenario against this agent (module:function)."),
    ] = None,
    mode: Annotated[Mode | None, typer.Option(help="Override every scenario's mode.")] = None,
    client: Annotated[
        str | None,
        typer.Option(help="Live model client for model-live: 'anthropic' or module:factory."),
    ] = None,
) -> None:
    """Replay SUITE without its judge checks and add each run to the golden set at DIR.

    Runs the judge would see identically are one item. Each item gets an
    objective label (labeller 'checks'): pass when the agent finished and every
    non-judge check passed. Re-running with the same cassettes changes nothing.

    Runs of one transcript with different verdicts are a conflict and get no new
    label, whether they come from one call or from two: within one call the
    item gets no objective label; when an item already has one and a later call
    disagrees, the earlier label stands and the conflict is reported. Exit 2
    when a scenario could not run (the others are still added).
    """
    try:
        golden = GoldenSet.load(out, must_exist=False)
        build = golden_from_suite(
            suite_path,
            variant=variant,
            agent=agent,
            mode=mode.value if mode else None,
            client=client,
        )
        before = {i.id: i for i in golden.items}
        golden, added = golden.add_items(build.items)
        gained = sum(
            1
            for i in build.items
            if i.id in before and golden.item(i.id).sources != before[i.id].sources
        )
        current = golden.current(source="objective", labeller=OBJECTIVE_LABELLER)
        fresh: list[Label] = []
        disagree: list[tuple[Label, Label]] = []
        for label in build.labels:
            old = current.get((label.item, label.criterion, label.source, label.labeller))
            if old is not None and old.label != label.label:
                disagree.append((old, label))
                continue
            if old is not None and old.note == label.note:
                continue
            fresh.append(label)
        golden.append_labels(fresh)
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    log.info(
        "golden.add",
        out=str(out),
        items=len(build.items),
        new=added,
        labels=len(fresh),
        conflicts=len(build.conflicts) + len(disagree),
    )
    console.print(
        f"{len(build.items)} item(s) from {escape(str(suite_path))} ({escape(variant)}): "
        f"{added} new, {gained} existing gained a source, "
        f"{len(build.items) - added - gained} unchanged; {len(fresh)} objective label(s) written"
    )
    for key in build.conflicts:
        console.print(
            f"[yellow]item {key}: runs with this transcript got different check verdicts; "
            "no objective label written for it[/]"
        )
    for old, new in disagree:
        console.print(
            f"[yellow]item {old.item}: conflict, this run's checks say {new.label} "
            f"({escape(variant)}) but its objective label says {old.label}; the earlier "
            "label stands and nothing was written for it[/]"
        )
    if build.skipped:
        for s in build.skipped:
            console.print(f"[bold yellow]skipped {escape(s.name)}:[/] {escape(s.reason)}")
        console.print(
            f"[bold yellow]{len(build.skipped)} scenario(s) could not run; fix the suite[/]"
        )
        raise typer.Exit(EXIT_USAGE)


@golden_app.command("show")
def show_cmd(
    directory: Annotated[Path, _DIR],
    rubric_path: Annotated[
        Path | None,
        typer.Option(
            "--rubric", help="Rubric YAML: also count stale and missing human labels for it."
        ),
    ] = None,
) -> None:
    """Summarise the golden set at DIR: items, labels per source, labeller and criterion."""
    try:
        golden = GoldenSet.load(directory)
        rubric = Rubric.load(rubric_path) if rubric_path is not None else None
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    variants = Counter(s.variant or "?" for i in golden.items for s in i.sources)
    runs = sum(variants.values())
    console.print(
        f"[bold]{escape(str(directory))}[/]: {len(golden.items)} item(s) from {runs} run(s) ("
        + ", ".join(f"{escape(v)} {n}" for v, n in sorted(variants.items()))
        + f"); {len(golden.labels)} label line(s)"
    )
    current = golden.current(rubric=rubric)
    if current:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        for column in ("source", "labeller", "criterion", "items", "pass", "fail"):
            table.add_column(column)
        groups: dict[tuple[str, str, str], Counter[str]] = {}
        for (_, criterion, source, labeller), label in sorted(current.items()):
            groups.setdefault((source, labeller, criterion), Counter())[label.label] += 1
        for (source, labeller, criterion), counts in sorted(groups.items()):
            table.add_row(
                source,
                escape(labeller),
                escape(criterion),
                str(counts.total()),
                str(counts["pass"]),
                str(counts["fail"]),
            )
        console.print(table)
    else:
        console.print("no current labels")
    if rubric is None:
        console.print("[dim]pass --rubric to count stale and missing human labels[/]")
        return
    stale = golden.stale_labels(rubric)
    if stale:
        console.print(
            f"[yellow]{stale} human label line(s) answer an earlier version of their "
            "criterion and no longer count; relabel with `replay label`[/]"
        )
    labelled = {(item, criterion) for (item, criterion, _, _), lb in current.items() if _human(lb)}
    for c in rubric.criteria:
        missing = sum(1 for i in golden.items if (i.id, c.id) not in labelled)
        console.print(f"{escape(c.id)}: {missing}/{len(golden.items)} item(s) lack a human label")


def _human(label: Label) -> bool:
    return label.source == "human"


# --- labelling -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Question:
    item: GoldenItem
    position: int
    """1-based position of the item among this session's items."""
    criterion: Criterion


_KEYS = "[bold]p[/]ass  [bold]f[/]ail  [bold]s[/]kip  [bold]n[/]ote  [bold]b[/]ack  [bold]q[/]uit"


def label_cmd(
    directory: Annotated[Path, _DIR],
    rubric_path: Annotated[
        Path, typer.Option("--rubric", help="Rubric YAML whose criteria are labelled.")
    ],
    labeller: Annotated[
        str | None,
        typer.Option(help=f"Who is labelling; defaults to ${LABELLER_ENV}."),
    ] = None,
    criterion_ids: Annotated[
        list[str] | None,
        typer.Option("--criterion", help="Label only this criterion (repeatable)."),
    ] = None,
    relabel: Annotated[
        bool,
        typer.Option(help="Ask again about items this labeller already labelled."),
    ] = False,
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle/--no-shuffle",
            help="Present items in a seeded random order, so runs of one task are not "
            "labelled back to back.",
        ),
    ] = True,
    seed: Annotated[int, typer.Option(help="Seed for --shuffle.")] = 0,
    limit: Annotated[int | None, typer.Option(min=1, help="Stop after this many items.")] = None,
) -> None:
    """Label the golden set at DIR by hand, one rubric criterion at a time.

    Labelling is blind: the judge's verdicts, the objective labels, other
    labellers' answers and which agent variant produced a run are never shown,
    because seeing them anchors the answer and inflates the agreement being
    measured. Each answer is saved as it is given, so quitting (q, Ctrl-C) keeps
    every earlier answer; run the command again to resume where you stopped.

    Keys: p pass, f fail, s skip, n note, b back, q quit. A note belongs to the
    question it was typed at and is saved with the next p or f on that question,
    even after going back (b) and returning; an empty note clears it, and a note
    on a question left unanswered is not saved (the summary counts it). b
    returns to the previous question to answer it again (the later answer wins);
    s on a question already answered keeps the earlier answer.
    """
    name = (labeller or os.environ.get(LABELLER_ENV, "")).strip()
    if not name:
        raise fail(f"who is labelling? Pass --labeller NAME or set {LABELLER_ENV}")
    try:
        golden = GoldenSet.load(directory)
        rubric = Rubric.load(rubric_path)
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    criteria = _select(rubric, criterion_ids or [])
    done = {
        (item, criterion)
        for (item, criterion, _, _), _lb in golden.current(
            source="human", labeller=name, rubric=rubric
        ).items()
    }
    items = sorted(golden.items, key=lambda i: i.id)
    if shuffle:
        random.Random(seed).shuffle(items)  # noqa: S311 - an order, not a secret
    pending = [i for i in items if relabel or any((i.id, c.id) not in done for c in criteria)][
        :limit
    ]
    if not pending:
        console.print(
            f"nothing to label: {escape(name)} has labelled every item on "
            f"{', '.join(c.id for c in criteria)} (pass --relabel to answer again)"
        )
        return
    questions = [
        _Question(item, n, c)
        for n, item in enumerate(pending, start=1)
        for c in criteria
        if relabel or (item.id, c.id) not in done
    ]
    console.print(
        f"labelling {len(pending)} item(s) of {len(golden.items)} as {escape(name)} on "
        f"{', '.join(c.id for c in criteria)}. Keys: {_KEYS}"
    )
    session = _Session(golden, questions, len(pending), name)
    stopped = session.run()
    answers = session.answers
    passed = sum(1 for v in answers.values() if v == "pass")
    unanswered = session.skipped - answers.keys()
    kept = session.kept & answers.keys()
    extra = f"; {len(kept)} skip(s) kept the earlier answer" if kept else ""
    if session.notes:
        extra += f"; {len(session.notes)} note(s) not saved (their questions went unanswered)"
    console.print(
        f"\n{'stopped' if stopped else 'done'}: {len(answers)} answer(s) saved "
        f"({passed} pass, {len(answers) - passed} fail), {len(unanswered)} skipped, "
        f"{len(questions) - len(answers) - len(unanswered)} not reached{extra}"
    )
    log.info("golden.label", labeller=name, answers=len(answers), skipped=len(unanswered))


class _Quit(Exception):
    pass


@dataclass
class _Session:
    """One labelling session: walks ``questions``, saving each answer as it is given."""

    golden: GoldenSet
    questions: list[_Question]
    items: int
    labeller: str
    answers: dict[int, Verdict] = field(default_factory=dict)
    """The latest answer per question index (a re-answer after ``b`` replaces it)."""
    skipped: set[int] = field(default_factory=set)
    """Questions skipped without an answer."""
    kept: set[int] = field(default_factory=set)
    """Questions skipped after ``b`` onto them: their earlier answer stands."""
    notes: dict[int, str] = field(default_factory=dict)
    """Notes typed but not yet saved, per question index; saved with that question's next
    answer, so going back (``b``) does not lose them or move them to another question."""

    def run(self) -> bool:
        """Ask every question; True when the labeller quit early."""
        index, shown, asked = 0, -1, -1
        try:
            while index < len(self.questions):
                q = self.questions[index]
                if q.position != shown:
                    _show_item(q.item.transcript, q.position, self.items, q.item.id)
                    shown = q.position
                if index != asked:
                    _show_criterion(q.criterion)
                    asked = index
                raw = _ask(f"{q.criterion.id} [p/f/s/n/b/q] > ").lower()
                if raw == "q":
                    return True
                if raw in ("p", "f"):
                    verdict: Verdict = "pass" if raw == "p" else "fail"
                    self.answer(index, verdict, self.notes.pop(index, None))
                    index += 1
                elif raw == "s":
                    self.skip(index)
                    index += 1
                elif raw == "n":
                    self.note(index)
                elif raw == "b":
                    back = self.back(index)
                    if self.questions[back].position != q.position:
                        shown = -1
                    index = back
                else:
                    console.print(f"[yellow]answer with one of: {_KEYS}[/]")
        except _Quit:
            return True
        return False

    def skip(self, index: int) -> None:
        if index in self.answers:
            self.kept.add(index)
            console.print(f"[dim]skipped; your earlier answer ({self.answers[index]}) stands[/]")
        else:
            self.skipped.add(index)

    def note(self, index: int) -> None:
        text = _ask("note > ")
        if text:
            self.notes[index] = text
            console.print("[dim]the note goes with your answer to this criterion[/]")
        elif self.notes.pop(index, None) is not None:
            console.print("[dim]note cleared[/]")

    def back(self, index: int) -> int:
        """The question before ``index`` (``index`` itself at the first question)."""
        if index == 0:
            console.print("[yellow]this is the first question of the session[/]")
            return 0
        if index in self.notes:
            console.print("[dim]your note stays with this criterion until you answer it[/]")
        return index - 1

    def answer(self, index: int, verdict: Verdict, note: str | None) -> None:
        q = self.questions[index]
        label = Label(
            item=q.item.id,
            criterion=q.criterion.id,
            criterion_fingerprint=q.criterion.fingerprint,
            label=verdict,
            source="human",
            labeller=self.labeller,
            labelled_at=now(),
            note=note,
        )
        self.golden = self.golden.append_labels([label])
        self.answers[index] = verdict
        self.skipped.discard(index)
        self.kept.discard(index)


def _ask(prompt: str) -> str:
    """One line of input; end of input or Ctrl-C ends the session (answers are saved)."""
    try:
        return console.input(escape(prompt)).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise _Quit from exc


def _select(rubric: Rubric, ids: Sequence[str]) -> list[Criterion]:
    if not ids:
        return list(rubric.criteria)
    unknown = [i for i in ids if i not in rubric.ids]
    if unknown:
        raise fail(
            f"the rubric has no criterion {', '.join(unknown)}; choose from {', '.join(rubric.ids)}"
        )
    return [c for c in rubric.criteria if c.id in ids]


def _value(value: JsonValue) -> Text:
    return Text(
        value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    )


def _show_item(transcript: JudgeInput, position: int, total: int, key: str) -> None:
    """The transcript, and nothing else about the item (see :func:`label_cmd` on blindness)."""
    console.rule(f"item {position}/{total} [dim]{key}[/]")
    console.print(Panel(Text(transcript.task), title="task", title_align="left"))
    if not transcript.tool_calls:
        console.print("[dim](the agent made no tool calls)[/]")
    for n, call in enumerate(transcript.tool_calls, start=1):
        console.print(Text(f"{n}. {call.name}", style="bold"))
        console.print(Text("   input: ", style="dim") + _value(call.input))
        if call.error is not None:
            console.print(
                Text(f"   error ({call.error.type}): ", style="red") + Text(call.error.message)
            )
        else:
            console.print(Text("   result: ", style="dim") + _value(call.result))
    if transcript.error is not None:
        console.print(
            Text("the agent did not finish: ", style="bold red")
            + Text(f"{transcript.error.type}: {transcript.error.message}")
        )
    console.print(Panel(_value(transcript.output), title="final output", title_align="left"))


def _show_criterion(c: Criterion) -> None:
    text = Text("\n")
    text.append(f"{c.id}: ", style="bold")
    text.append(c.question)
    if c.pass_when:
        text.append("\npass when: ", style="green")
        text.append(c.pass_when)
    if c.fail_when:
        text.append("\nfail when: ", style="red")
        text.append(c.fail_when)
    console.print(text)


def register(app: typer.Typer) -> None:
    """Add this module's commands to ``app``."""
    app.add_typer(golden_app, name="golden")
    app.command("label")(label_cmd)
