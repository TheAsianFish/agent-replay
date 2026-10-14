"""The golden set: judge transcripts with reference labels, what calibration measures against.

A golden set is a directory with two JSON Lines files::

    <dir>/items.jsonl    one GoldenItem per line: a run's judge transcript and where it came from
    <dir>/labels.jsonl   one Label per line, append-only: a verdict on one item for one criterion

**Items** are keyed by their transcript: ``id`` is the first 16 hex digits of
the transcript fingerprint, so two runs that the judge would see identically
(a mutant that changed nothing on one task, say) are one item with two
``sources`` and are counted once.

**Labels** come from a human (``replay label``) or from the suite's
deterministic checks (``objective``, written by ``replay golden add``). A human
label answers one rubric criterion and carries that criterion's fingerprint
(:attr:`agent_replay.judge.rubric.Criterion.fingerprint`): editing the
criterion's question invalidates its labels, editing another criterion does
not. An objective label is about the whole run (criterion :data:`OVERALL`,
"every criterion passes") because a deterministic check does not map onto one
rubric question. The file is append-only; for each item, criterion, source and
labeller the last line wins, so relabelling is an append and the history stays.
A human labeller named like the objective one (``checks``) is still a different
labeller: the source is part of the key.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from agent_replay.errors import ConfigError
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import JudgeInput

GOLDEN_SCHEMA_VERSION = 1
ITEMS_FILE = "items.jsonl"
LABELS_FILE = "labels.jsonl"
OVERALL = "_overall"
"""The criterion id of a whole-run label: the run passes when every rubric criterion passes.
It cannot collide with a rubric criterion (those start with a letter)."""

LabelSource = Literal["human", "objective"]
Verdict = Literal["pass", "fail"]

_ItemId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldenSource(_Frozen):
    """One run that produced an item's transcript."""

    suite: str | None = None
    scenario: str | None = None
    variant: str | None = None
    """The agent variant (``agent`` for the real one, else a mutant's name)."""
    cassette: str | None = None
    agent_model: str | None = None
    """The model that drove the agent in the recording, for the self-preference probe."""


class GoldenItem(_Frozen):
    schema_version: Literal[1] = 1
    id: _ItemId
    sources: tuple[GoldenSource, ...] = Field(min_length=1)
    transcript: JudgeInput

    @classmethod
    def from_transcript(cls, transcript: JudgeInput, source: GoldenSource) -> GoldenItem:
        return cls(id=item_id(transcript), sources=(source,), transcript=transcript)

    @property
    def agent_models(self) -> frozenset[str]:
        return frozenset(s.agent_model for s in self.sources if s.agent_model)


class Label(_Frozen):
    schema_version: Literal[1] = 1
    item: _ItemId
    criterion: str = Field(min_length=1)
    criterion_fingerprint: str | None = None
    """The labelled criterion's fingerprint; ``None`` for an :data:`OVERALL` label."""
    label: Verdict
    source: LabelSource
    labeller: str = Field(min_length=1)
    labelled_at: datetime
    note: str | None = None

    @property
    def passed(self) -> bool:
        return self.label == "pass"


def item_id(transcript: JudgeInput) -> str:
    return transcript.fingerprint.removeprefix("sha256:")[:16]


PROJECT_MARKERS = ("pyproject.toml", ".git")


def portable(path: Path) -> str:
    """``path`` as text that does not depend on where the command ran (POSIX separators):
    relative to the project it is in (the nearest ancestor holding one of
    :data:`PROJECT_MARKERS`), else to the working directory when inside it, else as given.
    So the same suite spelled from the repository root, from a subdirectory or by an
    absolute path records the same source, and ``golden add`` stays idempotent."""
    resolved = path.resolve()
    for ancestor in (resolved, *resolved.parents):
        if any((ancestor / marker).exists() for marker in PROJECT_MARKERS):
            return resolved.relative_to(ancestor).as_posix()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {str(path)!r}: {exc}") from exc
    rows: list[M] = []
    # Lines end in "\n" only: str.splitlines would also break on U+2028, U+0085 and the like,
    # which a transcript may contain unescaped.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "(root)"
            raise ConfigError(
                f"{str(path)!r} line {number} is not a valid {model.__name__} ({loc}: "
                f"{first['msg']}); fix or remove that line"
            ) from exc
    return rows


@dataclass(frozen=True)
class GoldenSet:
    """A golden set as read from disk. Build or extend one with :meth:`add_items` and
    :meth:`append_labels`; both write through to the directory."""

    directory: Path
    items: tuple[GoldenItem, ...]
    labels: tuple[Label, ...]

    @classmethod
    def load(cls, directory: str | Path, *, must_exist: bool = True) -> GoldenSet:
        directory = Path(directory)
        if must_exist and not (directory / ITEMS_FILE).is_file():
            raise ConfigError(
                f"{str(directory)!r} is not a golden set (no {ITEMS_FILE}); create one with "
                "`replay golden add SUITE --out DIR`"
            )
        items = _read_jsonl(directory / ITEMS_FILE, GoldenItem)
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise ConfigError(
                    f"{str(directory / ITEMS_FILE)!r} lists item {item.id} twice; keep one line"
                )
            if item.id != item_id(item.transcript):
                raise ConfigError(
                    f"item {item.id} in {str(directory / ITEMS_FILE)!r} does not match its "
                    "transcript's fingerprint; the transcript was edited, so re-add the run"
                )
            seen.add(item.id)
        labels = _read_jsonl(directory / LABELS_FILE, Label)
        unknown = sorted({lb.item for lb in labels} - seen)
        if unknown:
            raise ConfigError(
                f"{str(directory / LABELS_FILE)!r} labels items that are not in the set "
                f"({', '.join(unknown[:3])}{'…' if len(unknown) > 3 else ''}); restore "
                f"{ITEMS_FILE} or remove those labels"
            )
        return cls(directory=directory, items=tuple(items), labels=tuple(labels))

    @property
    def fingerprint(self) -> str:
        """The item ids, sorted and joined: which transcripts a report measured."""
        payload = "\n".join(sorted(i.id for i in self.items))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def item(self, item_id_: str) -> GoldenItem:
        for item in self.items:
            if item.id == item_id_:
                return item
        raise ConfigError(f"golden set {str(self.directory)!r} has no item {item_id_}")

    def add_items(self, new: Iterable[GoldenItem]) -> tuple[GoldenSet, int]:
        """Merge ``new`` into the set (an item already present gains the new sources) and
        rewrite ``items.jsonl`` atomically. Returns the new set and how many items were new."""
        by_id = {i.id: i for i in self.items}
        added = 0
        for item in new:
            old = by_id.get(item.id)
            if old is None:
                by_id[item.id] = item
                added += 1
                continue
            sources = old.sources + tuple(s for s in item.sources if s not in old.sources)
            by_id[item.id] = old.model_copy(update={"sources": sources})
        items = tuple(by_id.values())
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ITEMS_FILE
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
        os.replace(tmp, path)
        return GoldenSet(self.directory, items, self.labels), added

    def append_labels(self, new: Sequence[Label]) -> GoldenSet:
        """Append ``new`` to ``labels.jsonl`` and flush to disk, one line per label, so a
        labelling session that is interrupted keeps every answer already given."""
        ids = {i.id for i in self.items}
        for label in new:
            if label.item not in ids:
                raise ConfigError(f"cannot label item {label.item}: it is not in the golden set")
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / LABELS_FILE).open("a", encoding="utf-8") as f:
            for label in new:
                f.write(label.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return GoldenSet(self.directory, self.items, self.labels + tuple(new))

    def current(
        self,
        *,
        source: LabelSource | None = None,
        labeller: str | None = None,
        rubric: Rubric | None = None,
    ) -> dict[tuple[str, str, str, str], Label]:
        """The latest label per (item, criterion, source, labeller), filtered by ``source``
        and ``labeller``. With ``rubric``, human labels whose criterion is not in the rubric, or
        whose criterion fingerprint differs from the rubric's (the question was edited
        since), are left out; :data:`OVERALL` labels are always kept."""
        fingerprints = {c.id: c.fingerprint for c in rubric.criteria} if rubric else None
        latest: dict[tuple[str, str, str, str], Label] = {}
        for label in self.labels:
            if source is not None and label.source != source:
                continue
            if labeller is not None and label.labeller != labeller:
                continue
            if (
                fingerprints is not None
                and label.criterion != OVERALL
                and fingerprints.get(label.criterion) != label.criterion_fingerprint
            ):
                continue
            latest[(label.item, label.criterion, label.source, label.labeller)] = label
        return latest

    def labellers(self, source: LabelSource = "human") -> list[str]:
        return sorted({lb.labeller for lb in self.labels if lb.source == source})

    def stale_labels(self, rubric: Rubric) -> int:
        """Human labels for a criterion of ``rubric`` whose question has changed since."""
        fingerprints = {c.id: c.fingerprint for c in rubric.criteria}
        return sum(
            1
            for lb in self.labels
            if lb.source == "human"
            and lb.criterion in fingerprints
            and lb.criterion_fingerprint != fingerprints[lb.criterion]
        )


def reference_labels(
    golden: GoldenSet,
    rubric: Rubric,
    *,
    reference: LabelSource,
    labeller: str | None = None,
) -> dict[str, dict[str, bool]]:
    """``{criterion: {item_id: passed}}`` from one labeller of the ``reference`` source.

    With several labellers of that source and none chosen, this is a
    :class:`ConfigError` naming them: averaging people silently would hide their
    disagreement.
    """
    if labeller is None:
        names = golden.labellers(reference)
        if len(names) > 1:
            raise ConfigError(
                f"the golden set has {reference} labels from {len(names)} labellers "
                f"({', '.join(names)}); choose one with --labeller"
            )
    out: dict[str, dict[str, bool]] = {}
    for (item, criterion, _, _), label in golden.current(
        source=reference, labeller=labeller, rubric=rubric
    ).items():
        out.setdefault(criterion, {})[item] = label.passed
    return out
