"""Trace diff: align two trajectories step by step (PLAN.md decision 8).

Two runs of the same scenario are aligned with a Needleman-Wunsch global
alignment over their top-level steps, so an inserted verification call shows
up as one ``added`` row rather than "everything after step 3 moved".

Costs (all symmetric, so ``diff(a, b)`` and ``diff(b, a)`` have the same
distance):

- identical step (same kind, name, request, response and error, compared as
  canonical JSON): ``0``;
- same kind and name, anything else different: ``0.5 + 0.5 * d``, where ``d``
  in ``[0, 1]`` is the argument distance of :func:`argument_distance`;
- gap (a step only one side has): ``1``;
- different kind or name: never substituted. A renamed call is a removal plus
  an addition, so names never cross.

A changed step costs at most 1, less than the 2 of removing it and adding it
back, so matching names always pair up when the order allows. Status, cost and
latency are reported but never compared: a step served ``approximate`` with
the same content as a ``live`` one is ``same``.

Ties between equally cheap alignments are broken deterministically, reading
both lists front to back: at each point prefer pairing the next two steps
(same or changed), then removing the next base step, then adding the next head
step. So a repeated call pairs with its earliest copy and a replaced call reads
``removed`` before ``added``. Swapping the sides keeps the distance but not
necessarily the row order: ``removed`` still comes first. Time and memory are
O(n * m) for n and m steps; trajectories hold at most a few hundred top-level
steps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_replay._json import canonical_json
from agent_replay.cassette import Cassette, Kind
from agent_replay.store import RunRecord, StepRecord
from agent_replay.trajectory import Step, StepStatus, Trajectory

Op = Literal["same", "changed", "added", "removed"]
OPS: tuple[Op, ...] = ("same", "changed", "added", "removed")
GAP_COST = 1.0
CHANGED_BASE_COST = 0.5
LAST_MESSAGE = "messages[-1]"


class DiffStep(BaseModel):
    """One step as the diff sees it: the common ground of live and stored steps."""

    model_config = ConfigDict(frozen=True)

    index: int
    kind: str
    name: str
    status: str
    request: JsonValue
    response: JsonValue = None
    error: JsonValue | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    @classmethod
    def from_step(cls, step: Step) -> DiffStep:
        return cls(
            index=step.index,
            kind=step.kind.value,
            name=step.name,
            status=step.status.value,
            request=step.request,
            response=step.response,
            error=step.error.model_dump(mode="json") if step.error else None,
            cost_usd=step.cost_usd,
            latency_ms=step.latency_ms,
        )

    @classmethod
    def from_record(cls, record: StepRecord) -> DiffStep:
        return cls(
            index=record.index,
            kind=record.kind,
            name=record.name,
            status=record.status,
            request=record.request,
            response=record.response,
            error=record.error,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
        )

    @property
    def label(self) -> str:
        return f"{self.index} {self.kind} {self.name}"


def steps_from_trajectory(trajectory: Trajectory) -> list[DiffStep]:
    """The agent's own steps (model and tool), in the order they started."""
    return [DiffStep.from_step(s) for s in trajectory.top_level()]


def steps_from_record(run: RunRecord) -> list[DiffStep]:
    """A stored run's top-level steps, in the order they started."""
    return [DiffStep.from_record(s) for s in run.steps if s.parent_span_id is None]


def steps_from_cassette(cassette: Cassette) -> list[DiffStep]:
    """A recording's top-level steps, indexed and ordered as its replay would see them."""
    return [
        DiffStep(
            index=index,
            kind=e.kind.value,
            name=e.name,
            status=StepStatus.LIVE.value,
            request=e.request,
            response=e.response,
            error=e.error.model_dump(mode="json") if e.error else None,
            cost_usd=e.cost_usd,
            latency_ms=e.latency_ms,
        )
        for index, e in enumerate(cassette.in_start_order())
        if e.parent_span_id is None
    ]


class FieldChange(BaseModel):
    """One field that differs between the two sides of a ``changed`` row.

    ``path`` is ``request.<key>`` for a top-level request key, ``request`` for a
    request that is not an object, ``request.messages[-1]`` for a model call's
    newest message (``request.messages[<i>]`` for the first earlier message that
    differs, ``request.messages`` when only the count does), ``response`` or
    ``error``. A key or message one side lacks is ``None`` with its ``*_present``
    flag false, which tells it apart from a genuine ``null``.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    base: JsonValue
    head: JsonValue
    base_present: bool = True
    head_present: bool = True


class DiffRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Op
    base: DiffStep | None
    head: DiffStep | None
    cost: float
    changes: list[FieldChange] = Field(default_factory=list)


class TraceDiff(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_label: str | None = None
    head_label: str | None = None
    rows: list[DiffRow]
    distance: float
    """Total alignment cost: 0 exactly when the two step lists are identical."""
    counts: dict[str, int]
    """Rows per op; every op is present."""
    base_cost_usd: float
    head_cost_usd: float
    base_latency_ms: float
    head_latency_ms: float


@dataclass(frozen=True, slots=True)
class _Prepared:
    step: DiffStep
    request: str
    content: str
    fields: Mapping[str, str] | None
    """Canonical JSON per compared request field, or ``None`` for a non-object request."""


def _fields(step: DiffStep) -> dict[str, JsonValue] | None:
    """The request fields :func:`argument_distance` compares."""
    request = step.request
    if not isinstance(request, dict):
        return None
    if step.kind != Kind.MODEL.value:
        return dict(request)
    fields = {k: v for k, v in request.items() if k != "messages"}
    messages = request.get("messages")
    if isinstance(messages, list):
        fields[LAST_MESSAGE] = messages[-1] if messages else None
    elif "messages" in request:
        fields["messages"] = messages
    return fields


def _prepare(step: DiffStep) -> _Prepared:
    fields = _fields(step)
    return _Prepared(
        step=step,
        request=canonical_json(step.request),
        content=canonical_json([step.request, step.response, step.error]),
        fields=None if fields is None else {k: canonical_json(v) for k, v in fields.items()},
    )


def _distance(a: _Prepared, b: _Prepared) -> float:
    if a.fields is None or b.fields is None:
        return 0.0 if a.request == b.request else 1.0
    keys = a.fields.keys() | b.fields.keys()
    if not keys:
        return 0.0
    differing = sum(1 for k in keys if a.fields.get(k) != b.fields.get(k))
    return differing / len(keys)


def argument_distance(base: DiffStep, head: DiffStep) -> float:
    """How far apart two calls' requests are, in ``[0, 1]``.

    For object requests, the fraction of top-level keys (over the union of both
    sides' keys) whose canonical JSON values differ; a key one side lacks
    differs. A model call's request carries the whole conversation, whose
    history is already diffed as the earlier steps, so for model steps the
    ``messages`` list is compared by its last message only. A request that is
    not an object is 0 when canonically equal, else 1. Only the request counts:
    two calls asked the same thing but answered differently are distance 0 and
    cost 0.5.
    """
    return _distance(_prepare(base), _prepare(head))


def _substitution(a: _Prepared, b: _Prepared) -> float:
    if a.step.kind != b.step.kind or a.step.name != b.step.name:
        return math.inf
    if a.content == b.content:
        return 0.0
    return CHANGED_BASE_COST + (1 - CHANGED_BASE_COST) * _distance(a, b)


def _changes(a: _Prepared, b: _Prepared) -> list[FieldChange]:
    base, head = a.step, b.step
    changes: list[FieldChange] = []
    if a.request != b.request:
        if a.fields is None or b.fields is None:
            changes.append(FieldChange(path="request", base=base.request, head=head.request))
        else:
            base_fields, head_fields = _fields(base) or {}, _fields(head) or {}
            for key in sorted(a.fields.keys() | b.fields.keys()):
                if a.fields.get(key) != b.fields.get(key):
                    changes.append(
                        FieldChange(
                            path=f"request.{key}",
                            base=base_fields.get(key),
                            head=head_fields.get(key),
                            base_present=key in base_fields,
                            head_present=key in head_fields,
                        )
                    )
            if not changes:
                changes.append(_earlier_message_change(base.request, head.request))
    if canonical_json(base.response) != canonical_json(head.response):
        changes.append(FieldChange(path="response", base=base.response, head=head.response))
    if canonical_json(base.error) != canonical_json(head.error):
        changes.append(FieldChange(path="error", base=base.error, head=head.error))
    return changes


def _messages(request: JsonValue) -> list[JsonValue]:
    messages = request.get("messages") if isinstance(request, dict) else None
    return messages if isinstance(messages, list) else []


def _earlier_message_change(base: JsonValue, head: JsonValue) -> FieldChange:
    """The first message before the last that differs; or the counts when only those do."""
    base_messages, head_messages = _messages(base), _messages(head)
    for i, (b, h) in enumerate(zip(base_messages, head_messages, strict=False)):
        if canonical_json(b) != canonical_json(h):
            return FieldChange(path=f"request.messages[{i}]", base=b, head=h)
    i = min(len(base_messages), len(head_messages))
    if i < max(len(base_messages), len(head_messages)):
        shorter_is_base = len(base_messages) < len(head_messages)
        return FieldChange(
            path=f"request.messages[{i}]",
            base=None if shorter_is_base else base_messages[i],
            head=head_messages[i] if shorter_is_base else None,
            base_present=not shorter_is_base,
            head_present=shorter_is_base,
        )
    return FieldChange(
        path="request.messages",
        base=f"{len(base_messages)} message(s)",
        head=f"{len(head_messages)} message(s)",
    )


_DIAG, _DEL, _INS = 0, 1, 2


def _align(base: Sequence[_Prepared], head: Sequence[_Prepared]) -> list[DiffRow]:
    """Global alignment over suffixes, so the tie-break order reads front to back."""
    n, m = len(base), len(head)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[_DIAG] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        score[i][m], move[i][m] = (n - i) * GAP_COST, _DEL
    for j in range(m):
        score[n][j], move[n][j] = (m - j) * GAP_COST, _INS
    sub = [[_substitution(a, b) for b in head] for a in base]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best, best_move = score[i + 1][j + 1] + sub[i][j], _DIAG
            deletion = score[i + 1][j] + GAP_COST
            if deletion < best:
                best, best_move = deletion, _DEL
            insertion = score[i][j + 1] + GAP_COST
            if insertion < best:
                best, best_move = insertion, _INS
            score[i][j], move[i][j] = best, best_move

    rows: list[DiffRow] = []
    i = j = 0
    while i < n or j < m:
        step = move[i][j]
        if step == _DIAG:
            a, b, cost = base[i], head[j], sub[i][j]
            op: Op = "same" if cost == 0 else "changed"
            changes = _changes(a, b) if op == "changed" else []
            rows.append(DiffRow(op=op, base=a.step, head=b.step, cost=cost, changes=changes))
            i, j = i + 1, j + 1
        elif step == _DEL:
            rows.append(DiffRow(op="removed", base=base[i].step, head=None, cost=GAP_COST))
            i += 1
        else:
            rows.append(DiffRow(op="added", base=None, head=head[j].step, cost=GAP_COST))
            j += 1
    return rows


def _total(values: Iterable[float | None]) -> float:
    return sum(v for v in values if v is not None)


def diff_steps(
    base: Sequence[DiffStep],
    head: Sequence[DiffStep],
    *,
    base_label: str | None = None,
    head_label: str | None = None,
) -> TraceDiff:
    """Align ``head`` against ``base`` and summarise the differences."""
    rows = _align([_prepare(s) for s in base], [_prepare(s) for s in head])
    counts = dict.fromkeys(OPS, 0)
    for row in rows:
        counts[row.op] += 1
    return TraceDiff(
        base_label=base_label,
        head_label=head_label,
        rows=rows,
        distance=sum(r.cost for r in rows),
        counts=dict(counts),
        base_cost_usd=_total(s.cost_usd for s in base),
        head_cost_usd=_total(s.cost_usd for s in head),
        base_latency_ms=_total(s.latency_ms for s in base),
        head_latency_ms=_total(s.latency_ms for s in head),
    )


def diff_trajectories(base: Trajectory, head: Trajectory) -> TraceDiff:
    return diff_steps(steps_from_trajectory(base), steps_from_trajectory(head))


def diff_runs(base: RunRecord, head: RunRecord) -> TraceDiff:
    """Diff two stored runs, labelled by their ids."""
    return diff_steps(
        steps_from_record(base), steps_from_record(head), base_label=base.id, head_label=head.id
    )


def to_json(diff: TraceDiff) -> str:
    return diff.model_dump_json(indent=2) + "\n"


def preview(value: JsonValue, limit: int = 60) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def cost_text(value: float) -> str:
    """An alignment cost for display: at most three decimals, no trailing zeros."""
    return f"{round(value, 3):g}"


def describe_change(change: FieldChange, limit: int = 60) -> str:
    """``path: base → head`` with both values shortened to ``limit`` characters."""
    base = preview(change.base, limit) if change.base_present else "(absent)"
    head = preview(change.head, limit) if change.head_present else "(absent)"
    return f"{change.path}: {base} → {head}"


def summary(diff: TraceDiff) -> str:
    """One line: distance, row counts, cost and latency of each side."""
    counts = ", ".join(f"{diff.counts[op]} {op}" for op in OPS)
    return (
        f"distance {cost_text(diff.distance)} ({counts}); cost ${diff.base_cost_usd:.4f} → "
        f"${diff.head_cost_usd:.4f}; latency {diff.base_latency_ms:.0f} ms → "
        f"{diff.head_latency_ms:.0f} ms"
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _code(text: str) -> str:
    """A Markdown code span; a double-backtick span when the text has a backtick."""
    cell = _cell(text)
    return f"`` {cell} ``" if "`" in cell else f"`{cell}`"


def to_markdown(diff: TraceDiff) -> str:
    """A table with one row per aligned step pair, under a one-line summary."""
    title = "## Trace diff"
    if diff.base_label is not None and diff.head_label is not None:
        title += f": {_code(diff.base_label)} → {_code(diff.head_label)}"
    lines = [
        title,
        "",
        _cell(summary(diff)) + ".",
        "",
        "| Op | Base | Head | Cost | Changes |",
        "|---|---|---|---|---|",
    ]
    for row in diff.rows:
        base = _code(row.base.label) if row.base else "—"
        head = _code(row.head.label) if row.head else "—"
        changes = "<br>".join(_code(describe_change(c)) for c in row.changes)
        lines.append(f"| {row.op} | {base} | {head} | {cost_text(row.cost)} | {changes or '—'} |")
    return "\n".join(lines) + "\n"
