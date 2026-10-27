"""Tool-result drift: did a tool that ran live return what it returned when recorded?

In ``tools-live`` the tools run for real against recorded model replies. A tool
whose behaviour changed (a lookup that now builds an id, a status that is
stale) still answers the agent, the recorded model reply that follows is still
served, and unless a later check observes the effect the run passes. The
cassette already holds what each tool returned when the run was recorded; this
module compares the two.

Alignment. Each live tool step is paired with an unpaired recorded tool
envelope of the same name, at the same nesting level (top-level calls with
top-level ones, calls made inside another call with nested ones) and with the
same request fingerprint, i.e. the same arguments: the first such envelope (in
starting order) whose result it equals, else the first such envelope, and then
it is drift. Only such pairs are compared: when the arguments differ, a
different result is expected and says nothing about the tool (the agent or the
model changed, which other checks see). A live call with no such envelope is
counted, not compared, and consumes nothing, so an extra or changed call cannot
take the recording a later identical call is paired with; and a repeated read
(the same call before and after a write) that the agent adds or drops still
finds the recording with its result. A tool whose result depends on earlier
calls can still differ when the agent reorders, adds or drops a *write* before
a read: that is reported as drift, since the same call returned something no
recording of it holds.

Comparison. Results are compared as canonical JSON (keys sorted, so key order
never matters; ``1`` and ``1.0`` differ, as do ``true`` and ``1``). A call that
raised is compared by its exception type and message, and a call that raised
against one that returned is drift at the root. ``ignore_fields`` names, per
tool, paths whose values legitimately vary between runs (a timestamp, a
generated id): ``created_at``, ``items[*].updated_at``, ``*.etag``. ``*`` matches
any one key or index; ``[n]`` and ``.key`` address one. An ignored path is
skipped whether it is present on one side, both or neither; its parent is not
(``meta.created_at`` does not hide a missing ``meta``). Errors are compared
whole.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_replay._json import canonical_json
from agent_replay.cassette import Cassette, Envelope, ErrorInfo, Kind
from agent_replay.errors import ConfigError
from agent_replay.trajectory import Step, StepStatus, Trajectory

type Segment = str | int
type Path = tuple[Segment, ...]

WILDCARD = "*"
_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[(\*|\d+)\]")


def parse_path(text: str) -> Path:
    """``items[*].price`` -> ``("items", "*", "price")``; raises :class:`ConfigError`.

    Keys are separated by ``.``; ``[n]`` is a list index and ``[*]`` (or a bare
    ``*`` key) matches any one key or index.
    """
    segments: list[Segment] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        # A key is dotted everywhere but at the start: "a.b", never ".a" or "[0]b".
        misplaced_key = (
            match is not None
            and match.group(1) is not None
            and text.startswith(".", pos) == (pos == 0)
        )
        if match is None or misplaced_key:
            raise ConfigError(
                f"ignore_fields: cannot parse {text!r} at character {pos + 1}; write paths "
                "like 'created_at', 'items[*].updated_at' or 'items[0].id'"
            )
        key, index = match.groups()
        if key is not None:
            segments.append(key)
        else:
            segments.append(WILDCARD if index == WILDCARD else int(index))
        pos = match.end()
    if not segments:
        raise ConfigError("ignore_fields: an empty path ignores nothing; remove it")
    return tuple(segments)


def format_path(path: Path) -> str:
    """``("items", 0, "status")`` -> ``$.items[0].status``; the root is ``$``."""
    out = "$"
    for segment in path:
        out += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return out


def _ignored(path: Path, patterns: Sequence[Path]) -> bool:
    return any(
        len(pattern) == len(path)
        and all(p in (WILDCARD, s) for p, s in zip(pattern, path, strict=True))
        for pattern in patterns
    )


def first_difference(
    recorded: JsonValue, live: JsonValue, ignore: Sequence[Path] = ()
) -> Path | None:
    """The first path (keys in sorted order) where ``live`` differs from ``recorded``.

    ``None`` when they are equal as canonical JSON outside the ``ignore`` paths.
    A path present on one side only is a difference at that path; a value whose
    type differs (an object against a list) differs at its own path.
    """
    return _walk(recorded, live, (), ignore)


def _walk(a: JsonValue, b: JsonValue, path: Path, ignore: Sequence[Path]) -> Path | None:
    if ignore and _ignored(path, ignore):
        return None
    # (segment, value in a, value in b, present on both sides) for each child
    children: list[tuple[Segment, JsonValue, JsonValue, bool]]
    if isinstance(a, dict) and isinstance(b, dict):
        children = [(k, a.get(k), b.get(k), k in a and k in b) for k in sorted(a.keys() | b.keys())]
    elif isinstance(a, list) and isinstance(b, list):
        children = [
            (i, a[i] if i < len(a) else None, b[i] if i < len(b) else None, i < min(len(a), len(b)))
            for i in range(max(len(a), len(b)))
        ]
    else:
        return None if canonical_json(a) == canonical_json(b) else path
    for segment, x, y, both in children:
        child = (*path, segment)
        found = _walk(x, y, child, ignore) if both else child
        if found is not None and not (ignore and _ignored(found, ignore)):
            return found
    return None


class ToolDrift(BaseModel):
    """One live tool call whose result differs from its recording."""

    model_config = ConfigDict(frozen=True)

    tool: str
    step: int
    """The live step's index in the replayed trajectory."""
    envelope_seq: int
    """The recorded envelope it was compared with."""
    path: str
    """The first differing path, ``$`` for the whole result."""
    recorded: JsonValue
    live: JsonValue
    """The two values at ``path`` (an error as ``{"error": type, "message": ...}``;
    ``null`` where the path is missing on that side)."""


class DriftReport(BaseModel):
    """Every live tool call of one replay, aligned with the recording."""

    model_config = ConfigDict(frozen=True)

    compared: int = 0
    """Live calls paired with a recording made with the same arguments."""
    drifted: tuple[ToolDrift, ...] = ()
    arguments_differ: int = 0
    """Live calls of a recorded tool with no unpaired recording of the same arguments:
    not compared."""
    unrecorded: int = 0
    """Live calls of a tool with no unpaired recording at all: not compared."""

    @property
    def ok(self) -> bool:
        return not self.drifted


def _side(response: JsonValue, error: ErrorInfo | None) -> JsonValue:
    if error is not None:
        return {"error": error.type, "message": error.message}
    return response


def _at(value: JsonValue, path: Path) -> JsonValue:
    for segment in path:
        if isinstance(value, list) and isinstance(segment, int) and segment < len(value):
            value = value[segment]
            continue
        if isinstance(value, dict) and isinstance(segment, str) and segment in value:
            value = value[segment]
            continue
        return None
    return value


def compare(step: Step, envelope: Envelope, ignore: Sequence[Path] = ()) -> ToolDrift | None:
    """The drift between a live step and the envelope recorded for the same request."""
    recorded = _side(envelope.response, envelope.error)
    live = _side(step.response, step.error)
    if (envelope.error is None) != (step.error is None):
        path: Path | None = ()  # one raised, the other returned: no field can be ignored
    else:
        path = first_difference(recorded, live, ignore if step.error is None else ())
    if path is None:
        return None
    return ToolDrift(
        tool=step.name,
        step=step.index,
        envelope_seq=envelope.seq,
        path=format_path(path),
        recorded=_at(recorded, path),
        live=_at(live, path),
    )


def tool_drift(
    trajectory: Trajectory,
    cassette: Cassette,
    ignore_fields: Mapping[str, Sequence[str]] | None = None,
) -> DriftReport:
    """Align the live tool steps of a replay with the cassette's tool envelopes and compare.

    ``ignore_fields`` maps a tool name to paths (:func:`parse_path`) left out of
    its comparison. Steps served from the cassette are not compared: they *are*
    the recording.
    """
    ignore = {
        name: tuple(parse_path(p) for p in paths) for name, paths in (ignore_fields or {}).items()
    }
    queues: dict[tuple[str, bool], list[Envelope]] = {}
    for envelope in cassette.in_start_order():
        if envelope.kind is Kind.TOOL:
            nested = envelope.parent_span_id is not None
            queues.setdefault((envelope.name, nested), []).append(envelope)
    compared = arguments_differ = unrecorded = 0
    drifted: list[ToolDrift] = []
    for step in trajectory.steps:
        if step.kind is not Kind.TOOL or step.status is not StepStatus.LIVE:
            continue
        queue = queues.get((step.name, step.parent_span_id is not None), [])
        if not queue:
            unrecorded += 1
            continue
        same = [i for i, e in enumerate(queue) if e.fingerprint == step.fingerprint]
        if not same:
            arguments_differ += 1
            continue
        compared += 1
        # A repeated call (a read before and after a write) has several recordings; a
        # result equal to any of them is no drift, else it is reported against the first.
        paths = ignore.get(step.name, ())
        drifts = [compare(step, queue[i], paths) for i in same]
        match = next((i for i, d in zip(same, drifts, strict=True) if d is None), None)
        queue.pop(same[0] if match is None else match)
        if match is None and drifts[0] is not None:
            drifted.append(drifts[0])
    return DriftReport(
        compared=compared,
        drifted=tuple(drifted),
        arguments_differ=arguments_differ,
        unrecorded=unrecorded,
    )


def _preview(value: JsonValue, limit: int = 60) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(report: DriftReport) -> str:
    """One line for a check message: the count, then the first drift."""
    skipped = []
    if report.arguments_differ:
        skipped.append(f"{report.arguments_differ} with other arguments")
    if report.unrecorded:
        skipped.append(f"{report.unrecorded} unrecorded")
    not_compared = f" ({', '.join(skipped)} not compared)" if skipped else ""
    if report.ok:
        return f"{report.compared} live tool result(s) matched the recording{not_compared}"
    first = report.drifted[0]
    return (
        f"{len(report.drifted)} of {report.compared} live tool result(s) differ from the "
        f"recording{not_compared}; first: {first.tool} (step {first.step}) at {first.path}: "
        f"recorded {_preview(first.recorded)}, live {_preview(first.live)}"
    )
