"""Cassette schema v2: one JSON object per line.

Line 1 is a ``header``; then one ``envelope`` per intercepted call, in the order
the calls finished; the last line is a ``footer`` with the run's outcome.

Versions are frozen once released. v2 added the ``clock`` and ``random``
envelope kinds; v1 files are still read, through the frozen v1 models in
:mod:`agent_replay._cassette_v1` and a migration, and every write is v2.

A path ending in ``.xz`` (``run.jsonl.xz``) is the same JSONL in the xz format
(stdlib :mod:`lzma`, preset 6), read and written transparently; the standard
``xz`` tool converts either way. Every model envelope repeats the conversation
so far, so bytes grow with the square of the turns; xz's 8 MB dictionary sees
those copies (``make bench-scale``: 105x at 100 turns), where gzip's 32 KB
window stops seeing them once one envelope outgrows it (3.8x), so ``.gz`` and
other suffixes are refused rather than silently stored uncompressed.
"""

from __future__ import annotations

import json
import lzma
import os
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_replay import _cassette_v1 as v1
from agent_replay.errors import CassetteError

SCHEMA_VERSION = 2
READABLE_VERSIONS = (1, 2)
COMPRESSED_SUFFIX = ".xz"
XZ_PRESET = 6
_XZ_MAGIC = b"\xfd7zXZ\x00"
_REFUSED_SUFFIXES = (".gz", ".gzip", ".bz2", ".zst", ".zstd", ".lz4", ".zip", ".br")


class Kind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    CLOCK = "clock"
    """A clock read through :func:`agent_replay.now` (v2)."""
    RANDOM = "random"
    """A random draw through :func:`agent_replay.random` and friends (v2)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorInfo(_Strict):
    """An exception captured from a call or from the agent itself."""

    type: str
    """The exception class's qualified name, e.g. ``KeyError`` or ``Outer.Error``."""
    module: str | None = None
    """The module defining it, so replay can re-raise an instance of the same class."""
    message: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> ErrorInfo:
        from agent_replay.errors import RecordedException  # noqa: PLC0415 - avoid an import cycle

        if isinstance(exc, RecordedException):  # re-recording a replayed failure
            return cls(type=exc.type_name, module=exc.module, message=exc.message)
        kind = type(exc)
        return cls(type=kind.__qualname__, module=kind.__module__, message=str(exc))


class Usage(_Strict):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Header(_Strict):
    type: Literal["header"] = "header"
    schema_version: Literal[2] = 2
    cassette_id: str
    created_at: datetime
    agent: str | None = None
    task: str | None = None
    recorder: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Envelope(_Strict):
    """One intercepted boundary call: what was asked and what came back."""

    type: Literal["envelope"] = "envelope"
    seq: int
    kind: Kind
    name: str
    fingerprint: str
    request: JsonValue
    response: JsonValue = None
    error: ErrorInfo | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    latency_ms: float
    span_id: str
    parent_span_id: str | None = None


class Footer(_Strict):
    type: Literal["footer"] = "footer"
    status: Literal["ok", "error"]
    output: JsonValue = None
    error: ErrorInfo | None = None
    envelope_count: int


class Cassette(_Strict):
    header: Header
    envelopes: tuple[Envelope, ...]
    footer: Footer

    def top_level(self, *, order: Literal["finished", "started"] = "finished") -> list[Envelope]:
        """Envelopes for calls the agent made directly (not nested inside another call).

        ``finished`` is ``seq`` order, the order of the file. ``started`` is span
        order: what the agent asked for first, which is also the order a replay
        sees the calls in. They differ only when calls overlapped (thread pools).
        """
        top = [e for e in self.envelopes if e.parent_span_id is None]
        if order == "started":
            top.sort(key=_span_number)
        return top

    def in_start_order(self) -> list[Envelope]:
        """Every envelope, nested ones included, in the order the calls started."""
        return sorted(self.envelopes, key=_span_number)

    def tool_names(self) -> list[str]:
        """Names of the tools the agent called directly, in the order the calls started."""
        return [e.name for e in self.top_level(order="started") if e.kind is Kind.TOOL]

    def dumps(self) -> str:
        lines = [self.header.model_dump_json()]
        lines += [e.model_dump_json() for e in self.envelopes]
        lines.append(self.footer.model_dump_json())
        return "\n".join(lines) + "\n"

    def dump(self, path: str | Path) -> None:
        """Write atomically, so an interrupted recording never leaves half a cassette.

        A path ending in ``.xz`` is written xz-compressed.
        """
        path = Path(path)
        compressed = _is_compressed(path)
        data = self.dumps().encode("utf-8")
        if compressed:
            data = lzma.compress(data, format=lzma.FORMAT_XZ, preset=XZ_PRESET)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            # mkstemp creates the file owner-only (0600); give the cassette the
            # permissions a plain open() would, so it can be shared like any source file.
            os.chmod(tmp, 0o666 & ~_umask())
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def loads(cls, text: str, *, source: str = "<string>") -> Cassette:
        # Split on "\n" only: str.splitlines() also breaks on U+2028 and friends, which
        # JSON leaves unescaped inside strings.
        numbered = enumerate(text.split("\n"), start=1)
        lines = [(n, line) for n, line in numbered if line.strip()]
        if len(lines) < 2:
            raise CassetteError(
                f"{source}: a cassette needs a header and a footer line; "
                f"found {len(lines)} non-empty line(s). Re-record it with `replay record`."
            )
        version = _schema_version(source, lines[0][1])
        if version == 1:
            header, envelopes, footer = _load_v1(source, lines)
        else:
            header, envelopes, footer = _load_v2(source, lines)
        if footer.envelope_count != len(envelopes):
            raise CassetteError(
                f"{source}: footer says {footer.envelope_count} envelopes, found {len(envelopes)}; "
                "the file was edited or truncated. Re-record it with `replay record`."
            )
        if [e.seq for e in envelopes] != list(range(len(envelopes))):
            raise CassetteError(f"{source}: envelope seq numbers are not 0..n-1 in order")
        return cls(header=header, envelopes=tuple(envelopes), footer=footer)

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        """Read a cassette; a path ending in ``.xz`` is decompressed first."""
        path = Path(path)
        compressed = _is_compressed(path)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise CassetteError(
                f"cassette {path} does not exist; record one with `replay record`"
            ) from exc
        except OSError as exc:
            raise CassetteError(f"cannot read cassette {path}: {exc}") from exc
        if compressed:
            try:
                data = lzma.decompress(data, format=lzma.FORMAT_XZ)
            except lzma.LZMAError as exc:
                raise CassetteError(
                    f"cannot read cassette {path}: not a valid xz file ({exc}); a plain "
                    "JSONL cassette must not end in .xz (rename it, or compress it with `xz`)"
                ) from exc
        elif data.startswith(_XZ_MAGIC):
            raise CassetteError(
                f"cassette {path} is xz-compressed but its name does not end in .xz; "
                f"rename it to {path.name}.xz"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CassetteError(f"cannot read cassette {path}: {exc}") from exc
        return cls.loads(text, source=str(path))


def _is_compressed(path: Path) -> bool:
    """Whether ``path`` names an xz cassette; other compression suffixes are an error."""
    suffix = path.suffix.lower()
    if suffix == COMPRESSED_SUFFIX:
        return True
    if suffix in _REFUSED_SUFFIXES:
        raise CassetteError(
            f"cassette {path}: {suffix} is not a supported compression; name it "
            f"{path.with_suffix('').name}.xz (cassettes compress with xz: gzip's 32 KB "
            "window misses the conversation every model request repeats)"
        )
    return False


def is_cassette_path(path: str | Path) -> bool:
    """Whether ``path`` is named like a cassette: ``*.jsonl`` or ``*.jsonl.xz``."""
    name = Path(path).name.lower()
    return name.endswith((".jsonl", ".jsonl" + COMPRESSED_SUFFIX))


def _schema_version(source: str, line: str) -> int:
    try:
        version = json.loads(line).get("schema_version")
    except (ValueError, AttributeError) as exc:
        raise CassetteError(f"{source}:1: invalid header: not a JSON object") from exc
    if version not in READABLE_VERSIONS:
        raise CassetteError(
            f"{source}: cassette schema_version {version!r} is not supported by this "
            f"agent-replay (reads v1 and v{SCHEMA_VERSION}); upgrade agent-replay or re-record."
        )
    return int(version)


type _Lines = list[tuple[int, str]]


def _load_v2(source: str, lines: _Lines) -> tuple[Header, list[Envelope], Footer]:
    try:
        header = Header.model_validate_json(lines[0][1])
    except ValidationError as exc:
        raise CassetteError(f"{source}:1: invalid header: {exc}") from exc
    envelopes: list[Envelope] = []
    for n, line in lines[1:-1]:
        try:
            envelopes.append(Envelope.model_validate_json(line))
        except ValidationError as exc:
            raise CassetteError(f"{source}:{n}: invalid envelope: {exc}") from exc
    return header, envelopes, _footer(source, lines, Footer)


def _load_v1(source: str, lines: _Lines) -> tuple[Header, list[Envelope], Footer]:
    """Validate against the frozen v1 schema, then migrate to the current models."""
    try:
        header = v1.HeaderV1.model_validate_json(lines[0][1])
    except ValidationError as exc:
        raise CassetteError(f"{source}:1: invalid v1 header: {exc}") from exc
    envelopes: list[v1.EnvelopeV1] = []
    for n, line in lines[1:-1]:
        try:
            envelopes.append(v1.EnvelopeV1.model_validate_json(line))
        except ValidationError as exc:
            raise CassetteError(f"{source}:{n}: invalid v1 envelope: {exc}") from exc
    footer = _footer(source, lines, v1.FooterV1)
    h, es, f = v1.migrate(header, envelopes, footer)
    return (
        Header.model_validate(h),
        [Envelope.model_validate(e) for e in es],
        Footer.model_validate(f),
    )


def _footer[F: BaseModel](source: str, lines: _Lines, model: type[F]) -> F:
    n, line = lines[-1]
    try:
        return model.model_validate_json(line)
    except ValidationError as exc:
        raise CassetteError(
            f"{source}:{n}: the last line must be a footer; the recording may have been "
            f"cut short. Re-record it with `replay record`. ({exc.error_count()} error(s))"
        ) from exc


def _span_number(envelope: Envelope) -> tuple[int, int]:
    """Sort key for start order: the recorder's ``s<n>`` spans, with ``seq`` as the tiebreak."""
    span = envelope.span_id
    number = int(span[1:]) if span[:1] == "s" and span[1:].isdigit() else envelope.seq
    return (number, envelope.seq)


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current
