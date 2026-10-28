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

**Partial cassettes.** A recording that writes to a path (:class:`CassetteWriter`)
appends each line to ``<path>.partial`` as it completes, flushed but not fsynced,
so the file survives the process dying (SIGKILL, Ctrl-C, an OOM kill), not the
machine losing power. The partial is always plain JSONL, even for an ``.xz``
path: Python's ``lzma`` has no sync flush, and a stream compressor holds back
what it has not emitted yet (``make bench-scale``: after 140 of a 100-turn run's
201 lines, the emitted xz bytes decode to no envelope at all). Finishing appends
the footer and moves the partial into place (compressing it first for ``.xz``),
which gives the same bytes :meth:`Cassette.dump` writes. A ``.partial`` without a
footer loads as *interrupted*: a load-time state, not a footer status, so the
schema is unchanged. It replays (the run fails where the recording stopped) and
diffs like any cassette; ``replay ci`` refuses it.
"""

from __future__ import annotations

import gc
import json
import lzma
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_replay import _cassette_v1 as v1
from agent_replay.errors import CassetteError

SCHEMA_VERSION = 2
READABLE_VERSIONS = (1, 2)
COMPRESSED_SUFFIX = ".xz"
XZ_PRESET = 6
_XZ_MAGIC = b"\xfd7zXZ\x00"
_REFUSED_SUFFIXES = (".gz", ".gzip", ".bz2", ".zst", ".zstd", ".lz4", ".zip", ".br")
PARTIAL_SUFFIX = ".partial"
INTERRUPTED_ERROR = "RecordingInterrupted"
"""``Footer.error.type`` of the footer an interrupted cassette is given in memory."""
_COPY_CHUNK = 1 << 20


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
    interrupted: bool = False
    """Loaded from a ``.partial`` with no footer: the recording stopped after these
    envelopes. ``footer`` is then made up in memory (``status="error"``,
    :data:`INTERRUPTED_ERROR`) and never written; ``replay ci`` refuses such a cassette.
    """
    torn_bytes: int = 0
    """Bytes of an incomplete last line dropped from an interrupted cassette."""

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
        """The JSONL; an interrupted cassette is written as it was found, without a footer."""
        lines = [self.header.model_dump_json()]
        lines += [e.model_dump_json() for e in self.envelopes]
        if not self.interrupted:
            lines.append(self.footer.model_dump_json())
        return "\n".join(lines) + "\n"

    def dump(self, path: str | Path) -> None:
        """Write atomically, so an interrupted recording never leaves half a cassette.

        A path ending in ``.xz`` is written xz-compressed. An interrupted cassette
        can only be written to a ``.partial`` path, so it never passes for a finished one.
        """
        path = Path(path)
        if self.interrupted and not is_partial_path(path):
            raise CassetteError(
                f"cannot write an interrupted recording to {path}: it has no footer; "
                f"write it to {path.name}{PARTIAL_SUFFIX}, or re-record it with `replay record`"
            )
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
    def loads(cls, text: str, *, source: str = "<string>", partial: bool = False) -> Cassette:
        """Parse a cassette's JSONL; ``partial`` reads a footer-less file as interrupted.

        The garbage collector is paused while the envelopes are parsed (and only
        then): building a large cassette allocates millions of containers and none
        of them is garbage, so every collection the allocations trigger is wasted
        work. ``make bench-scale`` measures load with the collector on and off.
        """
        with _collector_paused():
            return cls._loads(text, source=source, partial=partial)

    @classmethod
    def _loads(cls, text: str, *, source: str, partial: bool) -> Cassette:
        # Split on "\n" only: str.splitlines() also breaks on U+2028 and friends, which
        # JSON leaves unescaped inside strings.
        torn = 0
        if partial:
            # A line is complete once its newline is written; a kill mid-write leaves
            # a torn last line, which is dropped (and counted) rather than misread.
            cut = text.rfind("\n") + 1
            torn = len(text[cut:].encode("utf-8"))
            text = text[:cut]
        numbered = enumerate(text.split("\n"), start=1)
        lines = [(n, line) for n, line in numbered if line.strip()]
        if partial and lines and not _is_footer(lines[-1][1]):
            return cls._interrupted(source, lines, torn)
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
    def _interrupted(cls, source: str, lines: _Lines, torn: int) -> Cassette:
        version = _schema_version(source, lines[0][1])
        if version != SCHEMA_VERSION:
            raise CassetteError(f"{source}: a partial cassette is always v{SCHEMA_VERSION}")
        try:
            header = Header.model_validate_json(lines[0][1])
        except ValidationError as exc:
            raise CassetteError(f"{source}:1: invalid header: {exc}") from exc
        envelopes = _envelopes(source, lines[1:])
        if [e.seq for e in envelopes] != list(range(len(envelopes))):
            raise CassetteError(f"{source}: envelope seq numbers are not 0..n-1 in order")
        footer = Footer(
            status="error",
            error=ErrorInfo(
                type=INTERRUPTED_ERROR,
                message=f"the recording stopped after {len(envelopes)} envelope(s)",
            ),
            envelope_count=len(envelopes),
        )
        return cls(
            header=header,
            envelopes=tuple(envelopes),
            footer=footer,
            interrupted=True,
            torn_bytes=torn,
        )

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        """Read a cassette; a path ending in ``.xz`` is decompressed first.

        A ``.partial`` path is read as plain JSONL; without a footer it is an
        interrupted recording (:attr:`interrupted`).
        """
        path = Path(path)
        if is_partial_path(path):
            return cls._load_partial(path)
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

    @classmethod
    def _load_partial(cls, path: Path) -> Cassette:
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise CassetteError(f"partial cassette {path} does not exist") from exc
        except OSError as exc:
            raise CassetteError(f"cannot read cassette {path}: {exc}") from exc
        if data.startswith(_XZ_MAGIC):
            raise CassetteError(
                f"{path} is xz-compressed, but a partial cassette is plain JSONL; "
                f"decompress it with `xz -dc {path.name}`"
            )
        # Cut at the last newline before decoding: a torn line may end inside a character.
        cut = data.rfind(b"\n") + 1
        try:
            text = data[:cut].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CassetteError(f"cannot read cassette {path}: {exc}") from exc
        if not text.strip():
            raise CassetteError(
                f"{path}: the recording stopped before its header was written; nothing to "
                "replay. Re-record it with `replay record`."
            )
        cassette = cls.loads(text, source=str(path), partial=True)
        if cassette.interrupted and cut < len(data):
            cassette = cassette.model_copy(update={"torn_bytes": len(data) - cut})
        return cassette

    def describe_interrupted(self, source: str | Path) -> str:
        """What an interrupted cassette holds, for an error message or a warning."""
        torn = (
            f", plus a torn last line of {self.torn_bytes} bytes (dropped)"
            if self.torn_bytes
            else ""
        )
        return (
            f"{source} is an interrupted recording: {len(self.envelopes)} envelope(s) and no "
            f"footer{torn}"
        )


class CassetteWriter:
    """Writes a recording line by line to ``<path>.partial``; :meth:`finish` moves it to ``path``.

    Every line is flushed as it is written, so a killed process leaves a partial
    that :meth:`Cassette.load` reads as interrupted. The finished file has the
    bytes :meth:`Cassette.dump` would write for the same cassette: the header, the
    envelopes in ``seq`` order and the footer, one compact JSON object a line; for
    an ``.xz`` path, those bytes through one :class:`lzma.LZMACompressor` at
    :data:`XZ_PRESET`, whose output does not depend on how the input is chunked.

    An existing partial is an earlier interrupted attempt, kept for diffing, so it
    is refused unless ``overwrite``. Not thread-safe by itself: the session calls
    it under its lock.
    """

    def __init__(
        self, path: str | os.PathLike[str], header: Header, *, overwrite: bool = False
    ) -> None:
        self.path = Path(path)
        _is_compressed(self.path)  # refuse .gz and friends before anything runs
        self.partial = partial_path(self.path)
        if self.partial.exists() and not overwrite:
            raise CassetteError(
                f"{self.partial} is an interrupted earlier recording; replay or diff it "
                f"(`replay run {self.partial} --mode full`), move it away, or pass --force"
            )
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[bytes] | None = self.partial.open("wb")
        self.envelopes = 0
        self._line(header.model_dump_json())

    def _line(self, json_line: str) -> None:
        if self._fh is None:
            raise CassetteError(f"the recording to {self.path} is already closed")
        self._fh.write(json_line.encode("utf-8") + b"\n")
        self._fh.flush()

    def append(self, envelope: Envelope) -> None:
        if envelope.seq != self.envelopes:
            raise CassetteError(f"envelope {envelope.seq} written out of order")
        self._line(envelope.model_dump_json())
        self.envelopes += 1

    def finish(self, footer: Footer) -> None:
        """Append the footer and move the finished cassette to :attr:`path`."""
        if footer.envelope_count != self.envelopes:
            raise CassetteError(
                f"footer counts {footer.envelope_count} envelopes, {self.envelopes} were written"
            )
        self._line(footer.model_dump_json())
        self.close()
        if not _is_compressed(self.path):
            self.partial.replace(self.path)
            return
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=XZ_PRESET)
            with os.fdopen(fd, "wb") as out, self.partial.open("rb") as src:
                while chunk := src.read(_COPY_CHUNK):
                    out.write(compressor.compress(chunk))
                out.write(compressor.flush())
            shutil.copymode(self.partial, tmp)
            Path(tmp).replace(self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.partial.unlink()

    def close(self) -> None:
        """Close the file and leave the partial where it is (an interrupted recording)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def partial_path(path: str | Path) -> Path:
    """Where a recording to ``path`` keeps its lines until it finishes: ``<path>.partial``."""
    path = Path(path)
    return path.with_name(path.name + PARTIAL_SUFFIX)


def is_partial_path(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(PARTIAL_SUFFIX)


_pause_lock = threading.Lock()
_pause_depth = 0
_paused_by_us = False


@contextmanager
def _collector_paused() -> Iterator[None]:
    """Disable the cyclic garbage collector for the block, unless something else already has.

    Process-wide, so nested and concurrent loads share one pause: the collector is
    re-enabled when the last of them ends, and left alone if it was off before.
    Agent code on other threads runs with the collector paused for as long as a
    load takes; that delays collections, it never skips freeing (reference
    counting still frees everything acyclic).
    """
    global _pause_depth, _paused_by_us  # noqa: PLW0603 - one process-wide pause
    with _pause_lock:
        if _pause_depth == 0:
            _paused_by_us = gc.isenabled()
            gc.disable()
        _pause_depth += 1
    try:
        yield
    finally:
        with _pause_lock:
            _pause_depth -= 1
            if _pause_depth == 0 and _paused_by_us:
                gc.enable()


def _is_footer(line: str) -> bool:
    try:
        obj = json.loads(line)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("type") == "footer"


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


def check_cassette_name(path: str | Path) -> None:
    """Refuse a compression suffix other than ``.xz`` before anything is recorded.

    ``dump`` refuses it too, but a command that runs a live agent first should
    fail before spending the calls, not after.
    """
    _is_compressed(Path(path))


def is_cassette_path(path: str | Path) -> bool:
    """Whether ``path`` is named like a cassette: ``*.jsonl`` or ``*.jsonl.xz``, or its partial."""
    name = Path(path).name.lower().removesuffix(PARTIAL_SUFFIX)
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
    return header, _envelopes(source, lines[1:-1]), _footer(source, lines, Footer)


def _envelopes(source: str, lines: _Lines) -> list[Envelope]:
    envelopes: list[Envelope] = []
    for n, line in lines:
        try:
            envelopes.append(Envelope.model_validate_json(line))
        except ValidationError as exc:
            raise CassetteError(f"{source}:{n}: invalid envelope: {exc}") from exc
    return envelopes


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
