"""Cassette schema v1: one JSON object per line.

Line 1 is a ``header``; then one ``envelope`` per intercepted call, in the order
the calls finished; the last line is a ``footer`` with the run's outcome. v1
fields are frozen: a later schema adds a v2 model and a migration rather than
changing them.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_replay.errors import CassetteError

SCHEMA_VERSION = 1


class Kind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


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
    schema_version: Literal[1] = 1
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

    def top_level(self) -> list[Envelope]:
        """Envelopes for calls the agent made directly (not nested inside another call)."""
        return [e for e in self.envelopes if e.parent_span_id is None]

    def tool_names(self) -> list[str]:
        """Names of the tools the agent called directly, in the order the calls finished."""
        return [e.name for e in self.top_level() if e.kind is Kind.TOOL]

    def dumps(self) -> str:
        lines = [self.header.model_dump_json()]
        lines += [e.model_dump_json() for e in self.envelopes]
        lines.append(self.footer.model_dump_json())
        return "\n".join(lines) + "\n"

    def dump(self, path: str | Path) -> None:
        """Write atomically, so an interrupted recording never leaves half a cassette."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(self.dumps())
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
        try:
            header = Header.model_validate_json(lines[0][1])
        except ValidationError as exc:
            raise CassetteError(_header_error(source, lines[0][1], exc)) from exc
        envelopes: list[Envelope] = []
        for n, line in lines[1:-1]:
            try:
                envelopes.append(Envelope.model_validate_json(line))
            except ValidationError as exc:
                raise CassetteError(f"{source}:{n}: invalid envelope: {exc}") from exc
        n, line = lines[-1]
        try:
            footer = Footer.model_validate_json(line)
        except ValidationError as exc:
            raise CassetteError(
                f"{source}:{n}: the last line must be a footer; the recording may have been "
                f"cut short. Re-record it with `replay record`. ({exc.error_count()} error(s))"
            ) from exc
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
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CassetteError(
                f"cassette {path} does not exist; record one with `replay record`"
            ) from exc
        return cls.loads(text, source=str(path))


def _header_error(source: str, line: str, exc: ValidationError) -> str:
    for err in exc.errors():
        if err["loc"] == ("schema_version",) and err["type"] == "literal_error":
            return (
                f"{source}: cassette schema_version {err['input']!r} is not supported by this "
                f"agent-replay (reads v{SCHEMA_VERSION}); upgrade agent-replay or re-record."
            )
    return f"{source}:1: invalid header: {exc}"
