"""Rubrics: what the judge grades a run on, loaded from YAML.

::

    name: retail-resolution        # [a-z0-9][a-z0-9_.-]*
    version: 1
    description: optional text
    instructions: optional extra guidance for the judge
    criteria:
      - id: resolved               # ^[a-z][a-z0-9_]*$, unique
        question: Did the agent ...?
        pass_when: optional
        fail_when: optional

A rubric's :attr:`Rubric.fingerprint` covers every field, so any edit
invalidates cached verdicts and calibration reports made with the old text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from agent_replay._json import canonical_json
from agent_replay.errors import ConfigError

MAX_CRITERIA = 10

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(_Strict):
    """One yes/no question the judge answers about a run."""

    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
    question: _Text
    pass_when: _Text | None = None
    fail_when: _Text | None = None


class Rubric(_Strict):
    name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=100)]
    version: int = Field(ge=1)
    description: _Text | None = None
    instructions: _Text | None = None
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=MAX_CRITERIA)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [c.id for c in self.criteria]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"criterion ids must be unique; repeated: {', '.join(repeated)}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        """Read and validate a rubric file; :class:`ConfigError` names the file and field."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(
                f"rubric file {str(path)!r} does not exist; check the path (suite paths are "
                "relative to the suite file)"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"cannot read rubric file {str(path)!r}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"rubric file {str(path)!r} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"rubric file {str(path)!r} must be a YAML mapping with name, version and criteria"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"rubric file {str(path)!r}: {_problems(exc)}") from exc

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.criteria)

    @property
    def fingerprint(self) -> str:
        """sha256 over the canonical JSON of the whole validated rubric."""
        payload = canonical_json(self.model_dump(mode="json"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render(self) -> str:
        """The rubric as prompt text: name, guidance, then one section per criterion."""
        lines = [f"# Rubric: {self.name} (version {self.version})"]
        if self.description:
            lines += ["", self.description]
        if self.instructions:
            lines += ["", "## Guidance", "", self.instructions]
        lines += ["", "## Criteria"]
        for c in self.criteria:
            lines += ["", f"### {c.id}", "", f"Question: {c.question}"]
            if c.pass_when:
                lines.append(f"Pass when: {c.pass_when}")
            if c.fail_when:
                lines.append(f"Fail when: {c.fail_when}")
        return "\n".join(lines) + "\n"


def _problems(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            fields = Criterion.model_fields if err["loc"][0] == "criteria" else Rubric.model_fields
            msg = f"unknown field; use {', '.join(fields)}"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)
