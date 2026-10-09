"""Calibration reports and the gate that decides which judge criteria may fail a build.

A judge verdict is advisory until it is shown to agree with humans (PLAN.md
decision 5). A calibration report records, per rubric criterion, Cohen's kappa
between the judge and human labels with a confidence interval. A criterion
*gates* (its failure fails the run) only when a report exists for this exact
rubric (fingerprint) and judge (``judge_id``), covers the criterion, and the
lower bound of its kappa interval is at least ``min_kappa``
(``REPLAY_JUDGE_MIN_KAPPA``, default 0.6). Everything else is advisory, with a
reason that names what to do.

This module defines the report format and the gate; the report itself is
written by the calibration command.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_replay.errors import ConfigError
from agent_replay.judge.rubric import Rubric

DEFAULT_MIN_KAPPA = 0.6


class CriterionCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kappa: float = Field(ge=-1, le=1)
    ci_low: float = Field(ge=-1, le=1)
    ci_high: float = Field(ge=-1, le=1)
    agreement: float = Field(ge=0, le=1)
    """Raw agreement: the fraction of items where the judge matched the human label."""
    n: int = Field(ge=0)


class CalibrationReport(BaseModel):
    """How well one judge agrees with human labels on one rubric version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    rubric: str
    rubric_fingerprint: str
    judge_id: str
    created_at: datetime
    n_items: int = Field(ge=0)
    method: str
    """How the numbers were computed, e.g. "Cohen's kappa, 95% bootstrap CI (2000 resamples)"."""
    criteria: dict[str, CriterionCalibration]

    @classmethod
    def load(cls, path: str | Path) -> CalibrationReport:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(
                f"calibration report {str(path)!r} does not exist; check the path, or drop the "
                "calibration setting to run the judge as advisory"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"cannot read calibration report {str(path)!r}: {exc}") from exc
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ConfigError(
                f"calibration report {str(path)!r} is invalid: {problems}. Regenerate it with "
                "the calibration command"
            ) from exc


@dataclass(frozen=True)
class CriterionGate:
    gating: bool
    """True when this criterion's failure fails the run."""
    reason: str


def resolve_min_kappa(value: float | None) -> float:
    """``value``, else ``REPLAY_JUDGE_MIN_KAPPA``, else 0.6; must lie in 0..1."""
    source = "min_kappa"
    if value is None:
        raw = os.environ.get("REPLAY_JUDGE_MIN_KAPPA")
        if raw is None or not raw.strip():
            return DEFAULT_MIN_KAPPA
        source = "REPLAY_JUDGE_MIN_KAPPA"
        try:
            value = float(raw)
        except ValueError:
            raise ConfigError(
                f"REPLAY_JUDGE_MIN_KAPPA={raw!r} is not a number; use a value from 0 to 1"
            ) from None
    if not (math.isfinite(value) and 0 <= value <= 1):
        raise ConfigError(f"{source}={value!r} is out of range; use a value from 0 to 1")
    return value


def gating(
    report: CalibrationReport | None, rubric: Rubric, judge_id: str, min_kappa: float
) -> dict[str, CriterionGate]:
    """For each rubric criterion, whether its failure may fail the run, and why."""

    def everywhere(reason: str) -> dict[str, CriterionGate]:
        return {i: CriterionGate(gating=False, reason=reason) for i in rubric.ids}

    if report is None:
        return everywhere("uncalibrated: calibrate the judge against human labels first")
    if report.rubric_fingerprint != rubric.fingerprint:
        return everywhere(
            f"the calibration report is for another version of rubric {report.rubric!r}; "
            "recalibrate after editing the rubric"
        )
    if report.judge_id != judge_id:
        return everywhere(
            f"the calibration report is for judge {report.judge_id}, not {judge_id}; "
            "recalibrate with this judge or switch back to that one"
        )
    gates: dict[str, CriterionGate] = {}
    for criterion_id in rubric.ids:
        calibration = report.criteria.get(criterion_id)
        if calibration is None:
            gates[criterion_id] = CriterionGate(
                gating=False,
                reason="not in the calibration report; label it and recalibrate",
            )
        elif calibration.ci_low >= min_kappa:
            gates[criterion_id] = CriterionGate(
                gating=True,
                reason=(
                    f"calibrated: kappa lower bound {calibration.ci_low:.2f} >= {min_kappa:.2f} "
                    f"(n={calibration.n})"
                ),
            )
        else:
            gates[criterion_id] = CriterionGate(
                gating=False,
                reason=(
                    f"kappa lower bound {calibration.ci_low:.2f} < {min_kappa:.2f} "
                    f"(n={calibration.n}); sharpen the criterion or label more runs, then "
                    "recalibrate"
                ),
            )
    return gates
