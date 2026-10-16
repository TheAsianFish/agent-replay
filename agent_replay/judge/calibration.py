"""Calibration reports and the gate that decides which judge criteria may fail a build.

A judge verdict is advisory until it is shown to agree with humans (PLAN.md
decision 5). A calibration report records, per rubric criterion, Cohen's kappa
between the judge and human labels with a confidence interval. A criterion
*gates* (its failure fails the run) only when a report exists for this exact
rubric (fingerprint) and judge (``judge_id``), covers the criterion, and the
lower bound of its kappa interval is at least ``min_kappa``
(``REPLAY_JUDGE_MIN_KAPPA``, default 0.6) over at least ``min_items`` labelled
items (``REPLAY_JUDGE_MIN_ITEMS``, default 20). Everything else is advisory,
with a reason that names what to do.

A report compared with objective labels rather than human ones, or not computed
with the ``core`` statistics (``baseline`` puts raw agreement in place of
kappa), never gates.

This module defines the report format and the gate; ``replay calibrate``
(:mod:`agent_replay.calibrate`) writes the reports.
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
from agent_replay.judge.verdict import PROMPT_VERSION

DEFAULT_MIN_KAPPA = 0.6
DEFAULT_MIN_ITEMS = 20
"""Below this many labelled items a bootstrap interval says little (two agreeing items give
the interval [1, 1]), so a criterion needs at least this ``n`` to gate."""


class Confusion(BaseModel):
    """Judge verdicts against the reference labels on the same items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    both_pass: int = Field(ge=0)
    judge_pass_reference_fail: int = Field(ge=0)
    """False passes: the costly error for a CI gate, a regression let through."""
    judge_fail_reference_pass: int = Field(ge=0)
    """False failures: a build broken for nothing."""
    both_fail: int = Field(ge=0)


class SelfConsistency(BaseModel):
    """How much the judge agrees with itself over repeated runs on the same items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(ge=2)
    n: int = Field(ge=0)
    """Items the judge answered on every run."""
    unanimous: float = Field(ge=0, le=1)
    """Fraction of those items with the same verdict on every run."""
    mean_pairwise_kappa: float | None = Field(default=None, ge=-1, le=1)
    """Kappa between each pair of runs, averaged; ``None`` when undefined for every pair."""


class CriterionCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kappa: float | None = Field(ge=-1, le=1)
    """``None`` when kappa is undefined (the labels leave no room for chance disagreement,
    e.g. every reference label is "pass")."""
    ci_low: float | None = Field(ge=-1, le=1)
    ci_high: float | None = Field(ge=-1, le=1)
    agreement: float = Field(ge=0, le=1)
    """Raw agreement: the fraction of items where the judge matched the human label."""
    n: int = Field(ge=0)
    # Schema 2 additions; absent from schema 1 reports.
    confusion: Confusion | None = None
    majority_label: Literal["pass", "fail"] | None = None
    """The reference's more common label (ties: "pass")."""
    majority_agreement: float | None = Field(default=None, ge=0, le=1)
    """Raw agreement of a judge that always answers ``majority_label``: the floor that raw
    agreement must beat to mean anything, and the reason kappa is the headline."""
    self_consistency: SelfConsistency | None = None


class FlipStats(BaseModel):
    """How often a perturbation that should not matter changed the judge's verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=0)
    flips: int = Field(ge=0)
    to_pass: int = Field(ge=0)
    """Flips from fail to pass under the perturbation."""
    to_fail: int = Field(ge=0)
    rate: float | None = Field(ge=0, le=1)
    ci_low: float | None = Field(default=None, ge=0, le=1)
    ci_high: float | None = Field(default=None, ge=0, le=1)


class GroupStats(BaseModel):
    """Judge leniency on one group of items, measured against the reference labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=0)
    judge_pass_rate: float | None = Field(ge=0, le=1)
    reference_pass_rate: float | None = Field(ge=0, le=1)
    false_pass_rate: float | None = Field(ge=0, le=1)
    """Of the items the reference failed, the fraction the judge passed."""
    reference_fail_n: int = Field(ge=0)


class ProbeResult(BaseModel):
    """One bias probe. ``flips`` is set for perturbation probes (position, verbosity),
    ``groups`` for the self-preference comparison; ``skipped`` says why neither is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe: Literal["position", "verbosity", "self_preference"]
    description: str
    flips: dict[str, FlipStats] | None = None
    """Per criterion (and ``_overall``)."""
    groups: dict[str, dict[str, GroupStats]] | None = None
    """Per criterion, per group (``same_model``, ``other_model``)."""
    skipped: str | None = None


class CalibrationReport(BaseModel):
    """How well one judge agrees with human labels on one rubric version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1, 2] = 2
    rubric: str
    rubric_fingerprint: str
    judge_id: str
    prompt_version: str
    """The judge prompt (``agent_replay.judge.verdict.PROMPT_VERSION``) the labels were
    compared against; a new prompt can change verdicts, so it needs a new report."""
    created_at: datetime
    n_items: int = Field(ge=0)
    method: str
    """How the numbers were computed, e.g. "Cohen's kappa, 95% bootstrap CI (2000 resamples)"."""
    criteria: dict[str, CriterionCalibration]
    # Schema 2 additions; absent from schema 1 reports (read as human labels, statistics unknown).
    reference: Literal["human", "objective"] = "human"
    """What the judge was compared with: human labels (may gate) or the suite's deterministic
    verdicts (informative only: agreeing with the rules does not show agreeing with people)."""
    labeller: str | None = None
    stats_impl: str | None = None
    """``REPLAY_STATS_IMPL`` when the report was made. A ``baseline`` report holds raw
    agreement in its kappa columns, so it never gates."""
    golden_set: str | None = None
    golden_fingerprint: str | None = None
    repeats: int = Field(default=1, ge=1)
    probes: tuple[ProbeResult, ...] = ()
    judge_calls: int | None = Field(default=None, ge=0)
    """Verdicts asked of the provider; the rest came from the cache."""
    cached_calls: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    """What this report's judge calls cost (cache hits are free); ``None`` if unpriced."""

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


def resolve_min_items(value: int | None = None) -> int:
    """``value``, else ``REPLAY_JUDGE_MIN_ITEMS``, else 20; must be at least 1."""
    source = "min_items"
    if value is None:
        raw = os.environ.get("REPLAY_JUDGE_MIN_ITEMS")
        if raw is None or not raw.strip():
            return DEFAULT_MIN_ITEMS
        source = "REPLAY_JUDGE_MIN_ITEMS"
        try:
            value = int(raw)
        except ValueError:
            raise ConfigError(
                f"REPLAY_JUDGE_MIN_ITEMS={raw!r} is not a whole number; use a count such as 20"
            ) from None
    if value < 1:
        raise ConfigError(f"{source}={value!r} is out of range; use a count of at least 1")
    return value


def report_blocker(report: CalibrationReport) -> str | None:
    """Why no criterion of ``report`` may gate whatever its numbers, or ``None``.

    Only kappa computed by the ``core`` statistics may gate: a ``baseline`` report holds
    raw agreement in its kappa columns, and a report that does not say (schema 1, or
    edited by hand) cannot be trusted to hold kappa either.
    """
    if report.reference != "human":
        return (
            f"the calibration report compares the judge with {report.reference} labels, not "
            "human ones; label the golden set with `replay label` and recalibrate"
        )
    if report.stats_impl != "core":
        what = (
            "REPLAY_STATS_IMPL=baseline (raw agreement, not kappa)"
            if report.stats_impl == "baseline"
            else f"statistics {report.stats_impl or 'that it does not name'}"
        )
        return (
            f"the calibration report was computed with {what}; recalibrate with "
            "REPLAY_STATS_IMPL=core"
        )
    return None


def _criterion_gate(
    calibration: CriterionCalibration, min_kappa: float, min_items: int
) -> CriterionGate:
    if calibration.ci_low is None:
        return CriterionGate(
            gating=False,
            reason=(
                f"kappa is undefined (n={calibration.n}): the labels leave no room for "
                "disagreement by chance; label runs with both outcomes and recalibrate"
            ),
        )
    if calibration.n < min_items:
        return CriterionGate(
            gating=False,
            reason=(
                f"only {calibration.n} labelled items, fewer than {min_items} "
                "(REPLAY_JUDGE_MIN_ITEMS); label more runs and recalibrate"
            ),
        )
    if calibration.ci_low >= min_kappa:
        return CriterionGate(
            gating=True,
            reason=(
                f"calibrated: kappa lower bound {calibration.ci_low:.2f} >= {min_kappa:.2f} "
                f"(n={calibration.n})"
            ),
        )
    return CriterionGate(
        gating=False,
        reason=(
            f"kappa lower bound {calibration.ci_low:.2f} < {min_kappa:.2f} "
            f"(n={calibration.n}); sharpen the criterion or label more runs, then "
            "recalibrate"
        ),
    )


def criterion_gates(
    report: CalibrationReport, min_kappa: float, *, min_items: int | None = None
) -> dict[str, bool]:
    """Per criterion in the report, whether it would gate under ``min_kappa`` (rubric, judge
    and prompt matching aside): what the dashboard shows. The whole-run entry
    (``_overall``) is not a rubric criterion and never gates."""
    if report_blocker(report) is not None:
        return dict.fromkeys(report.criteria, False)
    floor = resolve_min_items(min_items)
    return {
        name: not name.startswith("_") and _criterion_gate(c, min_kappa, floor).gating
        for name, c in report.criteria.items()
    }


def gating(
    report: CalibrationReport | None,
    rubric: Rubric,
    judge_id: str,
    min_kappa: float,
    *,
    prompt_version: str = PROMPT_VERSION,
    min_items: int | None = None,
) -> dict[str, CriterionGate]:
    """For each rubric criterion, whether its failure may fail the run, and why.

    A report counts only for the rubric fingerprint, judge id and judge prompt
    version it measured; any other combination is uncalibrated. A criterion also
    needs ``min_items`` labelled items (default ``REPLAY_JUDGE_MIN_ITEMS``, else 20).
    """

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
    blocked = report_blocker(report)
    if blocked is not None:
        return everywhere(blocked)
    if report.prompt_version != prompt_version:
        return everywhere(
            f"the calibration report was made with judge prompt version "
            f"{report.prompt_version}, not {prompt_version}; recalibrate after the prompt change"
        )
    floor = resolve_min_items(min_items)
    gates: dict[str, CriterionGate] = {}
    for criterion_id in rubric.ids:
        calibration = report.criteria.get(criterion_id)
        gates[criterion_id] = (
            CriterionGate(
                gating=False,
                reason="not in the calibration report; label it and recalibrate",
            )
            if calibration is None
            else _criterion_gate(calibration, min_kappa, floor)
        )
    return gates
