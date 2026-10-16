"""The ``judge`` check: an LLM grades the run against a rubric, advisory until calibrated."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import JsonValue

from agent_replay.checks.base import CheckResult
from agent_replay.errors import ConfigError, JudgeCacheMiss, JudgeError
from agent_replay.judge.cache import JudgeCache, env_flag
from agent_replay.judge.calibration import (
    CalibrationReport,
    CriterionGate,
    gating,
    resolve_min_kappa,
)
from agent_replay.judge.providers import JudgeProvider, make_provider
from agent_replay.judge.rubric import Rubric
from agent_replay.judge.transcript import DEFAULT_MAX_CHARS, JudgeInput
from agent_replay.judge.verdict import JudgeResponse

if TYPE_CHECKING:
    from agent_replay.runner import RunResult

log = structlog.get_logger("agent_replay")

_REASON_CHARS = 160


_ADVISORY_BECAUSE = {
    "kappa lower bound": "calibrated below the kappa threshold",
    "kappa is undefined": "kappa undefined on the labels",
    "only ": "too few labelled items",
    "the calibration report": "not calibrated for this rubric, judge and prompt version",
}


def _gate_summary(gates: dict[str, CriterionGate], *, report: CalibrationReport | None) -> str:
    gated = [i for i, g in gates.items() if g.gating]
    advisory = [i for i, g in gates.items() if not g.gating]
    if gated:
        return (
            f"gating: {', '.join(gated)}; advisory: {', '.join(advisory)}"
            if advisory
            else "calibrated"
        )
    if report is None:
        why = "uncalibrated"
    elif report.reference != "human":
        why = f"calibrated against {report.reference} labels, not human ones"
    elif report.stats_impl != "core":
        why = "calibrated without the core statistics (no kappa to trust)"
    else:
        why = next(
            (
                summary
                for prefix, summary in _ADVISORY_BECAUSE.items()
                if all(g.reason.startswith(prefix) for g in gates.values())
            ),
            "no criterion meets the gate (see each criterion's reason)",
        )
    return f"advisory: {why}"


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _REASON_CHARS else text[: _REASON_CHARS - 1] + "…"


@dataclass(frozen=True)
class Judge:
    """Grade the run against ``rubric`` with ``provider``; see :func:`judge`.

    The check passes when every criterion passes. A failed criterion fails the
    run only if it is *gating* (calibrated, see
    :func:`agent_replay.judge.calibration.gating`); otherwise the result is
    advisory: reported, never blocking. When no verdict can be had (provider
    down, bad answer, offline cache miss) the check fails; that failure blocks
    only if some criterion is gating, so an uncalibrated judge's outage never
    breaks a build while a calibrated one fails closed.
    """

    rubric: Rubric
    provider: JudgeProvider = field(compare=False)
    cache: JudgeCache | None = field(default=None, compare=False)
    calibration: CalibrationReport | None = None
    min_kappa: float = 0.6
    max_chars: int = DEFAULT_MAX_CHARS

    @property
    def name(self) -> str:
        return f"judge:{self.rubric.name}"

    def _verdict(self, judge_input: JudgeInput) -> tuple[JudgeResponse, bool]:
        if self.cache is not None:
            return self.cache.verdict(self.provider, self.rubric, judge_input)
        return self.provider.evaluate(self.rubric, judge_input), False

    def __call__(self, run: RunResult, /) -> CheckResult:
        judge_id = self.provider.judge_id
        gates = gating(self.calibration, self.rubric, judge_id, self.min_kappa)
        any_gating = any(g.gating for g in gates.values())
        summary = _gate_summary(gates, report=self.calibration)
        details: dict[str, JsonValue] = {
            "judge_id": judge_id,
            "rubric": self.rubric.name,
            "rubric_fingerprint": self.rubric.fingerprint,
            "calibrated": any_gating,
        }
        judge_input = JudgeInput.from_run(run, max_chars=self.max_chars)
        try:
            response, cached = self._verdict(judge_input)
        except JudgeCacheMiss:
            raise  # the suite's committed cache is stale: a broken suite, not a verdict
        except Exception as exc:  # a custom provider's bug must not crash the suite either
            reason = str(exc) if isinstance(exc, JudgeError) else f"{type(exc).__name__}: {exc}"
            log.warning("judge.error", check=self.name, error=reason)
            details["error"] = reason
            details["criteria"] = {
                i: {"verdict": None, "gating": g.gating, "gate_reason": g.reason}
                for i, g in gates.items()
            }
            consequence = (
                "failing closed: the judge is calibrated"
                if any_gating
                else f"{summary}, so this does not fail the run"
            )
            return CheckResult(
                check=self.name,
                passed=False,
                advisory=not any_gating,
                message=f"judge verdict unknown ({consequence}): {reason}",
                details=details,
            )

        failed = [v for v in response.verdicts if not v.passed]
        details.update(
            {
                "cached": cached,
                "cost_usd": 0.0 if cached else response.cost_usd,
                "original_cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "criteria": {
                    v.id: {
                        "verdict": v.verdict,
                        "reasoning": v.reasoning,
                        "gating": gates[v.id].gating,
                        "gate_reason": gates[v.id].reason,
                    }
                    for v in response.verdicts
                },
            }
        )
        if response.usage is not None:
            details["usage"] = response.usage.model_dump(mode="json")
        total = len(response.verdicts)
        message = f"judge: {total - len(failed)}/{total} criteria passed ({summary})"
        mixed = any_gating and not all(g.gating for g in gates.values())
        if failed:
            message += "; failed: " + "; ".join(
                f"{v.id}{' (advisory)' if mixed and not gates[v.id].gating else ''}"
                f" — {_short(v.reasoning)}"
                for v in failed
            )
        return CheckResult(
            check=self.name,
            passed=not failed,
            # A pass is advisory when nothing gates; a failure when no failed criterion gates.
            advisory=not any(gates[v.id].gating for v in failed) if failed else not any_gating,
            message=message,
            details=details,
        )


def judge(
    rubric: Rubric | str | Path,
    *,
    provider: JudgeProvider | None = None,
    cache: JudgeCache | Literal["default"] | None = "default",
    calibration: CalibrationReport | str | Path | None = None,
    min_kappa: float | None = None,
    offline: bool | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Judge:
    """Build a ``judge`` check. See :class:`Judge`.

    ``rubric`` is a :class:`Rubric` or a YAML file. ``provider`` defaults to
    :func:`make_provider` (``REPLAY_JUDGE_PROVIDER``, else Anthropic; no API key
    is needed until a verdict is missing from the cache). ``cache`` is a
    :class:`JudgeCache`, ``"default"`` (``REPLAY_JUDGE_CACHE`` or
    ``.replay/judge-cache``) or ``None``. ``offline`` (default
    ``REPLAY_JUDGE_OFFLINE``) applies to the default cache and judges from it
    only; a cache you pass keeps its own setting. ``calibration`` is a report or its JSON
    file; without one every criterion is advisory. ``max_chars`` caps each value
    in the transcript.
    """
    loaded = rubric if isinstance(rubric, Rubric) else Rubric.load(rubric)
    if max_chars < 1:
        raise ConfigError(f"max_chars must be >= 1, got {max_chars}")
    if cache == "default":
        cache = JudgeCache(offline=offline)
    elif cache is None and (offline if offline is not None else env_flag("REPLAY_JUDGE_OFFLINE")):
        raise ConfigError(
            "offline judging (offline=True or REPLAY_JUDGE_OFFLINE) reads verdicts from the "
            "cache, but the cache is disabled; pass a cache or turn offline off"
        )
    report = (
        calibration
        if calibration is None or isinstance(calibration, CalibrationReport)
        else CalibrationReport.load(calibration)
    )
    return Judge(
        rubric=loaded,
        provider=provider if provider is not None else make_provider(),
        cache=cache,
        calibration=report,
        min_kappa=resolve_min_kappa(min_kappa),
        max_chars=max_chars,
    )
