"""Build golden items and objective labels by replaying a suite.

Each scenario of a ``replay.toml`` is replayed (with an optional agent, mode or
client override, so a seeded mutant contributes runs the checks should fail) and
becomes one :class:`~agent_replay.calibrate.golden.GoldenItem`: the transcript the
judge would see. Its objective label is the suite's deterministic verdict, the
same definition as ``eval/judge_smoke.py``: the agent finished and every
non-judge check passed.

The suite's judge checks are dropped before the replay. The golden set is what
the judge is measured against, so building it must neither need the judge (a
key, a verdict cache) nor be influenced by it; the transcript is built with the
judge check's own ``max_chars`` so items match what that check sends.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from agent_replay.calibrate.golden import (
    OVERALL,
    GoldenItem,
    GoldenSource,
    Label,
    item_id,
    portable,
)
from agent_replay.cassette import Cassette
from agent_replay.checks.base import CheckResult
from agent_replay.ci import (
    Overrides,
    Scenario,
    check_identify_tools,
    load_baselines,
    load_suite,
    run_scenario,
)
from agent_replay.errors import ConfigError
from agent_replay.judge.transcript import DEFAULT_MAX_CHARS, JudgeInput
from agent_replay.runner import RunResult
from agent_replay.session import ReplayMode

log = structlog.get_logger("agent_replay")

OBJECTIVE_LABELLER = "checks"
"""The labeller name on objective labels: the suite's deterministic checks."""

DEFAULT_VARIANT = "agent"


@dataclass(frozen=True)
class SkippedScenario:
    """A scenario that produced no item: it could not run at all."""

    name: str
    reason: str


@dataclass(frozen=True)
class GoldenBuild:
    """What one suite replay contributes to a golden set.

    ``items`` are in suite order, one per distinct transcript (scenarios whose
    transcripts coincide share an item with several sources). ``labels`` hold
    one objective label per item. ``conflicts`` are transcripts whose runs got
    different deterministic verdicts, which no single label can describe: they
    are left out of ``items`` and ``labels`` both, so a set never holds an item
    whose sources disagree with its label.
    """

    items: tuple[GoldenItem, ...]
    labels: tuple[Label, ...]
    skipped: tuple[SkippedScenario, ...] = ()
    conflicts: tuple[str, ...] = ()


def golden_from_suite(
    suite: Path,
    *,
    variant: str = DEFAULT_VARIANT,
    agent: str | None = None,
    mode: ReplayMode | None = None,
    client: str | None = None,
    clock: Callable[[], datetime] | None = None,
    timer: Callable[[], float] = time.perf_counter,
    scenarios: Sequence[Scenario] | None = None,
) -> GoldenBuild:
    """Replay every scenario of ``suite`` without its judge checks and build the items.

    ``scenarios`` replaces the suite's own list (same defaults and checks), for
    example the suite's scenarios pointed at new recordings of the same tasks.

    ``labelled_at`` on an objective label is the recording time of the scenario's
    cassette unless ``clock`` is given, so rebuilding from the same cassettes
    writes byte-identical labels. Raises :class:`ConfigError` for a broken suite
    file or baseline; a scenario that cannot run is skipped and reported.
    """
    if not variant.strip():
        raise ConfigError("the variant name is empty; pass --variant NAME (e.g. the mutant's name)")
    loaded = load_suite(suite)
    overrides = Overrides(agent=agent, mode=mode, client=client)
    base_dir = suite.parent
    try:
        check_identify_tools(loaded, overrides, base_dir)
    except ConfigError as exc:
        raise ConfigError(f"{suite}: {exc}") from exc
    baselines = load_baselines(loaded, overrides, base_dir)
    defaults = loaded.defaults.model_copy(update={"checks": _without_judge(loaded.defaults.checks)})
    items: dict[str, GoldenItem] = {}
    verdicts: dict[str, Label] = {}
    conflicts: list[str] = []
    skipped: list[SkippedScenario] = []
    for scenario in loaded.scenario if scenarios is None else scenarios:
        max_chars = _max_chars(scenario, loaded.defaults.checks)
        stripped = scenario.model_copy(update={"checks": _without_judge(scenario.checks)})
        report, result = run_scenario(
            stripped, defaults, overrides, base_dir, timer, baselines=baselines
        )
        if result is None:
            reason = f"{report.error.type}: {report.error.message}" if report.error else "?"
            skipped.append(SkippedScenario(scenario.name, reason))
            log.warning("golden.skipped", scenario=scenario.name, reason=reason)
            continue
        cassette_path = base_dir / scenario.cassette
        transcript = JudgeInput.from_run(result, max_chars=max_chars)
        source = GoldenSource(
            suite=portable(suite),  # the same sources from a relative or absolute suite path
            scenario=scenario.name,
            variant=variant,
            cassette=portable(cassette_path),
            agent_model=agent_model(result),
        )
        key = item_id(transcript)
        old = items.get(key)
        if old is None:
            items[key] = GoldenItem.from_transcript(transcript, source)
        elif source not in old.sources:
            items[key] = old.model_copy(update={"sources": (*old.sources, source)})
        labelled_at = (
            clock() if clock is not None else Cassette.load(cassette_path).header.created_at
        )
        label = objective_label(key, result, report.checks, labelled_at)
        previous = verdicts.get(key)
        if previous is None:
            verdicts[key] = label
        elif previous.label != label.label and key not in conflicts:
            conflicts.append(key)
        log.info("golden.scenario", scenario=scenario.name, item=key, label=label.label)
    kept = tuple(item for key, item in items.items() if key not in conflicts)
    labels = tuple(lb for key, lb in verdicts.items() if key not in conflicts)
    return GoldenBuild(kept, labels, tuple(skipped), tuple(conflicts))


NOT_OBJECTIVE = frozenset({"budget"})
"""Check types left out of the objective label. A ``budget`` check compares cost, tokens,
latency or the number of model calls with a baseline: nothing a transcript shows, so a
correct run that happened to be dearer must not count as a failed one."""


def objective_label(
    item: str, result: RunResult, checks: tuple[CheckResult, ...], labelled_at: datetime
) -> Label:
    """Pass iff the run finished and every check of the run's behaviour passed, advisory
    ones included: the judge is measured on the run, not on what the suite chose to gate.
    ``checks`` exclude the judge (:func:`golden_from_suite` drops it); the types in
    :data:`NOT_OBJECTIVE` are skipped here."""
    failed = [c.check for c in checks if not c.passed and c.check not in NOT_OBJECTIVE]
    problems = ([f"agent error {result.error.type}"] if result.error is not None else []) + failed
    return Label(
        item=item,
        criterion=OVERALL,
        label="fail" if problems else "pass",
        source="objective",
        labeller=OBJECTIVE_LABELLER,
        labelled_at=labelled_at,
        note=("failed: " + ", ".join(problems)) if problems else None,
    )


def agent_model(result: RunResult) -> str | None:
    """The model the run's model calls asked for, when they all asked for the same one."""
    models = {
        step.request["model"]
        for step in result.trajectory.model_steps()
        if isinstance(step.request, dict) and isinstance(step.request.get("model"), str)
    }
    return str(next(iter(models))) if len(models) == 1 else None


def _is_judge(spec: dict[str, Any]) -> bool:
    return spec.get("type") == "judge"


def _without_judge(specs: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(s for s in specs if not _is_judge(s))


def _max_chars(scenario: Scenario, defaults: tuple[dict[str, Any], ...]) -> int:
    """The ``max_chars`` of the scenario's last judge check (the default without one)."""
    judges = [s for s in (*defaults, *scenario.checks) if _is_judge(s)]
    if not judges:
        return DEFAULT_MAX_CHARS
    value = judges[-1].get("max_chars", DEFAULT_MAX_CHARS)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(
            f"scenario {scenario.name!r}: the judge check's max_chars must be a positive "
            f"integer, got {value!r}"
        )
    return value
