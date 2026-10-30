"""Which regressions does replay catch, with which check, at which cut point? (PLAN.md D4)

Replays the retail suite (``examples/retail_agent/replay.toml``, six τ-bench
tasks with their committed fake-model cassettes) against every mutation and
benign refactor in ``eval/mutations/``, in seven configurations:

- ``full``: every model and tool call served from the cassette (strict). This
  is the stub-everything baseline: it tests the agent's own code only.
- ``tools-live``: tools run for real against recorded model replies (strict).
- ``model-live``: the model runs (the fake model, or the mutated one for
  ``model`` mutations) against recorded tools (strict: a tool call the
  cassette has no envelope for ends the run).
- ``model-live+fallback``: the same, but a tool call with no recorded envelope
  left for that tool runs the real tool instead of ending the run. (A call
  whose arguments merely differ from the recording is still served the
  recorded result, flagged ``approximate``.)
- ``tools-live+strict``: ``tools-live`` with ``fail_on_approximate``, so a served
  model step whose request differs from the recording fails the scenario
  (``no_approximate_steps``). This is the drift signal the other configurations
  only record; it catches changed prompts and changed tool outputs at the cost
  of false alarms on refactors that change request bytes.
- ``tools-live+drift``: ``tools-live`` with ``fail_on_tool_drift``, so a live
  tool whose result differs from the one recorded for the same arguments fails
  the scenario (``no_tool_drift``, :mod:`agent_replay.drift`). It looks at tool
  results only, so a refactor that changes request bytes does not trip it.
- ``model-live+synth`` (M17): ``model-live``, but a tool call with no recording left
  for it is answered with a schema-minimal stub of the tool's return type
  (:mod:`agent_replay.synth`: an empty record, an empty string, ``0.0``) instead of
  ending the run or running the tool. Listed last so the six earlier columns keep
  their place.

Definitions:

- A mutation is *detected* in a configuration when the suite fails there (any
  scenario not passed). A scenario that did not run at all is a harness bug:
  the study exits 2 and names it.
- A check type *catches* a mutation in a configuration when it is among the
  blocking failed checks of at least one scenario. The check type is
  ``CheckResult.check`` up to its first ``:`` (``arg_schema:cancel_pending_order``
  counts as ``arg_schema``), except that rules keep their name
  (``rule:matches_ground_truth``). A run that ends in an exception counts as
  ``run_error:<ExceptionType>``. A check type catches a mutation *alone* when no
  other check type caught it in that configuration.
- A benign refactor raises a *false alarm* in a configuration when the suite fails.

These tables run the suite without its identifying arguments (``[defaults.identify]``),
as v1 did. The ``identify`` section (M11) runs every variant again with them, plus two
mutations only they can see (``guesses_user_id``, ``reads_wrong_order``: a read of an
entity the recording never saw, served the right recording without the keys), and
lists every v1 cell the keys changed. The ``path_variants`` section (M11) replays five
correct-but-different paths of the fake model (``eval/mutations/paths.py``) in
``model-live`` under ``strict`` and ``fallback``, with and without the keys, and
records per scenario which checks failed and how the run ended; M17 appends the same
rows under ``synth``.

Everything is offline and deterministic; the output has no timestamps or
absolute paths, so re-running it reproduces the JSON byte for byte.

Usage::

    uv run python eval/mutation_study.py --out eval/results/mutations.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from agent_replay._json import canonical_json
from agent_replay._logging import configure_logging
from agent_replay.cassette import Cassette, Kind
from agent_replay.ci import Overrides, RunSink, ScenarioReport, SuiteReport, load_suite, run_suite
from agent_replay.runner import RunResult
from agent_replay.session import Policy, ReplayMode
from eval.mutations import (
    IDENTITY_MUTATIONS,
    MUTATIONS,
    REAL_AGENT,
    REFACTORS,
    Mutation,
    Refactor,
    Target,
)
from eval.mutations.paths import PATH_VARIANTS, PathVariant
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "retail_agent" / "replay.toml"
TARGETS: tuple[Target, ...] = ("agent", "model", "tool")


@dataclass(frozen=True)
class Config:
    id: str
    mode: ReplayMode
    policy: Policy
    note: str
    strict: bool = False
    """``fail_on_approximate``: a served step whose request differs from the recording fails."""
    drift: bool = False
    """``fail_on_tool_drift``: a live tool result that differs from the recording fails."""

    @property
    def model_live(self) -> bool:
        return self.mode == "model-live"


CONFIGS: tuple[Config, ...] = (
    Config("full", "full", "strict", "everything served: the stub-everything baseline"),
    Config("tools-live", "tools-live", "strict", "tools run, model replies served"),
    Config("model-live", "model-live", "strict", "model runs, tools served"),
    Config(
        "model-live+fallback",
        "model-live",
        "fallback",
        "model runs, tools served; unrecorded tool calls run for real",
    ),
    Config(
        "tools-live+strict",
        "tools-live",
        "strict",
        "tools run, model replies served; a served step whose request differs from the "
        "recording fails (fail_on_approximate)",
        strict=True,
    ),
    Config(
        "tools-live+drift",
        "tools-live",
        "strict",
        "tools run, model replies served; a live tool result that differs from the one "
        "recorded for the same arguments fails (fail_on_tool_drift)",
        drift=True,
    ),
    Config(
        "model-live+synth",
        "model-live",
        "synth",
        "model runs, tools served; unrecorded tool calls are answered with a "
        "schema-minimal stub of the tool's return type",
    ),
)


class HarnessError(RuntimeError):
    """A scenario did not run: the study's own setup is broken, not the variant."""


@dataclass(frozen=True)
class Outcome:
    """One variant's suite run in one configuration."""

    passed: bool
    caught_by: tuple[str, ...]
    failed_scenarios: tuple[str, ...]
    approximate_steps: int

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "passed": self.passed,
            "caught_by": list(self.caught_by),
            "failed_scenarios": list(self.failed_scenarios),
            "approximate_steps": self.approximate_steps,
        }


def check_type(name: str) -> str:
    """``arg_schema:cancel_pending_order`` -> ``arg_schema``; rules keep their name."""
    return name if name.startswith("rule:") else name.split(":", 1)[0]


def suite_check_types() -> list[str]:
    """Every check type the suite configures, so a type that caught nothing still gets a row."""
    suite = load_suite(SUITE)
    specs = [*suite.defaults.checks, *(c for s in suite.scenario for c in s.checks)]
    types = {
        f"rule:{str(spec['ref']).rsplit(':', 1)[-1]}" if spec["type"] == "rule" else spec["type"]
        for spec in specs
    }
    return sorted(str(t) for t in types)


def scenario_catches(scenario: ScenarioReport) -> set[str]:
    types = {check_type(c.check) for c in scenario.failed_checks}
    if scenario.error is not None:
        types.add(f"run_error:{scenario.error.type}")
    return types


def outcome(report: SuiteReport) -> Outcome:
    caught: set[str] = set()
    for scenario in report.scenarios:
        caught |= scenario_catches(scenario)
    return Outcome(
        passed=report.passed,
        caught_by=tuple(sorted(caught)),
        failed_scenarios=tuple(s.name for s in report.failed),
        approximate_steps=sum(s.steps.by_status.get("approximate", 0) for s in report.scenarios),
    )


def replay_suite(agent: str, client: str, config: Config, *, identify: bool = False) -> Outcome:
    """Run the suite once with ``agent`` in ``config``; the database starts fresh.

    ``identify``: with the suite's identifying arguments. Off, the suite runs as it did
    before it had them (the v1 tables)."""
    report = _run(
        Overrides(
            agent=agent,
            mode=config.mode,
            policy=config.policy,
            client=client if config.model_live else None,
            fail_on_approximate=True if config.strict else None,
            fail_on_tool_drift=True if config.drift else None,
            identify=None if identify else {},
        )
    )
    return outcome(report)


def _run(overrides: Overrides, sink: RunSink | None = None) -> SuiteReport:
    retail_tools.reset_db()
    try:
        report = run_suite(SUITE, overrides=overrides, sink=sink)
    finally:
        retail_tools.reset_db()
    if report.broken:
        problems = "; ".join(
            f"{s.name} ({s.error.type}: {s.error.message})" if s.error else s.name
            for s in report.broken
        )
        raise HarnessError(
            f"{overrides.agent} in {overrides.mode}/{overrides.policy}: scenario(s) did not "
            f"run: {problems}. Fix the catalogue entry (agent or client reference) or the suite"
        )
    return report


def _rows(
    variants: Sequence[Mutation | Refactor], configs: Sequence[Config], *, identify: bool
) -> list[JsonValue]:
    rows: list[JsonValue] = []
    for v in variants:
        results = {c.id: replay_suite(v.agent, v.client, c, identify=identify) for c in configs}
        row: dict[str, JsonValue] = {"id": v.id}
        if isinstance(v, Mutation):
            row["target"] = v.target
        row |= {
            "description": v.description,
            "agent": v.agent,
            "client": v.client,
            "results": {k: o.to_json() for k, o in results.items()},
        }
        rows.append(row)
    return rows


def run(
    mutations: Sequence[Mutation] = MUTATIONS,
    refactors: Sequence[Refactor] = REFACTORS,
    configs: Sequence[Config] = CONFIGS,
    identity_mutations: Sequence[Mutation] = IDENTITY_MUTATIONS,
    path_variants: Sequence[PathVariant] = PATH_VARIANTS,
) -> dict[str, JsonValue]:
    """Replay every variant in every configuration; the study's results as JSON data.

    The top-level tables run the suite without its identifying arguments, as v1 did;
    ``identify`` re-runs everything with them, plus the mutations only they can see;
    ``path_variants`` replays correct-but-different paths of the fake model."""
    mutation_rows = _rows(mutations, configs, identify=False)
    refactor_rows = _rows(refactors, configs, identify=False)
    config_ids = [c.id for c in configs]
    return {
        "suite": SUITE.relative_to(ROOT).as_posix(),
        "definitions": _definitions(),
        "configurations": [
            {
                "id": c.id,
                "mode": c.mode,
                "policy": c.policy,
                "fail_on_approximate": c.strict,
                "fail_on_tool_drift": c.drift,
                "note": c.note,
            }
            for c in configs
        ],
        "mutations": mutation_rows,
        "refactors": refactor_rows,
        "summary": summarise(mutation_rows, refactor_rows, config_ids, suite_check_types()),
        "identify": identify_study(
            mutation_rows, refactor_rows, mutations, refactors, identity_mutations, configs
        ),
        "path_variants": path_study(path_variants),
    }


# --- identifying arguments (M11) ----------------------------------------------------


def suite_identify() -> dict[str, JsonValue]:
    """The suite's ``[defaults.identify]`` table."""
    return {
        tool: list(paths) for tool, paths in sorted(load_suite(SUITE).defaults.identify.items())
    }


def identify_study(
    mutation_rows: Sequence[JsonValue],
    refactor_rows: Sequence[JsonValue],
    mutations: Sequence[Mutation],
    refactors: Sequence[Refactor],
    identity_mutations: Sequence[Mutation],
    configs: Sequence[Config],
) -> dict[str, JsonValue]:
    """Every variant again with the suite's identifying arguments, the identity mutations
    without them, and every v1 cell the keys changed."""
    config_ids = [c.id for c in configs]
    keyed_mutations = _rows([*mutations, *identity_mutations], configs, identify=True)
    keyed_refactors = _rows(refactors, configs, identify=True)
    without = _rows(identity_mutations, configs, identify=False)
    keyed = {_field(row, "id"): row for row in [*keyed_mutations, *keyed_refactors]}
    changed: list[JsonValue] = []
    for before in [*mutation_rows, *refactor_rows]:
        after = keyed[_field(before, "id")]
        for config in config_ids:
            if _results(before, config) != _results(after, config):
                changed.append(
                    {
                        "id": _field(before, "id"),
                        "config": config,
                        "without_keys": _results(before, config),
                        "with_keys": _results(after, config),
                    }
                )
    return {
        "keys": suite_identify(),
        "mutations": keyed_mutations,
        "refactors": keyed_refactors,
        "summary": summarise(keyed_mutations, keyed_refactors, config_ids, suite_check_types()),
        "identity_mutations": [m.id for m in identity_mutations],
        "identity_mutations_without_keys": without,
        "changed_by_keys": changed,
    }


# --- correct-but-different paths (M11) ----------------------------------------------

PATH_POLICIES: tuple[Policy, ...] = ("strict", "fallback", "synth")
PATH_POLICY_GROUPS: tuple[tuple[Policy, ...], ...] = (("strict", "fallback"), ("synth",))
"""Rows are listed group by group, so M11's ``strict`` and ``fallback`` rows keep their
place and M17's ``synth`` rows follow them."""


def termination(scenario: ScenarioReport) -> str:
    """How a scenario ended: ``passed``, ``unrecorded`` (an unrecorded call ended the run),
    ``run_error:<Type>`` or ``checks`` (it finished and a check failed)."""
    if scenario.passed:
        return "passed"
    if scenario.error is not None:
        if scenario.error.type == "UnrecordedCallError":
            return "unrecorded"
        return f"run_error:{scenario.error.type}"
    return "checks"


def _tool_calls(steps: Sequence[tuple[str, JsonValue]]) -> list[str]:
    return [f"{name} {canonical_json(request)}" for name, request in steps]


def path_row(variant: PathVariant, policy: Policy, *, identify: bool) -> dict[str, JsonValue]:
    """One variant in ``model-live`` under ``policy``, scenario by scenario."""
    results: dict[str, RunResult] = {}

    def save(report: ScenarioReport, result: RunResult | None, _: object) -> None:
        if result is not None:
            results[report.name] = result

    report = _run(
        Overrides(
            agent=REAL_AGENT,
            mode="model-live",
            policy=policy,
            client=variant.client,
            identify=None if identify else {},
        ),
        RunSink(save=save),
    )
    configured = suite_check_types()
    scenarios: list[JsonValue] = []
    for s in report.scenarios:
        recorded = Cassette.load(Path(s.cassette)).in_start_order()
        steps = results[s.name].trajectory.tool_steps()
        affected = _tool_calls([(e.name, e.request) for e in recorded if e.kind is Kind.TOOL]) != (
            _tool_calls([(st.name, st.request) for st in steps])
        )
        scenarios.append(
            {
                "name": s.name,
                "affected": affected,
                "termination": termination(s),
                "failed_checks": [
                    str(t) for t in sorted({check_type(c.check) for c in s.failed_checks})
                ],
                "steps": dict(sorted(s.steps.by_status.items())),
            }
        )
    causes: dict[str, int] = {}
    for sc in scenarios:
        cause = _field(sc, "termination")
        causes[cause] = causes.get(cause, 0) + 1
    return {
        "variant": variant.id,
        "policy": policy,
        "identify": identify,
        "affected": sum(1 for sc in scenarios if _flag(sc, "affected")),
        "causes": dict(sorted(causes.items())),
        "check_types": {
            t: sum(1 for sc in scenarios if t in _strings(sc, "failed_checks")) for t in configured
        },
        "approximate_steps": sum(_steps(sc, "approximate") for sc in scenarios),
        "fallback_steps": sum(_steps(sc, "fallback") for sc in scenarios),
        "scenarios": scenarios,
    }


def path_study(variants: Sequence[PathVariant]) -> dict[str, JsonValue]:
    rows: list[JsonValue] = [
        path_row(v, policy, identify=identify)
        for group in PATH_POLICY_GROUPS
        for v in variants
        for policy in group
        for identify in (True, False)
    ]
    summary: list[JsonValue] = []
    for policy in PATH_POLICIES:
        for identify in (True, False):
            subset = [
                r
                for r in rows
                if isinstance(r, dict) and r["policy"] == policy and r["identify"] == identify
            ]
            check_types: dict[str, JsonValue] = {
                t: [_field(r, "variant") for r in subset if _cell(r, "check_types", t)]
                for t in suite_check_types()
            }
            summary.append(
                {
                    "policy": policy,
                    "identify": identify,
                    "variants": len(subset),
                    "failed": [
                        _field(r, "variant") for r in subset if _cell(r, "causes", "passed") < 6
                    ],
                    "unrecorded": [
                        _field(r, "variant") for r in subset if _cell(r, "causes", "unrecorded")
                    ],
                    "approximate": [
                        _field(r, "variant") for r in subset if _int(r, "approximate_steps")
                    ],
                    "failed_by_check_type": check_types,
                }
            )
    return {
        "mode": "model-live",
        "agent": REAL_AGENT,
        "variants": [
            {"id": v.id, "description": v.description, "client": v.client} for v in variants
        ],
        "rows": rows,
        "summary": summary,
    }


def _flag(row: JsonValue, key: str) -> bool:
    assert isinstance(row, dict)
    return bool(row[key])


def _int(row: JsonValue, key: str) -> int:
    assert isinstance(row, dict)
    value = row[key]
    assert isinstance(value, int)
    return value


def _strings(row: JsonValue, key: str) -> list[str]:
    assert isinstance(row, dict)
    value = row[key]
    assert isinstance(value, list)
    return [str(v) for v in value]


def _steps(scenario: JsonValue, status: str) -> int:
    assert isinstance(scenario, dict)
    steps = scenario["steps"]
    assert isinstance(steps, dict)
    value = steps.get(status, 0)
    assert isinstance(value, int)
    return value


def _cell(row: JsonValue, key: str, name: str) -> int:
    assert isinstance(row, dict)
    table = row[key]
    assert isinstance(table, dict)
    value = table.get(name, 0)
    assert isinstance(value, int)
    return value


def _definitions() -> dict[str, JsonValue]:
    return {
        "detected": "the suite fails in that configuration (any scenario not passed)",
        "caught_by": "check types among the blocking failed checks of at least one scenario; "
        "a run that ends in an exception counts as run_error:<ExceptionType>",
        "check_type": "CheckResult.check up to its first ':'; rules keep their name",
        "caught_alone": "no other check type caught that mutation in that configuration",
        "false_alarm": "a benign refactor fails the suite in that configuration",
        "full": "every call served from the cassette: the stub-everything baseline",
        "no_approximate_steps": "with fail_on_approximate, a served step whose request "
        "differed from the recording (the drift signal) fails the scenario",
        "no_tool_drift": "with fail_on_tool_drift, a live tool whose result differs from the "
        "one recorded for the same arguments fails the scenario; with identifying arguments, "
        "so does a live call whose identity no recording has (identity_unrecorded)",
        "identify": "the suite's identifying arguments per tool: a served call is only given "
        "a recording of the same entity, else it is unrecorded and the policy decides",
        "synth": "model-live+synth answers an unrecorded tool call with a schema-minimal "
        "stub of the tool's return type (agent_replay.synth) instead of ending the run",
        "termination": "how a path-variant scenario ended: passed, unrecorded (an unrecorded "
        "call ended the run), run_error:<Type>, or checks (it finished and a check failed)",
        "affected": "a path-variant scenario whose tool calls (names and arguments, in "
        "starting order) differ from the recording's",
    }


# --- summary ----------------------------------------------------------------------


def _results(row: JsonValue, config: str) -> dict[str, JsonValue]:
    assert isinstance(row, dict)
    results = row["results"]
    assert isinstance(results, dict)
    result = results[config]
    assert isinstance(result, dict)
    return result


def _caught(row: JsonValue, config: str) -> list[str]:
    caught = _results(row, config)["caught_by"]
    assert isinstance(caught, list)
    return [str(c) for c in caught]


def _detected(row: JsonValue, config: str) -> bool:
    return not _results(row, config)["passed"]


def _field(row: JsonValue, key: str) -> str:
    assert isinstance(row, dict)
    return str(row[key])


def summarise(
    mutations: Sequence[JsonValue],
    refactors: Sequence[JsonValue],
    configs: Sequence[str],
    configured: Sequence[str] = (),
) -> dict[str, JsonValue]:
    """Detection rates, per-check-type catches and false alarms. ``configured`` check
    types get a row even when they caught nothing."""
    detection: dict[str, JsonValue] = {}
    for config in configs:
        by_target: dict[str, JsonValue] = {
            "overall": _rate([m for m in mutations if _detected(m, config)], mutations)
        }
        for target in TARGETS:
            of_target = [m for m in mutations if _field(m, "target") == target]
            by_target[target] = _rate([m for m in of_target if _detected(m, config)], of_target)
        detection[config] = by_target
    types = sorted({*configured, *(t for m in mutations for c in configs for t in _caught(m, c))})
    check_types: dict[str, JsonValue] = {}
    for t in types:
        per_config: dict[str, JsonValue] = {}
        for config in configs:
            caught = [m for m in mutations if t in _caught(m, config)]
            alone = [m for m in caught if _caught(m, config) == [t]]
            per_config[config] = {"caught": len(caught), "alone": len(alone)}
        check_types[t] = per_config
    false_alarms: dict[str, JsonValue] = {
        config: [_field(r, "id") for r in refactors if _detected(r, config)] for config in configs
    }
    survivors = [_field(m, "id") for m in mutations if not any(_detected(m, c) for c in configs)]
    return {
        "mutations": len(mutations),
        "refactors": len(refactors),
        "detection": detection,
        "check_types": check_types,
        "false_alarms": false_alarms,
        "survived_every_configuration": list(survivors),
    }


def _rate(hits: Sequence[object], of: Sequence[object]) -> dict[str, JsonValue]:
    return {"detected": len(hits), "total": len(of)}


# --- Markdown ---------------------------------------------------------------------


def _fraction(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n * 100:.0f}%)" if n else "0/0"


def to_markdown(study: dict[str, JsonValue]) -> str:
    configs = [_field(c, "id") for c in _list(study["configurations"])]
    mutations = _list(study["mutations"])
    refactors = _list(study["refactors"])
    summary = study["summary"]
    assert isinstance(summary, dict)
    detection = summary["detection"]
    check_types = summary["check_types"]
    assert isinstance(detection, dict)
    assert isinstance(check_types, dict)
    header = "| " + " | ".join(configs) + " |"
    rule = "|" + "---|" * len(configs)
    lines = [
        "## Mutation study: retail suite",
        "",
        f"{len(mutations)} mutations ({', '.join(_count(mutations, t) for t in TARGETS)}) and "
        f"{len(refactors)} benign refactors, each replayed against the six-scenario suite "
        f"`{study['suite']}` in {len(configs)} configurations. `full` serves every call "
        "from the cassette: it is the stub-everything baseline.",
        "",
        "Detected: the suite fails. A check type catches a mutation when it is among the "
        "blocking failed checks of some scenario; `run_error:<Type>` is a run that ended in "
        "that exception. Alone: no other check type caught it in that configuration.",
        "",
        "### Detection rate by configuration",
        "",
        "| configuration | all | agent | model | tool |",
        "|---|---|---|---|---|",
    ]
    for config in configs:
        rates = detection[config]
        assert isinstance(rates, dict)
        cells = [_rate_cell(rates[k]) for k in ("overall", *TARGETS)]
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Mutations caught per check type (caught / all mutations, alone)",
        "",
        "| check type " + header,
        "|---" + rule,
    ]
    for t, per_config in sorted(check_types.items()):
        assert isinstance(per_config, dict)
        cells = []
        for config in configs:
            stats = per_config[config]
            assert isinstance(stats, dict)
            cells.append(f"{stats['caught']}/{len(mutations)} (alone {stats['alone']})")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")
    lines += ["", "### Per mutation (check types that caught it; — survived)", ""]
    lines += ["| mutation | target " + header, "|---|---" + rule]
    for m in mutations:
        cells = [", ".join(f"`{t}`" for t in _caught(m, c)) or "—" for c in configs]
        lines.append(f"| `{_field(m, 'id')}` | {_field(m, 'target')} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Benign refactors (false alarms; approximate steps)",
        "",
        "| refactor " + header,
        "|---" + rule,
    ]
    for r in refactors:
        cells = []
        for config in configs:
            result = _results(r, config)
            verdict = (
                "pass" if result["passed"] else "FALSE ALARM: " + ", ".join(_caught(r, config))
            )
            cells.append(f"{verdict}; {result['approximate_steps']} approx")
        lines.append(f"| `{_field(r, 'id')}` | " + " | ".join(cells) + " |")
    false_alarms = summary["false_alarms"]
    assert isinstance(false_alarms, dict)
    lines += [
        "",
        "False alarms: "
        + ", ".join(f"{c} {len(_list(false_alarms[c]))}/{len(refactors)}" for c in configs)
        + ".",
    ]
    survivors = _list(summary["survived_every_configuration"])
    lines.append(
        "Survived every configuration: "
        + (", ".join(f"`{s}`" for s in survivors) if survivors else "none")
        + "."
    )
    if "identify" in study:
        lines += identify_markdown(study, configs)
    if "path_variants" in study:
        lines += paths_markdown(study["path_variants"])
    return "\n".join(lines) + "\n"


def _caught_cell(row: JsonValue, config: str) -> str:
    return ", ".join(f"`{t}`" for t in _caught(row, config)) or "—"


def identify_markdown(study: dict[str, JsonValue], configs: Sequence[str]) -> list[str]:
    section = study["identify"]
    assert isinstance(section, dict)
    keys = section["keys"]
    assert isinstance(keys, dict)
    keyed = _list(section["mutations"])
    keyed_refactors = _list(section["refactors"])
    new_ids = [str(i) for i in _list(section["identity_mutations"])]
    with_keys = {_field(m, "id"): m for m in keyed}
    without_keys = {_field(m, "id"): m for m in _list(section["identity_mutations_without_keys"])}
    summary = section["summary"]
    assert isinstance(summary, dict)
    detection = summary["detection"]
    assert isinstance(detection, dict)
    header = "| " + " | ".join(configs) + " |"
    rule = "|" + "---|" * len(configs)
    lines = [
        "",
        "## Identifying arguments (M11)",
        "",
        "The suite's `[defaults.identify]` table names the arguments that identify what each "
        "tool reads or writes: "
        + "; ".join(
            f"`{tool}` {', '.join(f'`{p}`' for p in _list(paths))}" for tool, paths in keys.items()
        )
        + ". With it, a served call is only given a recording of the same entity; a call for "
        "an entity no unused recording has is unrecorded, and in `tools-live+drift` a live "
        "call whose identity no recording has is drift (`identity_unrecorded`). The tables "
        "above ran without it, as v1 did.",
        "",
        "### Mutations only identity can see (caught by; — survived)",
        "",
        "| mutation | identify " + header,
        "|---|---" + rule,
    ]
    for mutation_id in new_ids:
        for label, row in (
            ("without", without_keys[mutation_id]),
            ("with", with_keys[mutation_id]),
        ):
            cells = [_caught_cell(row, c) for c in configs]
            lines.append(f"| `{mutation_id}` | {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Detected (of {len(new_ids)}) without → with the keys: "
        + ", ".join(
            f"{c} {sum(_detected(without_keys[i], c) for i in new_ids)} → "
            f"{sum(_detected(with_keys[i], c) for i in new_ids)}"
            for c in configs
        )
        + ".",
        "",
        f"### Detection rate with the keys ({len(keyed)} mutations)",
        "",
        "| configuration | all | agent | model | tool |",
        "|---|---|---|---|---|",
    ]
    for config in configs:
        rates = detection[config]
        assert isinstance(rates, dict)
        cells = [_rate_cell(rates[k]) for k in ("overall", *TARGETS)]
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    false_alarms = summary["false_alarms"]
    assert isinstance(false_alarms, dict)
    lines += [
        "",
        "False alarms with the keys: "
        + ", ".join(f"{c} {len(_list(false_alarms[c]))}/{len(keyed_refactors)}" for c in configs)
        + ".",
        "",
        "Cells of the v1 tables the keys changed:",
        "",
    ]
    changed = _list(section["changed_by_keys"])
    for cell in changed:
        assert isinstance(cell, dict)
        before, after = cell["without_keys"], cell["with_keys"]
        assert isinstance(before, dict)
        assert isinstance(after, dict)

        def show(result: dict[str, JsonValue]) -> str:
            caught = ", ".join(f"`{c}`" for c in _list(result["caught_by"])) or "passed"
            return f"{caught} ({result['approximate_steps']} approx)"

        lines.append(f"- `{cell['id']}` in {cell['config']}: {show(before)} → {show(after)}")
    if not changed:
        lines.append("- none")
    return lines


def paths_markdown(section: JsonValue) -> list[str]:
    assert isinstance(section, dict)
    rows = _list(section["rows"])
    first = rows[0]
    assert isinstance(first, dict)
    check_types = first["check_types"]
    assert isinstance(check_types, dict)
    types = list(check_types)
    lines = [
        "",
        "## Correct-but-different paths (M11)",
        "",
        "Five variants of the fake model reach the same database and answer by another path, "
        "each replayed in `model-live` against the six recordings under `strict`, "
        "`fallback` and `synth`, with and without the identify keys. Affected: scenarios "
        "whose tool calls differ from the recording. Termination: how each scenario ended "
        "(`unrecorded`: an unrecorded call ended the run; `checks`: it finished and a check "
        "failed). A check type's cell counts the scenarios it failed (· for none); every "
        "failure here is a false alarm.",
        "",
        "| variant | policy | identify | affected | termination | approx | fallback | "
        + " | ".join(f"`{t}`" for t in types)
        + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(types),
    ]
    for row in rows:
        assert isinstance(row, dict)
        causes = row["causes"]
        assert isinstance(causes, dict)
        termination_cell = ", ".join(f"{n} {cause}" for cause, n in causes.items())
        cells = [str(_cell(row, "check_types", t) or "·") for t in types]
        lines.append(
            f"| `{row['variant']}` | {row['policy']} | {'on' if row['identify'] else 'off'} "
            f"| {row['affected']}/6 | {termination_cell} | {row['approximate_steps']} "
            f"| {row['fallback_steps']} | " + " | ".join(cells) + " |"
        )
    lines += ["", "Per policy (variants out of 5):", ""]
    for entry in _list(section["summary"]):
        assert isinstance(entry, dict)
        failed_by = entry["failed_by_check_type"]
        assert isinstance(failed_by, dict)
        caught = ", ".join(f"`{t}` {len(_list(v))}" for t, v in failed_by.items() if _list(v))
        lines.append(
            f"- {entry['policy']}, identify {'on' if entry['identify'] else 'off'}: "
            f"{len(_list(entry['failed']))} raise a false alarm; "
            f"{len(_list(entry['unrecorded']))} end on an unrecorded call "
            f"({', '.join(f'`{v}`' for v in _list(entry['unrecorded'])) or 'none'}); "
            f"failing check types: {caught or 'none'}."
        )
    return lines


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _count(mutations: Sequence[JsonValue], target: str) -> str:
    return f"{sum(1 for m in mutations if _field(m, 'target') == target)} {target}"


def _rate_cell(rate: JsonValue) -> str:
    assert isinstance(rate, dict)
    detected, total = rate["detected"], rate["total"]
    assert isinstance(detected, int)
    assert isinstance(total, int)
    return _fraction(detected, total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, help="write the results as JSON here")
    parser.add_argument("--markdown", type=Path, help="also write the Markdown tables here")
    args = parser.parse_args(argv)
    # Mutants end runs in exceptions by design; the agent's warnings would drown the tables.
    os.environ.setdefault("REPLAY_LOG_LEVEL", "ERROR")
    configure_logging()
    try:
        study = run()
    except HarnessError as exc:
        print(f"error: harness bug, not a regression: {exc}", file=sys.stderr)
        return 2
    markdown = to_markdown(study)
    print(markdown, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
