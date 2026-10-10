"""Which regressions does replay catch, with which check, at which cut point? (PLAN.md D4)

Replays the retail suite (``examples/retail_agent/replay.toml``, six τ-bench
tasks with their committed fake-model cassettes) against every mutation and
benign refactor in ``eval/mutations/``, in four configurations:

- ``full``: every model and tool call served from the cassette (strict). This
  is the stub-everything baseline: it tests the agent's own code only.
- ``tools-live``: tools run for real against recorded model replies (strict).
- ``model-live``: the model runs (the fake model, or the mutated one for
  ``model`` mutations) against recorded tools (strict: a tool call the
  cassette has no envelope for ends the run).
- ``model-live+fallback``: the same, but unrecorded tool calls run the real
  tools, so a model that leaves the recording is judged by the checks.

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

from agent_replay._logging import configure_logging
from agent_replay.ci import Overrides, ScenarioReport, SuiteReport, load_suite, run_suite
from agent_replay.session import Policy, ReplayMode
from eval.mutations import MUTATIONS, REFACTORS, Mutation, Refactor, Target
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


def replay_suite(agent: str, client: str, config: Config) -> Outcome:
    """Run the suite once with ``agent`` in ``config``; the database starts fresh."""
    retail_tools.reset_db()
    try:
        report = run_suite(
            SUITE,
            overrides=Overrides(
                agent=agent,
                mode=config.mode,
                policy=config.policy,
                client=client if config.model_live else None,
            ),
        )
    finally:
        retail_tools.reset_db()
    if report.broken:
        problems = "; ".join(
            f"{s.name} ({s.error.type}: {s.error.message})" if s.error else s.name
            for s in report.broken
        )
        raise HarnessError(
            f"{agent} in {config.id}: scenario(s) did not run: {problems}. Fix the "
            "catalogue entry (agent or client reference) or the suite"
        )
    return outcome(report)


def run(
    mutations: Sequence[Mutation] = MUTATIONS,
    refactors: Sequence[Refactor] = REFACTORS,
    configs: Sequence[Config] = CONFIGS,
) -> dict[str, JsonValue]:
    """Replay every variant in every configuration; the study's results as JSON data."""
    mutation_rows: list[JsonValue] = []
    for m in mutations:
        results = {c.id: replay_suite(m.agent, m.client, c) for c in configs}
        mutation_rows.append(
            {
                "id": m.id,
                "target": m.target,
                "description": m.description,
                "agent": m.agent,
                "client": m.client,
                "results": {k: v.to_json() for k, v in results.items()},
            }
        )
    refactor_rows: list[JsonValue] = []
    for r in refactors:
        results = {c.id: replay_suite(r.agent, r.client, c) for c in configs}
        refactor_rows.append(
            {
                "id": r.id,
                "description": r.description,
                "agent": r.agent,
                "client": r.client,
                "results": {k: v.to_json() for k, v in results.items()},
            }
        )
    return {
        "suite": SUITE.relative_to(ROOT).as_posix(),
        "definitions": _definitions(),
        "configurations": [
            {"id": c.id, "mode": c.mode, "policy": c.policy, "note": c.note} for c in configs
        ],
        "mutations": mutation_rows,
        "refactors": refactor_rows,
        "summary": summarise(
            mutation_rows, refactor_rows, [c.id for c in configs], suite_check_types()
        ),
    }


def _definitions() -> dict[str, JsonValue]:
    return {
        "detected": "the suite fails in that configuration (any scenario not passed)",
        "caught_by": "check types among the blocking failed checks of at least one scenario; "
        "a run that ends in an exception counts as run_error:<ExceptionType>",
        "check_type": "CheckResult.check up to its first ':'; rules keep their name",
        "caught_alone": "no other check type caught that mutation in that configuration",
        "false_alarm": "a benign refactor fails the suite in that configuration",
        "full": "every call served from the cassette: the stub-everything baseline",
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
    return "\n".join(lines) + "\n"


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
