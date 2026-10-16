"""The ``replay calibrate`` command (registered on the main app by :mod:`agent_replay.cli`).

It prints what the run will cost before making any judge call, so a large
golden set or a high ``--repeats`` never spends money by surprise.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.markup import escape
from rich.table import Table
from sqlalchemy.exc import SQLAlchemyError

from agent_replay._cli import DB_OPTION, console, fail
from agent_replay.calibrate.calibrate import PROBES, Estimate, Reference, calibrate, estimate
from agent_replay.calibrate.golden import OVERALL, GoldenSet
from agent_replay.errors import ReplayError
from agent_replay.judge.cache import JudgeCache
from agent_replay.judge.calibration import (
    CalibrationReport,
    CriterionCalibration,
    criterion_gates,
    report_blocker,
    resolve_min_items,
    resolve_min_kappa,
)
from agent_replay.judge.providers import make_provider
from agent_replay.judge.rubric import Rubric
from agent_replay.store import Store

log = structlog.get_logger("agent_replay")


class ReferenceChoice(StrEnum):
    human = "human"
    objective = "objective"


class ProbeChoice(StrEnum):
    position = "position"
    verbosity = "verbosity"
    self_preference = "self-preference"


def _cost(value: float | None) -> str:
    return (
        "unknown (the judge model has no price; add it with REPLAY_PRICES_FILE)"
        if (value is None)
        else f"${value:.4f}"
    )


def _print_estimate(est: Estimate) -> None:
    console.print(
        f"estimate: {est.calls} judge call(s) to make, {est.cached} cached (of {est.total} "
        f"verdicts); estimated cost {_cost(est.est_cost_usd)}"
    )
    for note in est.notes:
        console.print(f"[dim]  note: {escape(note)}[/]")


def _num(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.2f}"


def _kappa(c: CriterionCalibration) -> str:
    if c.kappa is None:
        return "undefined"
    interval = f" [{_num(c.ci_low)}, {_num(c.ci_high)}]" if c.ci_low is not None else ""
    return f"{c.kappa:.2f}{interval}"


def _consistency(c: CriterionCalibration) -> str:
    sc = c.self_consistency
    if sc is None:
        return "-"
    pairwise = "" if sc.mean_pairwise_kappa is None else f", pairwise {sc.mean_pairwise_kappa:.2f}"
    return f"{sc.unanimous:.0%} unanimous{pairwise} (n={sc.n})"


def _print_report(report: CalibrationReport, min_kappa: float) -> None:
    measure = "kappa [95% CI]" if report.stats_impl != "baseline" else "kappa* [95% CI]"
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in (
        "criterion",
        measure,
        "agreement",
        "majority",
        "pass/pass, false pass, false fail, fail/fail",
        "self-consistency",
        "n",
        "gates",
    ):
        table.add_column(column)
    gates = criterion_gates(report, min_kappa)
    for name, c in report.criteria.items():
        confusion = c.confusion
        table.add_row(
            escape(name),
            _kappa(c),
            f"{c.agreement:.0%}",
            f"{c.majority_agreement:.0%} ({c.majority_label})"
            if c.majority_agreement is not None
            else "-",
            (
                f"{confusion.both_pass}, {confusion.judge_pass_reference_fail}, "
                f"{confusion.judge_fail_reference_pass}, {confusion.both_fail}"
            )
            if confusion is not None
            else "-",
            _consistency(c),
            str(c.n),
            "-" if name == OVERALL else ("[green]yes[/]" if gates[name] else "no"),
        )
    console.print(table)
    console.print(f"[dim]{escape(report.method)}[/]")
    blocker = report_blocker(report)
    if blocker is not None:
        console.print(f"[yellow]no criterion gates: {escape(blocker)}[/]")
    else:
        console.print(
            f"a criterion gates when its kappa lower bound is at least {min_kappa:.2f} over at "
            f"least {resolve_min_items()} labelled items (REPLAY_JUDGE_MIN_ITEMS)"
        )
    for probe in report.probes:
        name = probe.probe.replace("_", "-")
        if probe.skipped is not None:
            console.print(f"probe {name}: skipped ({escape(probe.skipped)})")
        elif probe.flips is not None:
            parts = [
                f"{escape(cid)} {f.flips}/{f.n} ({f.to_pass} to pass, {f.to_fail} to fail)"
                for cid, f in probe.flips.items()
            ]
            console.print(f"probe {name}: flips " + "; ".join(parts))
        elif probe.groups is not None:
            parts = []
            for cid, groups in probe.groups.items():
                same, other = groups["same_model"], groups["other_model"]
                parts.append(
                    f"{escape(cid)} false pass {_rate(same.false_pass_rate)} same model "
                    f"(n={same.n}) vs {_rate(other.false_pass_rate)} other (n={other.n})"
                )
            console.print(f"probe {name}: " + "; ".join(parts))
    console.print(
        f"{report.n_items} item(s), {report.judge_calls} judge call(s), {report.cached_calls} "
        f"cached; cost {_cost(report.cost_usd)}"
    )


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def calibrate_cmd(
    golden_dir: Annotated[
        Path, typer.Argument(metavar="GOLDEN_DIR", help="Golden set directory (replay golden).")
    ],
    rubric_path: Annotated[Path, typer.Option("--rubric", help="Rubric YAML the judge grades.")],
    provider: Annotated[
        str | None,
        typer.Option(
            help="Judge provider: anthropic, ollama, fake or module:factory. Default: "
            "$REPLAY_JUDGE_PROVIDER, then anthropic."
        ),
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Judge model (anthropic, ollama); $REPLAY_JUDGE_MODEL.")
    ] = None,
    effort: Annotated[
        str | None, typer.Option(help="Anthropic judge effort; $REPLAY_JUDGE_EFFORT, then low.")
    ] = None,
    reference: Annotated[
        ReferenceChoice,
        typer.Option(help="Compare with human labels (may gate) or objective ones (never gates)."),
    ] = ReferenceChoice.human,
    labeller: Annotated[
        str | None, typer.Option(help="Whose labels, when the reference has several labellers.")
    ] = None,
    repeats: Annotated[
        int, typer.Option(min=1, help="Verdicts per item; 2 or more measure self-consistency.")
    ] = 3,
    probe: Annotated[
        list[ProbeChoice] | None,
        typer.Option("--probe", help="Run only this bias probe (repeatable). Default: all."),
    ] = None,
    no_probes: Annotated[bool, typer.Option("--no-probes", help="Run no bias probe.")] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option(help="Judge cache directory; $REPLAY_JUDGE_CACHE, then .replay/judge-cache."),
    ] = None,
    offline: Annotated[
        bool | None,
        typer.Option(
            "--offline/--online",
            help=(
                "Use cached verdicts only and exit 2 if any is missing; --online lets the "
                "judge fill misses. Default: $REPLAY_JUDGE_OFFLINE."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the estimate and stop.")
    ] = False,
    max_cost_usd: Annotated[
        float | None,
        typer.Option(min=0, help="Abort (exit 2) when the estimated cost is above this."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report here.")
    ] = None,
    use_store: Annotated[
        bool,
        typer.Option("--store/--no-store", help="Save the report in the run store (dashboard)."),
    ] = True,
    db: Annotated[str | None, DB_OPTION] = None,
    min_kappa: Annotated[
        float | None,
        typer.Option(help="Kappa lower bound a criterion needs to gate; $REPLAY_JUDGE_MIN_KAPPA."),
    ] = None,
    resamples: Annotated[int, typer.Option(min=1, help="Bootstrap resamples.")] = 2000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
) -> None:
    """Measure how well the judge agrees with the golden set's labels on RUBRIC.

    Prints the estimated number of judge calls and their cost first. Per
    criterion: kappa with a bootstrap CI (raw agreement under
    REPLAY_STATS_IMPL=baseline), agreement against the majority-label baseline,
    the confusion matrix, self-consistency over --repeats verdicts, and whether
    the criterion would gate a judge check; then the bias probes. Exit 0 when
    the report was made, 2 on bad input, a missing cached verdict offline, or a
    cost above --max-cost-usd.
    """
    if no_probes and probe:
        raise fail("--probe and --no-probes contradict each other; pass one")
    probes = () if no_probes else tuple(p.value for p in probe) if probe else PROBES
    ref: Reference = "human" if reference is ReferenceChoice.human else "objective"
    try:
        threshold = resolve_min_kappa(min_kappa)
        golden = GoldenSet.load(golden_dir)
        rubric = Rubric.load(rubric_path)
        judge = make_provider(provider, model=model, effort=effort)
        cache = JudgeCache(cache_dir, offline=offline)
        est = estimate(
            golden,
            rubric,
            judge,
            cache,
            repeats=repeats,
            probes=probes,
            reference=ref,
            labeller=labeller,
        )
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    stale = golden.stale_labels(rubric) if ref == "human" else 0
    if stale:
        console.print(
            f"[yellow]{stale} human label(s) answer an earlier version of their criterion and "
            "are left out; relabel with `replay label`[/]"
        )
    _print_estimate(est)
    if max_cost_usd is not None and est.calls:
        if est.est_cost_usd is None:
            raise fail(
                f"cannot estimate the cost of judge {judge.judge_id} (its model has no price), "
                "so --max-cost-usd cannot be checked; add the model with REPLAY_PRICES_FILE or "
                "drop --max-cost-usd"
            )
        if est.est_cost_usd > max_cost_usd:
            raise fail(
                f"the estimated cost ${est.est_cost_usd:.4f} is above --max-cost-usd "
                f"{max_cost_usd}; raise --max-cost-usd, lower --repeats, or drop probes"
            )
    if dry_run:
        return
    try:
        report = calibrate(
            golden,
            rubric,
            judge,
            cache=cache,
            reference=ref,
            labeller=labeller,
            repeats=repeats,
            probes=probes,
            n_resamples=resamples,
            seed=seed,
        )
    except ReplayError as exc:
        raise fail(str(exc)) from exc
    _print_report(report, threshold)
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise fail(f"cannot write the calibration report to {output}: {exc}") from exc
        console.print(f"wrote {escape(str(output))}")
    if use_store:
        try:
            with Store.open(db) as store:
                report_id = store.save_calibration(report)
        except (ReplayError, SQLAlchemyError) as exc:
            raise fail(f"could not store the calibration report: {exc}") from exc
        console.print(f"stored calibration report {report_id[:12]}")
    log.info("calibrate.cli_done", rubric=rubric.name, judge_id=judge.judge_id)


def register(app: typer.Typer) -> None:
    """Add this module's commands to ``app``."""
    app.command("calibrate")(calibrate_cmd)
