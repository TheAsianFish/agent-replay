"""The run store: every replayed or recorded run, its steps and its check results.

One SQLAlchemy model set serves two backends (PLAN.md decision 7): SQLite under
``.replay/`` for the CLI, PostgreSQL for the dashboard server. Only portable
column types are used so the same models work on both.

The schema is versioned with Alembic (``agent_replay/migrations``). Opening a
store upgrades it to the newest revision under a lock, so two processes opening
the same new or older store at once neither collide nor migrate twice: SQLite
takes its write lock with ``BEGIN IMMEDIATE`` and migrates in that transaction;
PostgreSQL takes a transaction-scoped advisory lock. Both run DDL inside the
transaction, so a failed migration leaves the store as it was.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import structlog
from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Select,
    String,
    Text,
    create_engine,
    event,
    false,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, Connection, Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import NullPool, StaticPool
from sqlalchemy.schema import CreateColumn

from agent_replay.cassette import ErrorInfo
from agent_replay.checks.base import CheckResult
from agent_replay.errors import StoreError
from agent_replay.flake import FlakeReport, FlakeSettings, RunOutcome
from agent_replay.judge.calibration import CalibrationReport
from agent_replay.runner import RunResult
from agent_replay.trajectory import StepStatus

log = structlog.get_logger("agent_replay")

DEFAULT_URL = "sqlite:///.replay/replay.db"
MIN_PREFIX = 6


def _utc(value: datetime) -> datetime:
    """SQLite drops tzinfo; every stored timestamp is UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SuiteRow(Base):
    __tablename__ = "suites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_path: Mapped[str] = mapped_column(String)
    git_sha: Mapped[str | None] = mapped_column(String)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    scenario_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    suite_id: Mapped[str | None] = mapped_column(ForeignKey("suites.id"), index=True)
    scenario: Mapped[str] = mapped_column(String, index=True)
    cassette_path: Mapped[str] = mapped_column(String)
    cassette_id: Mapped[str | None] = mapped_column(String)
    agent: Mapped[str] = mapped_column(String)
    mode: Mapped[str] = mapped_column(String)
    policy: Mapped[str] = mapped_column(String)
    task: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    output: Mapped[Any] = mapped_column(JSON)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    approximate_count: Mapped[int] = mapped_column(Integer, default=0)
    unconsumed_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)

    steps: Mapped[list[StepRow]] = relationship(
        order_by="StepRow.index", cascade="all, delete-orphan", passive_deletes=True
    )
    checks: Mapped[list[CheckRow]] = relationship(
        order_by="CheckRow.position", cascade="all, delete-orphan", passive_deletes=True
    )


class StepRow(Base):
    __tablename__ = "steps"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    index: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    kind: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    fingerprint: Mapped[str] = mapped_column(String)
    request: Mapped[Any] = mapped_column(JSON)
    response: Mapped[Any] = mapped_column(JSON)
    error: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    span_id: Mapped[str | None] = mapped_column(String)
    parent_span_id: Mapped[str | None] = mapped_column(String)
    envelope_seq: Mapped[int | None] = mapped_column(Integer)


class CheckRow(Base):
    __tablename__ = "check_results"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    check: Mapped[str] = mapped_column(String)
    passed: Mapped[bool] = mapped_column(Boolean)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[Any] = mapped_column(JSON)
    advisory: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())


class FlakeReportRow(Base):
    __tablename__ = "flake_reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    suite_id: Mapped[str | None] = mapped_column(ForeignKey("suites.id"), index=True)
    suite: Mapped[str] = mapped_column(String)
    stats_impl: Mapped[str] = mapped_column(String)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON)
    passed: Mapped[bool] = mapped_column(Boolean)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_ms: Mapped[float] = mapped_column(Float)
    git_sha: Mapped[str | None] = mapped_column(String)
    scenario_count: Mapped[int] = mapped_column(Integer)
    failed_count: Mapped[int] = mapped_column(Integer)
    flaky_count: Mapped[int] = mapped_column(Integer)
    run_count: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)

    scenarios: Mapped[list[FlakeScenarioRow]] = relationship(
        order_by="FlakeScenarioRow.position", cascade="all, delete-orphan", passive_deletes=True
    )


class FlakeScenarioRow(Base):
    __tablename__ = "flake_scenarios"

    report_id: Mapped[str] = mapped_column(
        ForeignKey("flake_reports.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String, index=True)
    cassette: Mapped[str] = mapped_column(String)
    agent: Mapped[str] = mapped_column(String)
    mode: Mapped[str] = mapped_column(String)
    policy: Mapped[str] = mapped_column(String)
    classification: Mapped[str] = mapped_column(String)
    decision: Mapped[str] = mapped_column(String)
    runs: Mapped[int] = mapped_column(Integer)
    passes: Mapped[int] = mapped_column(Integer)
    pass_rate: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
    failures: Mapped[dict[str, Any]] = mapped_column(JSON)
    executed: Mapped[int] = mapped_column(Integer)
    concurrency: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    error: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    outcomes: Mapped[list[Any]] = mapped_column(JSON)


class CalibrationRow(Base):
    __tablename__ = "calibration_reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    rubric: Mapped[str] = mapped_column(String, index=True)
    rubric_fingerprint: Mapped[str] = mapped_column(String)
    judge_id: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    n_items: Mapped[int] = mapped_column(Integer)
    report: Mapped[dict[str, Any]] = mapped_column(JSON)
    """The whole :class:`CalibrationReport`, so fields added later survive the round trip."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class SuiteSummary(_Model):
    id: str
    started_at: datetime
    finished_at: datetime | None
    config_path: str
    git_sha: str | None
    passed: bool | None
    scenario_count: int
    failed_count: int
    run_count: int = 0
    """Runs actually stored under this suite."""


class RunSummary(_Model):
    id: str
    suite_id: str | None
    scenario: str
    cassette_path: str
    cassette_id: str | None
    agent: str
    mode: str
    policy: str
    task: str
    passed: bool
    error_type: str | None
    error_message: str | None
    cost_usd: float
    tokens: int
    step_count: int
    approximate_count: int
    unconsumed_count: int
    started_at: datetime
    duration_ms: float


class StepRecord(_Model):
    index: int
    kind: str
    name: str
    status: str
    fingerprint: str
    request: JsonValue
    response: JsonValue
    error: JsonValue
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    span_id: str | None
    parent_span_id: str | None
    envelope_seq: int | None


class CheckRecord(_Model):
    position: int
    check: str
    passed: bool
    message: str
    details: dict[str, JsonValue]
    advisory: bool = False


class RunRecord(RunSummary):
    output: JsonValue
    steps: list[StepRecord]
    checks: list[CheckRecord]


class FlakeReportSummary(_Model):
    id: str
    suite_id: str | None
    suite: str
    stats_impl: str
    passed: bool
    started_at: datetime
    duration_ms: float
    git_sha: str | None
    scenario_count: int
    failed_count: int
    flaky_count: int
    run_count: int
    cost_usd: float


class FlakeScenarioRecord(_Model):
    name: str
    cassette: str
    agent: str
    mode: str
    policy: str
    classification: str
    decision: str
    runs: int
    passes: int
    pass_rate: float | None
    ci_low: float
    ci_high: float
    failures: dict[str, int]
    executed: int
    concurrency: int
    cost_usd: float
    error: ErrorInfo | None
    outcomes: list[RunOutcome]


class FlakeReportRecord(FlakeReportSummary):
    settings: FlakeSettings
    scenarios: list[FlakeScenarioRecord]


class FlakePoint(_Model):
    """One scenario's estimate in one ``replay flake`` report: a point on its trend."""

    report_id: str
    started_at: datetime
    stats_impl: str
    git_sha: str | None
    name: str
    classification: str
    decision: str
    runs: int
    passes: int
    pass_rate: float | None
    ci_low: float
    ci_high: float


class CalibrationSummary(_Model):
    id: str
    rubric: str
    rubric_fingerprint: str
    judge_id: str
    prompt_version: str
    created_at: datetime
    stored_at: datetime
    n_items: int


class CalibrationRecord(_Model):
    id: str
    stored_at: datetime
    report: CalibrationReport


def _summary(row: RunRow) -> RunSummary:
    summary = RunSummary.model_validate(row)
    return summary.model_copy(update={"started_at": _utc(summary.started_at)})


def _record(row: RunRow) -> RunRecord:
    base = RunSummary.model_validate(row).model_dump()
    base["started_at"] = _utc(base["started_at"])
    return RunRecord(
        **base,
        output=row.output,
        steps=[StepRecord.model_validate(s) for s in row.steps],
        checks=[CheckRecord.model_validate(c) for c in row.checks],
    )


def _suite_summary(row: SuiteRow, run_count: int) -> SuiteSummary:
    return SuiteSummary(
        id=row.id,
        started_at=_utc(row.started_at),
        finished_at=_utc(row.finished_at) if row.finished_at else None,
        config_path=row.config_path,
        git_sha=row.git_sha,
        passed=row.passed,
        scenario_count=row.scenario_count,
        failed_count=row.failed_count,
        run_count=run_count,
    )


def _flake_summary(row: FlakeReportRow) -> FlakeReportSummary:
    summary = FlakeReportSummary.model_validate(row)
    return summary.model_copy(update={"started_at": _utc(summary.started_at)})


def _calibration_summary(row: CalibrationRow) -> CalibrationSummary:
    summary = CalibrationSummary.model_validate(row)
    return summary.model_copy(
        update={"created_at": _utc(summary.created_at), "stored_at": _utc(summary.stored_at)}
    )


def _like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class Store:
    """Persistent record of runs. Methods return pydantic models, never ORM objects."""

    def __init__(self, engine: Engine, *, clock: Callable[[], datetime] = _now) -> None:
        self._engine = engine
        self._clock = clock
        self._session = sessionmaker(engine, expire_on_commit=False)
        # An in-memory SQLite store shares one connection between every checkout, and
        # pysqlite is not safe to use from several threads at once (the server runs
        # its endpoints in a thread pool), so its sessions take turns.
        self._lock: AbstractContextManager[object] = (
            threading.RLock() if isinstance(engine.pool, StaticPool) else nullcontext()
        )

    @classmethod
    def open(cls, url: str | None = None, *, clock: Callable[[], datetime] = _now) -> Store:
        """Open (and create if needed) the store.

        ``url`` defaults to ``$REPLAY_DB_URL``, then ``sqlite:///.replay/replay.db``.
        """
        raw = url or os.environ.get("REPLAY_DB_URL") or DEFAULT_URL
        try:
            parsed = make_url(raw)
        except ArgumentError as exc:
            # The message never echoes the URL: it may carry a password.
            raise StoreError(
                "invalid database URL in REPLAY_DB_URL or --db. "
                f"Set it to a SQLAlchemy URL such as {DEFAULT_URL!r} or "
                "'postgresql+psycopg://user:password@host/db'."
            ) from exc
        shown = parsed.render_as_string(hide_password=True)
        sqlite = parsed.get_backend_name() == "sqlite"
        memory = sqlite and parsed.database in (None, "", ":memory:")
        kwargs: dict[str, Any] = {}
        if memory:
            # One shared connection, or every checkout would see a fresh empty database.
            kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        engine: Engine | None = None
        try:
            if sqlite and not memory:
                _ensure_parent(parsed)
            engine = create_engine(parsed, **kwargs)
            if sqlite:
                event.listen(engine, "connect", _sqlite_pragmas)
            migrate(engine)
        except StoreError:
            if engine is not None:
                engine.dispose()
            raise
        except (SQLAlchemyError, OSError, ImportError, CommandError) as exc:
            if engine is not None:
                engine.dispose()
            raise StoreError(
                f"cannot open the run store at {shown!r}: {exc}. "
                "Check that the location is writable, or set REPLAY_DB_URL to another one."
            ) from exc
        return cls(engine, clock=clock)

    def close(self) -> None:
        self._engine.dispose()

    @property
    def backend(self) -> str:
        """``sqlite`` or ``postgresql``."""
        return self._engine.dialect.name

    def revision(self) -> str | None:
        """The schema revision this store is at."""
        return current_revision(self._engine)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def start_suite(self, *, config_path: str, git_sha: str | None = None) -> str:
        suite_id = _new_id()
        with self._lock, self._session.begin() as session:
            session.add(
                SuiteRow(
                    id=suite_id,
                    started_at=self._clock(),
                    config_path=config_path,
                    git_sha=git_sha,
                )
            )
        return suite_id

    def finish_suite(
        self, suite_id: str, *, passed: bool, scenario_count: int, failed_count: int
    ) -> None:
        with self._lock, self._session.begin() as session:
            suite = session.get(SuiteRow, suite_id)
            if suite is None:
                raise StoreError(f"no suite {suite_id!r} in the store; call start_suite first")
            suite.finished_at = self._clock()
            suite.passed = passed
            suite.scenario_count = scenario_count
            suite.failed_count = failed_count

    def save_run(
        self,
        result: RunResult,
        checks: Sequence[CheckResult],
        *,
        scenario: str,
        cassette_path: str,
        agent: str,
        passed: bool,
        started_at: datetime,
        duration_ms: float,
        suite_id: str | None = None,
    ) -> str:
        """Store a run with its steps and checks in one transaction."""
        run_id = _new_id()
        steps = [
            StepRow(
                index=step.index,
                kind=step.kind.value,
                name=step.name,
                status=step.status.value,
                fingerprint=step.fingerprint,
                request=step.request,
                response=step.response,
                error=step.error.model_dump(mode="json") if step.error else None,
                input_tokens=step.usage.input_tokens if step.usage else None,
                output_tokens=step.usage.output_tokens if step.usage else None,
                cache_read_input_tokens=step.usage.cache_read_input_tokens if step.usage else None,
                cache_creation_input_tokens=(
                    step.usage.cache_creation_input_tokens if step.usage else None
                ),
                cost_usd=step.cost_usd,
                latency_ms=step.latency_ms,
                span_id=step.span_id,
                parent_span_id=step.parent_span_id,
                envelope_seq=step.envelope_seq,
            )
            for step in result.trajectory.steps
        ]
        row = RunRow(
            id=run_id,
            suite_id=suite_id,
            scenario=scenario,
            cassette_path=cassette_path,
            cassette_id=result.cassette_id,
            agent=agent,
            mode=result.mode,
            policy=result.policy,
            task=result.task,
            passed=passed,
            error_type=result.error.type if result.error else None,
            error_message=result.error.message if result.error else None,
            output=result.output,
            cost_usd=result.cost_usd,
            tokens=result.tokens,
            step_count=len(steps),
            approximate_count=result.trajectory.count(StepStatus.APPROXIMATE),
            unconsumed_count=len(result.unconsumed),
            started_at=_utc(started_at),
            duration_ms=duration_ms,
            steps=steps,
            checks=[
                CheckRow(
                    position=i,
                    check=c.check,
                    passed=c.passed,
                    message=c.message,
                    details=c.details,
                    advisory=c.advisory,
                )
                for i, c in enumerate(checks)
            ],
        )
        with self._lock, self._session.begin() as session:
            session.add(row)
        log.debug("store.run_saved", run_id=run_id, scenario=scenario)
        return run_id

    def list_runs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        scenario: str | None = None,
        suite_id: str | None = None,
        mode: str | None = None,
        passed: bool | None = None,
    ) -> list[RunSummary]:
        """Runs newest first."""
        stmt = (
            self._filter_runs(select(RunRow), scenario, suite_id, mode, passed)
            .order_by(RunRow.started_at.desc(), RunRow.id)
            .limit(limit)
            .offset(offset)
        )
        with self._lock, self._session() as session:
            return [_summary(r) for r in session.scalars(stmt)]

    def count_runs(
        self,
        *,
        scenario: str | None = None,
        suite_id: str | None = None,
        mode: str | None = None,
        passed: bool | None = None,
    ) -> int:
        stmt = self._filter_runs(
            select(func.count()).select_from(RunRow), scenario, suite_id, mode, passed
        )
        with self._lock, self._session() as session:
            return session.scalar(stmt) or 0

    @staticmethod
    def _filter_runs[S: Select[Any]](
        stmt: S, scenario: str | None, suite_id: str | None, mode: str | None, passed: bool | None
    ) -> S:
        if scenario is not None:
            stmt = stmt.where(RunRow.scenario == scenario)
        if suite_id is not None:
            stmt = stmt.where(RunRow.suite_id == suite_id)
        if mode is not None:
            stmt = stmt.where(RunRow.mode == mode)
        if passed is not None:
            stmt = stmt.where(RunRow.passed == passed)
        return stmt

    def scenarios(self) -> list[str]:
        """Every scenario name with a stored run, sorted."""
        with self._lock, self._session() as session:
            return list(
                session.scalars(select(RunRow.scenario).distinct().order_by(RunRow.scenario))
            )

    def get_run(self, run_id: str) -> RunRecord:
        """One run with steps and checks; ``run_id`` may be a unique prefix (>= 6 chars)."""
        with self._lock, self._session() as session:
            row = self._by_prefix(session, RunRow, run_id, "run", "List ids with `replay runs`.")
            return _record(row)

    def _by_prefix[R: (RunRow, FlakeReportRow, CalibrationRow)](
        self, session: Session, model: type[R], prefix: str, what: str, hint: str
    ) -> R:
        exact = session.get(model, prefix)
        if exact is not None:
            return exact
        if len(prefix) < MIN_PREFIX:
            raise StoreError(
                f"no {what} with id {prefix!r}; ids may be abbreviated to at least "
                f"{MIN_PREFIX} characters. {hint}"
            )
        matches = session.scalars(
            select(model).where(model.id.like(_like_prefix(prefix), escape="\\")).limit(2)
        ).all()
        if not matches:
            raise StoreError(f"no {what} with id {prefix!r} in the store. {hint}")
        if len(matches) > 1:
            raise StoreError(
                f"{what} id prefix {prefix!r} is ambiguous; use more characters. {hint}"
            )
        return matches[0]

    def list_suites(self, *, limit: int = 20) -> list[SuiteSummary]:
        """Suites newest first, with the number of runs stored under each."""
        stmt = (
            select(SuiteRow, func.count(RunRow.id))
            .outerjoin(RunRow, RunRow.suite_id == SuiteRow.id)
            .group_by(SuiteRow.id)
            .order_by(SuiteRow.started_at.desc(), SuiteRow.id)
            .limit(limit)
        )
        with self._lock, self._session() as session:
            return [_suite_summary(s, n) for s, n in session.execute(stmt)]

    def get_suite(self, suite_id: str) -> SuiteSummary:
        with self._lock, self._session() as session:
            row = session.get(SuiteRow, suite_id)
            if row is None:
                raise StoreError(
                    f"no suite {suite_id!r} in the store. List ids with `replay runs`."
                )
            count = session.scalar(select(func.count()).where(RunRow.suite_id == suite_id)) or 0
            return _suite_summary(row, count)

    def save_flake(self, report: FlakeReport) -> str:
        """Store one ``replay flake`` report with every scenario's estimate."""
        report_id = _new_id()
        row = FlakeReportRow(
            id=report_id,
            suite_id=report.suite_id,
            suite=report.suite,
            stats_impl=report.stats_impl,
            settings=report.settings.model_dump(mode="json"),
            passed=report.passed,
            started_at=_utc(report.started_at),
            duration_ms=report.duration_ms,
            git_sha=report.git_sha,
            scenario_count=len(report.scenarios),
            failed_count=len(report.failed),
            flaky_count=len(report.flaky),
            run_count=report.runs,
            cost_usd=report.cost_usd,
            scenarios=[
                FlakeScenarioRow(
                    position=i,
                    name=s.name,
                    cassette=s.cassette,
                    agent=s.agent,
                    mode=s.mode,
                    policy=s.policy,
                    classification=s.classification,
                    decision=s.decision,
                    runs=s.runs,
                    passes=s.passes,
                    pass_rate=s.pass_rate,
                    ci_low=s.interval[0],
                    ci_high=s.interval[1],
                    failures=dict(s.failures),
                    executed=s.executed,
                    concurrency=s.concurrency,
                    cost_usd=s.cost_usd,
                    error=s.error.model_dump(mode="json") if s.error else None,
                    outcomes=[o.model_dump(mode="json") for o in s.outcomes],
                )
                for i, s in enumerate(report.scenarios)
            ],
        )
        with self._lock, self._session.begin() as session:
            session.add(row)
        log.debug("store.flake_saved", report_id=report_id, suite=report.suite)
        return report_id

    def list_flake_reports(self, *, limit: int = 20) -> list[FlakeReportSummary]:
        """``replay flake`` reports newest first."""
        stmt = (
            select(FlakeReportRow)
            .order_by(FlakeReportRow.started_at.desc(), FlakeReportRow.id)
            .limit(limit)
        )
        with self._lock, self._session() as session:
            return [_flake_summary(r) for r in session.scalars(stmt)]

    def get_flake_report(self, report_id: str) -> FlakeReportRecord:
        """One flake report with its scenarios; ``report_id`` may be a unique prefix."""
        with self._lock, self._session() as session:
            row = self._by_prefix(
                session, FlakeReportRow, report_id, "flake report", "List them on /api/flake."
            )
            return FlakeReportRecord(
                **_flake_summary(row).model_dump(),
                settings=FlakeSettings.model_validate(row.settings),
                scenarios=[FlakeScenarioRecord.model_validate(s) for s in row.scenarios],
            )

    def flake_trend(self, *, scenario: str | None = None, limit: int = 500) -> list[FlakePoint]:
        """Each scenario's estimate per flake report, oldest first, over the newest ``limit``
        points."""
        stmt = (
            select(FlakeScenarioRow, FlakeReportRow)
            .join(FlakeReportRow, FlakeScenarioRow.report_id == FlakeReportRow.id)
            .order_by(
                FlakeReportRow.started_at.desc(),
                FlakeReportRow.id.desc(),
                FlakeScenarioRow.position.desc(),
            )
            .limit(limit)
        )
        if scenario is not None:
            stmt = stmt.where(FlakeScenarioRow.name == scenario)
        with self._lock, self._session() as session:
            points = [
                FlakePoint(
                    report_id=r.id,
                    started_at=_utc(r.started_at),
                    stats_impl=r.stats_impl,
                    git_sha=r.git_sha,
                    name=s.name,
                    classification=s.classification,
                    decision=s.decision,
                    runs=s.runs,
                    passes=s.passes,
                    pass_rate=s.pass_rate,
                    ci_low=s.ci_low,
                    ci_high=s.ci_high,
                )
                for s, r in session.execute(stmt)
            ]
        points.reverse()
        return points

    def save_calibration(self, report: CalibrationReport) -> str:
        """Store a calibration report; the dashboard's calibration page reads these."""
        report_id = _new_id()
        with self._lock, self._session.begin() as session:
            session.add(
                CalibrationRow(
                    id=report_id,
                    rubric=report.rubric,
                    rubric_fingerprint=report.rubric_fingerprint,
                    judge_id=report.judge_id,
                    prompt_version=report.prompt_version,
                    created_at=_utc(report.created_at),
                    stored_at=self._clock(),
                    n_items=report.n_items,
                    report=report.model_dump(mode="json"),
                )
            )
        log.debug("store.calibration_saved", report_id=report_id, rubric=report.rubric)
        return report_id

    def list_calibrations(self, *, limit: int = 50) -> list[CalibrationSummary]:
        """Calibration reports, most recently created first."""
        stmt = (
            select(CalibrationRow)
            .order_by(CalibrationRow.created_at.desc(), CalibrationRow.id)
            .limit(limit)
        )
        with self._lock, self._session() as session:
            return [_calibration_summary(r) for r in session.scalars(stmt)]

    def get_calibration(self, report_id: str) -> CalibrationRecord:
        with self._lock, self._session() as session:
            row = self._by_prefix(
                session,
                CalibrationRow,
                report_id,
                "calibration report",
                "List them on /api/calibration.",
            )
            return CalibrationRecord(
                id=row.id,
                stored_at=_utc(row.stored_at),
                report=CalibrationReport.model_validate(row.report),
            )


_LEGACY_COLUMNS = {"check_results": ("advisory",)}
"""Columns added to stores created before Alembic, by the add-on-open code that preceded it."""

_PG_MIGRATION_LOCK = 0x7265706C6179
"""Advisory lock key ("replay") held while a PostgreSQL store migrates."""


def _alembic_config(connection: Connection | None = None) -> AlembicConfig:
    config = AlembicConfig()
    config.set_main_option("script_location", "agent_replay:migrations")
    config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    """The newest schema revision this version of the package knows."""
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    assert head is not None
    return head


def current_revision(engine: Engine) -> str | None:
    """The schema revision a store is at; ``None`` before its first migration."""
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def migrate(engine: Engine) -> None:
    """Upgrade the store behind ``engine`` to the newest revision, under a lock.

    A store made before migrations existed (tables but no ``alembic_version``)
    gains the columns added since it was created and is stamped at ``0001``
    rather than having its tables created again.
    """
    backend = engine.dialect.name
    file_sqlite = backend == "sqlite" and engine.url.database not in (None, "", ":memory:")
    lock_engine = _immediate_engine(engine.url) if file_sqlite else engine
    try:
        with lock_engine.begin() as conn:
            if backend == "postgresql":
                lock = text("SELECT pg_advisory_xact_lock(:key)")
                conn.execute(lock, {"key": _PG_MIGRATION_LOCK})
            config = _alembic_config(conn)
            tables = set(inspect(conn).get_table_names())
            if "alembic_version" not in tables and tables and "runs" not in tables:
                raise StoreError(
                    f"the run store has tables ({', '.join(sorted(tables))}) but neither a "
                    "schema version nor a 'runs' table, so it is not a store this version can "
                    "adopt; delete it, or set REPLAY_DB_URL to another location"
                )
            if "runs" in tables and "alembic_version" not in tables:
                _upgrade_legacy(conn)
                command.stamp(config, "0001")
                log.info("store.legacy_stamped", revision="0001")
            before = MigrationContext.configure(conn).get_current_revision()
            _check_known(config, before)
            command.upgrade(config, "head")
            after = MigrationContext.configure(conn).get_current_revision()
            if before != after:
                log.info("store.migrated", backend=backend, before=before, after=after)
    finally:
        if lock_engine is not engine:
            lock_engine.dispose()


def _check_known(config: AlembicConfig, revision: str | None) -> None:
    if revision is None:
        return
    try:
        ScriptDirectory.from_config(config).get_revision(revision)
    except CommandError:
        raise StoreError(
            f"the run store is at schema revision {revision!r}, which this version of "
            "agent-replay does not know (a newer version wrote it); upgrade agent-replay, or "
            "set REPLAY_DB_URL to another store"
        ) from None


def _immediate_engine(url: URL) -> Engine:
    """A SQLite engine whose transactions begin with ``BEGIN IMMEDIATE``.

    pysqlite would otherwise defer ``BEGIN`` until the first write, so two
    processes could both read "no tables yet" before either takes the lock.
    """
    engine = create_engine(url, poolclass=NullPool, connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _autocommit_driver(dbapi_connection: Any, _record: object) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


def _upgrade_legacy(conn: Connection) -> None:
    for table_name, columns in _LEGACY_COLUMNS.items():
        present = {c["name"] for c in inspect(conn).get_columns(table_name)}
        for name in columns:
            if name in present:
                continue
            column = Base.metadata.tables[table_name].columns[name]
            ddl = CreateColumn(column).compile(dialect=conn.dialect)
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))
            log.info("store.column_added", table=table_name, column=name)


def _ensure_parent(url: URL) -> None:
    if url.database:
        Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _sqlite_pragmas(dbapi_connection: Any, _record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
