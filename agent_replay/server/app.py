"""FastAPI app: JSON endpoints under ``/api`` and, when built, the React dashboard at ``/``.

Every endpoint is a read over :class:`agent_replay.store.Store`; nothing here
writes. Each request is logged once as structured JSON (``REPLAY_LOG_FORMAT``)
with a request id that is also returned in ``X-Request-ID``.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from agent_replay import __version__
from agent_replay.diff import TraceDiff, diff_runs
from agent_replay.errors import StoreError
from agent_replay.judge.calibration import resolve_min_kappa
from agent_replay.store import (
    CalibrationRecord,
    CalibrationSummary,
    FlakePoint,
    FlakeReportRecord,
    FlakeReportSummary,
    RunRecord,
    RunSummary,
    Store,
    SuiteSummary,
)

log = structlog.get_logger("agent_replay.server")

PACKAGED_DASHBOARD = Path(__file__).parent / "static"
MAX_PAGE = 500


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class Health(_Model):
    status: str
    version: str
    backend: str
    revision: str | None


class RunPage(_Model):
    total: int
    items: list[RunSummary]


class RunDiff(_Model):
    base: RunSummary
    head: RunSummary
    diff: TraceDiff


class CriterionGateView(_Model):
    gating: bool
    """Whether ``ci_low`` reaches ``min_kappa``: the criterion could fail a build."""


class CalibrationView(_Model):
    record: CalibrationRecord
    min_kappa: float
    """``REPLAY_JUDGE_MIN_KAPPA`` (default 0.6) on the server."""
    gates: dict[str, CriterionGateView]


def dashboard_dir() -> Path | None:
    """``$REPLAY_DASHBOARD_DIR``, else the build packaged with the server, if present."""
    raw = os.environ.get("REPLAY_DASHBOARD_DIR")
    path = Path(raw) if raw else PACKAGED_DASHBOARD
    return path if (path / "index.html").is_file() else None


def _store(request: Request) -> Store:
    store: Store = request.app.state.store
    return store


StoreDep = Annotated[Store, Depends(_store)]


def create_app(
    store: Store | None = None,
    *,
    db_url: str | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the app over ``store``, or over a store opened from ``db_url`` at startup.

    ``static_dir`` holds a built dashboard (``index.html`` and assets); without
    one only the API is served.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = store is None
        app.state.store = Store.open(db_url) if store is None else store
        log.info("server.started", backend=app.state.store.backend, dashboard=bool(static_dir))
        try:
            yield
        finally:
            if owned:
                app.state.store.close()

    app = FastAPI(
        title="agent-replay",
        version=__version__,
        summary="Read-only API over the agent-replay run store.",
        lifespan=lifespan,
    )
    if store is not None:
        app.state.store = store

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                log.exception("http.error", method=request.method, path=request.url.path)
                raise
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(StoreError)
    async def not_found(_request: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/api/health", tags=["meta"])
    def health(store: StoreDep) -> Health:
        return Health(
            status="ok", version=__version__, backend=store.backend, revision=store.revision()
        )

    @app.get("/api/runs", tags=["runs"])
    def list_runs(
        store: StoreDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        scenario: str | None = None,
        suite_id: str | None = None,
        mode: str | None = None,
        passed: bool | None = None,
    ) -> RunPage:
        return RunPage(
            total=store.count_runs(scenario=scenario, suite_id=suite_id, mode=mode, passed=passed),
            items=store.list_runs(
                limit=limit,
                offset=offset,
                scenario=scenario,
                suite_id=suite_id,
                mode=mode,
                passed=passed,
            ),
        )

    @app.get("/api/runs/{run_id}", tags=["runs"])
    def get_run(store: StoreDep, run_id: str) -> RunRecord:
        return store.get_run(run_id)

    @app.get("/api/scenarios", tags=["runs"])
    def scenarios(store: StoreDep) -> list[str]:
        return store.scenarios()

    @app.get("/api/diff", tags=["runs"])
    def diff(store: StoreDep, base: str, head: str) -> RunDiff:
        base_run, head_run = store.get_run(base), store.get_run(head)
        return RunDiff(
            base=RunSummary.model_validate(base_run.model_dump()),
            head=RunSummary.model_validate(head_run.model_dump()),
            diff=diff_runs(base_run, head_run),
        )

    @app.get("/api/suites", tags=["suites"])
    def list_suites(
        store: StoreDep, limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50
    ) -> list[SuiteSummary]:
        return store.list_suites(limit=limit)

    @app.get("/api/suites/{suite_id}", tags=["suites"])
    def get_suite(store: StoreDep, suite_id: str) -> SuiteSummary:
        return store.get_suite(suite_id)

    @app.get("/api/flake", tags=["flake"])
    def list_flake(
        store: StoreDep, limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50
    ) -> list[FlakeReportSummary]:
        return store.list_flake_reports(limit=limit)

    @app.get("/api/flake/trend", tags=["flake"])
    def flake_trend(
        store: StoreDep,
        scenario: str | None = None,
        limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
    ) -> list[FlakePoint]:
        return store.flake_trend(scenario=scenario, limit=limit)

    @app.get("/api/flake/{report_id}", tags=["flake"])
    def get_flake(store: StoreDep, report_id: str) -> FlakeReportRecord:
        return store.get_flake_report(report_id)

    @app.get("/api/calibration", tags=["calibration"])
    def list_calibrations(store: StoreDep) -> list[CalibrationSummary]:
        return store.list_calibrations()

    @app.get("/api/calibration/{report_id}", tags=["calibration"])
    def get_calibration(store: StoreDep, report_id: str) -> CalibrationView:
        record = store.get_calibration(report_id)
        min_kappa = resolve_min_kappa(None)
        return CalibrationView(
            record=record,
            min_kappa=min_kappa,
            gates={
                name: CriterionGateView(gating=c.ci_low >= min_kappa)
                for name, c in record.report.criteria.items()
            },
        )

    @app.get("/api/{rest:path}", include_in_schema=False)
    def unknown_api(rest: str) -> None:
        raise HTTPException(status_code=404, detail=f"no API endpoint /api/{rest}")

    if static_dir is not None:
        _mount_dashboard(app, static_dir)
    return app


def _mount_dashboard(app: FastAPI, static_dir: Path) -> None:
    """Serve the built dashboard; any other path gets ``index.html`` (client-side routes)."""
    index = static_dir / "index.html"
    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)
