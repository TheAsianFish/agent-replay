"""The dashboard API: every endpoint over a store filled by real ``replay ci`` and
``replay flake`` runs, errors, request logging, the dashboard mount and ``replay serve``.

The API cases also run on PostgreSQL when ``REPLAY_TEST_PG_URL`` is set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from structlog.testing import capture_logs
from typer.testing import CliRunner

from agent_replay import cli
from agent_replay._logging import configure_logging
from agent_replay.cli import app as cli_app
from agent_replay.judge.calibration import CalibrationReport, CriterionCalibration
from agent_replay.server import create_app
from agent_replay.server.openapi import schema_json
from agent_replay.store import Store

ROOT = Path(__file__).parents[1]
SUITE = str(ROOT / "examples" / "retail_agent" / "replay.toml")
FLAKY = "examples.retail_agent.flaky_model:make_client"
BOOTS = "exchange_waterproof_boots"
PG_URL = os.environ.get("REPLAY_TEST_PG_URL")
ENV = {"COLUMNS": "200"}
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

runner = CliRunner()


def _fill(db: str) -> None:
    passing = runner.invoke(cli_app, ["ci", SUITE, "--db", db], env=ENV)
    assert passing.exit_code == 0, passing.output
    mutant = runner.invoke(
        cli_app,
        ["ci", SUITE, "--db", db, "--agent", "examples.retail_agent.mutants:skips_order_lookup"],
        env=ENV,
    )
    assert mutant.exit_code == 1, mutant.output
    flake = runner.invoke(
        cli_app,
        [
            "flake",
            SUITE,
            "--db",
            db,
            "--client",
            FLAKY,
            "--max-runs",
            "4",
            "-s",
            "exchange_waterproof_boots",
        ],
        env=ENV,
    )
    assert flake.exit_code in (0, 1), flake.output
    with Store.open(db) as store:
        store.save_calibration(
            CalibrationReport(
                rubric="retail",
                rubric_fingerprint="f" * 16,
                judge_id="fake:default",
                prompt_version="1",
                created_at=T0,
                n_items=24,
                method="test",
                criteria={
                    "resolved": CriterionCalibration(
                        kappa=0.8, ci_low=0.65, ci_high=0.95, agreement=0.92, n=24
                    ),
                    "policy": CriterionCalibration(
                        kappa=0.5, ci_low=0.2, ci_high=0.8, agreement=0.8, n=24
                    ),
                },
            )
        )


def _reset_pg(url: str) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


@pytest.fixture(
    scope="module", params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres)]
)
def db(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path_factory.mktemp('server') / 'r.db'}"
    else:
        if not PG_URL:
            pytest.skip("set REPLAY_TEST_PG_URL to run the API tests on PostgreSQL")
        _reset_pg(PG_URL)
        url = PG_URL
    with pytest.MonkeyPatch.context() as mp:
        # The module-scoped fill runs outside the per-test network guard; it never needs one.
        mp.setenv("REPLAY_LOG_LEVEL", "WARNING")
        _fill(url)
    yield url
    if request.param == "postgresql" and PG_URL:
        _reset_pg(PG_URL)


@pytest.fixture
def client(db: str) -> Iterator[TestClient]:
    with TestClient(create_app(db_url=db)) as c:
        yield c


def get(client: TestClient, path: str, **params: Any) -> Any:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_names_the_backend_and_revision(client: TestClient, db: str):
    body = get(client, "/api/health")
    assert body["status"] == "ok"
    assert body["backend"] == db.split(":", maxsplit=1)[0].split("+", maxsplit=1)[0]
    assert body["revision"] == "0002"


def test_runs_page_filters_and_counts(client: TestClient):
    everything = get(client, "/api/runs", limit=500)
    assert everything["total"] == len(everything["items"]) > 0
    failed = get(client, "/api/runs", passed=False, limit=500)
    assert failed["total"] > 0
    assert all(not r["passed"] for r in failed["items"])
    page = get(client, "/api/runs", limit=2, offset=1)
    assert [r["id"] for r in page["items"]] == [r["id"] for r in everything["items"][1:3]]
    scenarios = get(client, "/api/scenarios")
    assert scenarios == sorted({r["scenario"] for r in everything["items"]})
    one = get(client, "/api/runs", scenario=scenarios[0], limit=500)
    assert {r["scenario"] for r in one["items"]} == {scenarios[0]}


def test_run_detail_has_steps_and_checks_and_accepts_a_prefix(client: TestClient):
    run_id = get(client, "/api/runs", limit=1)["items"][0]["id"]
    run = get(client, f"/api/runs/{run_id[:8]}")
    assert run["id"] == run_id
    assert run["steps"]
    assert run["checks"]
    assert {"kind", "name", "status", "request", "response"} <= set(run["steps"][0])


def test_unknown_run_is_a_404_that_names_the_fix(client: TestClient):
    response = client.get("/api/runs/doesnotexist")
    assert response.status_code == 404
    assert "List ids with `replay runs`" in response.json()["detail"]


def test_bad_query_is_a_422(client: TestClient):
    assert client.get("/api/runs", params={"limit": 0}).status_code == 422
    assert client.get("/api/runs", params={"passed": "maybe"}).status_code == 422


def test_unknown_api_path_is_a_json_404(client: TestClient):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "no API endpoint /api/nope"}


def test_diff_aligns_a_passing_run_with_its_regression(client: TestClient):
    scenario = "cancel_pending_laptop"
    runs = get(client, "/api/runs", scenario=scenario, limit=500)["items"]
    good = next(r for r in runs if r["passed"])
    bad = next(r for r in runs if not r["passed"])
    body = get(client, "/api/diff", base=good["id"], head=bad["id"])
    assert body["base"]["id"] == good["id"]
    assert body["head"]["id"] == bad["id"]
    ops = [row["op"] for row in body["diff"]["rows"]]
    assert "removed" in ops  # the mutant skips the order lookup
    removed = [row["base"]["name"] for row in body["diff"]["rows"] if row["op"] == "removed"]
    assert "get_order_details" in removed
    assert body["diff"]["distance"] > 0


def test_diff_of_a_run_with_itself_is_all_same(client: TestClient):
    run_id = get(client, "/api/runs", limit=1)["items"][0]["id"]
    body = get(client, "/api/diff", base=run_id, head=run_id)
    assert {row["op"] for row in body["diff"]["rows"]} == {"same"}
    assert body["diff"]["distance"] == 0


def test_suites_list_and_detail(client: TestClient):
    suites = get(client, "/api/suites")
    assert len(suites) == 3  # two replay ci runs and one replay flake
    suite = get(client, f"/api/suites/{suites[0]['id']}")
    assert suite == suites[0]
    runs = get(client, "/api/runs", suite_id=suite["id"], limit=500)
    assert runs["total"] == suite["run_count"]
    assert client.get("/api/suites/missing").status_code == 404


def test_flake_reports_and_trend(client: TestClient):
    (summary,) = get(client, "/api/flake")
    report = get(client, f"/api/flake/{summary['id']}")
    (scenario,) = report["scenarios"]
    assert scenario["name"] == "exchange_waterproof_boots"
    assert 0 <= scenario["ci_low"] <= scenario["ci_high"] <= 1
    assert report["settings"]["max_runs"] == 4
    trend = get(client, "/api/flake/trend", scenario="exchange_waterproof_boots")
    assert [p["report_id"] for p in trend] == [summary["id"]]
    assert get(client, "/api/flake/trend", scenario="unknown") == []


def test_calibration_view_marks_which_criteria_gate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    (summary,) = get(client, "/api/calibration")
    view = get(client, f"/api/calibration/{summary['id']}")
    assert view["min_kappa"] == 0.6
    assert view["gates"] == {"resolved": {"gating": True}, "policy": {"gating": False}}
    assert view["record"]["report"]["criteria"]["resolved"]["kappa"] == 0.8
    monkeypatch.setenv("REPLAY_JUDGE_MIN_KAPPA", "0.7")
    assert get(client, f"/api/calibration/{summary['id']}")["gates"]["resolved"] == {
        "gating": False
    }


def test_each_request_is_logged_once_with_its_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("REPLAY_LOG_LEVEL", "INFO")
    configure_logging()
    with capture_logs() as logs:
        response = client.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    (entry,) = [e for e in logs if e["event"] == "http.request"]
    assert entry["path"] == "/api/health"
    assert entry["status"] == 200
    assert entry["duration_ms"] >= 0
    generated = client.get("/api/health").headers["X-Request-ID"]
    assert len(generated) == 16


def test_json_request_logs_carry_the_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("REPLAY_LOG_LEVEL", "INFO")
    monkeypatch.setenv("REPLAY_LOG_FORMAT", "json")
    configure_logging()
    capsys.readouterr()
    client.get("/api/runs", params={"limit": 1}, headers={"X-Request-ID": "req-42"})
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line]
    (entry,) = [e for e in lines if e["event"] == "http.request"]
    assert entry["request_id"] == "req-42"
    assert entry["level"] == "info"
    assert entry["path"] == "/api/runs"


def test_the_schema_on_disk_is_current():
    committed = (ROOT / "dashboard" / "openapi.json").read_text(encoding="utf-8")
    assert committed == schema_json(), (
        "dashboard/openapi.json is stale; run `make openapi` and commit it with the regenerated "
        "dashboard types"
    )


def test_dashboard_mount_serves_assets_and_falls_back_to_index(tmp_path: Path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("no", encoding="utf-8")
    store = Store.open("sqlite://")
    with TestClient(create_app(store, static_dir=tmp_path)) as client:
        assert client.get("/").text == "<div id=root></div>"
        assert client.get("/runs/abc").text == "<div id=root></div>"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/favicon.svg").text == "<svg/>"
        assert "no" not in client.get("/..%2Fsecret.txt").text
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/nope").status_code == 404
    store.close()


def test_replay_serve_opens_the_store_and_runs_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, Any]] = []

    def fake_run(app: object, **kwargs: Any) -> None:
        with TestClient(app) as c:  # type: ignore[arg-type]
            kwargs["health"] = c.get("/api/health").json()
        calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.delenv("REPLAY_DASHBOARD_DIR", raising=False)
    monkeypatch.setattr("agent_replay.server.app.PACKAGED_DASHBOARD", tmp_path / "none")
    db = f"sqlite:///{tmp_path / 'r.db'}"
    result = runner.invoke(cli.app, ["serve", "--db", db, "--port", "8123"], env=ENV)
    assert result.exit_code == 0, result.output
    assert "serving the sqlite run store on http://127.0.0.1:8123" in result.output
    assert "API only" in result.output
    (call,) = calls
    assert call["port"] == 8123
    assert call["health"]["revision"] == "0002"


def test_replay_serve_rejects_a_dashboard_without_index(tmp_path: Path):
    result = runner.invoke(
        cli.app, ["serve", "--db", "sqlite://", "--dashboard", str(tmp_path)], env=ENV
    )
    assert result.exit_code == 2
    assert "has no index.html; build the dashboard with `make ui` first" in result.output
