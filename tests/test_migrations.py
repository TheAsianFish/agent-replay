"""Alembic migrations: the models match the migrated schema, older stores are adopted,
concurrent first opens are safe, and a failed migration leaves the store untouched.

The PostgreSQL cases run when ``REPLAY_TEST_PG_URL`` names a database this suite may
reset (CI starts one as a service); they are skipped otherwise.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from agent_replay import Cassette
from agent_replay import store as store_module
from agent_replay.checks import tool_sequence
from agent_replay.errors import StoreError
from agent_replay.runner import replay
from agent_replay.store import Base, Store, current_revision, head_revision
from examples.toy_agent import agent as toy

CASSETTE_FILE = (
    Path(__file__).parents[1] / "examples" / "toy_agent" / "cassettes" / "refund_eligible.jsonl"
)
PG_URL = os.environ.get("REPLAY_TEST_PG_URL")
LEGACY_TABLES = ("suites", "runs", "steps", "check_results")
T0 = datetime(2026, 10, 10, tzinfo=UTC)


def _save(store: Store) -> str:
    result = replay(Cassette.load(CASSETTE_FILE), toy.run)
    checks = result.check(tool_sequence(["lookup_order", "get_refund_policy", "issue_refund"]))
    return store.save_run(
        result,
        checks,
        scenario="refund_eligible",
        cassette_path=str(CASSETTE_FILE),
        agent="examples.toy_agent.agent:run",
        passed=True,
        started_at=T0,
        duration_ms=1.0,
    )


def _reset_pg(url: str) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


@pytest.fixture(params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgres)])
def url(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str]:
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'm.db'}"
        return
    if not PG_URL:
        pytest.skip("set REPLAY_TEST_PG_URL to run the PostgreSQL migration tests")
    _reset_pg(PG_URL)
    yield PG_URL
    _reset_pg(PG_URL)


def _diffs(engine: Engine) -> list[object]:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        return list(compare_metadata(context, Base.metadata))


def test_a_new_store_is_at_head_and_matches_the_models(url: str):
    with Store.open(url) as store:
        assert current_revision(store._engine) == head_revision()
        assert _diffs(store._engine) == []
        _save(store)
        assert store.count_runs() == 1


def test_reopening_is_a_no_op(url: str):
    with Store.open(url) as store:
        run_id = _save(store)
    with Store.open(url) as store:
        assert store.get_run(run_id).id == run_id
        assert current_revision(store._engine) == head_revision()


def test_downgrade_to_base_and_back(url: str):
    with Store.open(url) as store:
        engine = store._engine
        with engine.begin() as conn:
            config = store_module._alembic_config(conn)
            command.downgrade(config, "base")
        assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    with Store.open(url) as store:
        assert _diffs(store._engine) == []


def _make_legacy(url: str, *, drop_advisory: bool) -> str:
    """A store as ``create_all`` made it before migrations: 0001's tables, no version."""
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[t] for t in LEGACY_TABLES])
    engine.dispose()
    with Store(create_engine(url)) as store:
        run_id = _save(store)
    engine = create_engine(url)
    with engine.begin() as conn:
        # 0003 adds it; a store from before migrations never had it.
        conn.execute(text("ALTER TABLE runs DROP COLUMN unpriced_calls"))
        if drop_advisory:
            conn.execute(text("ALTER TABLE check_results DROP COLUMN advisory"))
    engine.dispose()
    return run_id


@pytest.mark.parametrize("drop_advisory", [True, False])
def test_an_older_store_is_adopted(url: str, drop_advisory: bool):
    run_id = _make_legacy(url, drop_advisory=drop_advisory)
    with Store.open(url) as store:
        assert current_revision(store._engine) == head_revision()
        assert _diffs(store._engine) == []
        old = store.get_run(run_id)
        assert [c.advisory for c in old.checks] == [False]
        assert old.unpriced_calls == 0
        _save(store)
        assert store.count_runs() == 2


def test_a_failed_migration_leaves_the_store_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    real = command.upgrade

    def upgrade_then_fail(config: object, revision: str) -> None:
        real(config, revision)  # type: ignore[arg-type]
        raise RuntimeError("disk full")

    monkeypatch.setattr(store_module.command, "upgrade", upgrade_then_fail)
    with pytest.raises(RuntimeError, match="disk full"):
        Store.open(url)
    engine = create_engine(url)
    assert inspect(engine).get_table_names() == []
    engine.dispose()


def test_a_half_made_legacy_store_is_refused_with_the_fix(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["suites"]])
    engine.dispose()
    with pytest.raises(StoreError, match="neither a schema version nor a 'runs' table"):
        Store.open(url)


def test_a_store_from_a_newer_version_names_the_fix(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    Store.open(url).close()
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '9999'"))
    engine.dispose()
    with pytest.raises(StoreError, match="9999"):
        Store.open(url)


def _open_and_save(url: str) -> str:
    with Store.open(url) as store:
        _save(store)
        return current_revision(store._engine) or ""


@pytest.mark.parametrize("legacy", [False, True])
def test_concurrent_first_opens_do_not_collide(url: str, legacy: bool):
    if legacy:
        _make_legacy(url, drop_advisory=True)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        revisions = pool.map(_open_and_save, [url] * 8)
    assert revisions == [head_revision()] * 8
    with Store.open(url) as store:
        assert store.count_runs() == 8 + legacy
        assert _diffs(store._engine) == []
