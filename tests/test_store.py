"""The run store: round trips, ordering, id prefixes, URL resolution, integrity."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from agent_replay import Cassette
from agent_replay.cassette import ErrorInfo
from agent_replay.checks import CheckResult, tool_sequence
from agent_replay.errors import ReplayError, StoreError
from agent_replay.runner import RunResult, replay
from agent_replay.store import Store
from agent_replay.trajectory import Step, StepStatus, Trajectory
from examples.toy_agent import agent as toy

CASSETTE_PATH = Path(__file__).parents[1] / "examples" / "toy_agent" / "cassettes"
CASSETTE_FILE = CASSETTE_PATH / "refund_eligible.jsonl"
T0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    return T0


@pytest.fixture
def cassette() -> Cassette:
    return Cassette.load(CASSETTE_FILE)


@pytest.fixture
def result(cassette: Cassette) -> RunResult:
    return replay(cassette, toy.run)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(f"sqlite:///{tmp_path / 'r.db'}", clock=fixed_clock) as s:
        yield s


def save(
    store: Store,
    result: RunResult,
    *,
    scenario: str = "refund_eligible",
    started_at: datetime = T0,
    suite_id: str | None = None,
) -> str:
    checks = result.check(tool_sequence(["lookup_order", "get_refund_policy", "issue_refund"]))
    return store.save_run(
        result,
        checks,
        scenario=scenario,
        cassette_path=str(CASSETTE_FILE),
        agent="examples.toy_agent.agent:run",
        passed=all(c.passed for c in checks),
        started_at=started_at,
        duration_ms=12.5,
        suite_id=suite_id,
    )


def test_round_trip_run_steps_and_checks(store: Store, result: RunResult, cassette: Cassette):
    run_id = save(store, result)

    run = store.get_run(run_id)

    assert run.id == run_id
    assert run.scenario == "refund_eligible"
    assert run.agent == "examples.toy_agent.agent:run"
    assert run.cassette_id == cassette.header.cassette_id
    assert run.mode == "full"
    assert run.policy == "strict"
    assert run.task == result.task
    assert run.passed is True
    assert run.error_type is None
    assert run.error_message is None
    assert run.output == result.output
    assert run.started_at == T0
    assert run.duration_ms == 12.5
    assert run.step_count == len(result.trajectory.steps) > 0
    assert run.approximate_count == 0
    assert run.unconsumed_count == 0

    assert [s.index for s in run.steps] == list(range(run.step_count))
    by_seq = {e.seq: e for e in cassette.envelopes}
    for stored, step in zip(run.steps, result.trajectory.steps, strict=True):
        assert stored.kind == step.kind.value
        assert stored.name == step.name
        assert stored.status == step.status.value
        assert stored.fingerprint == step.fingerprint
        assert stored.request == step.request
        assert stored.response == step.response
        assert stored.error is None
        assert stored.envelope_seq == step.envelope_seq
        assert step.envelope_seq is not None
        env = by_seq[step.envelope_seq]
        assert env.usage is not None or stored.input_tokens is None
        if env.usage is not None:
            assert stored.input_tokens == env.usage.input_tokens
            assert stored.output_tokens == env.usage.output_tokens
            assert stored.cache_read_input_tokens == env.usage.cache_read_input_tokens
            assert stored.cache_creation_input_tokens == env.usage.cache_creation_input_tokens
        assert stored.cost_usd == env.cost_usd
        assert stored.latency_ms == env.latency_ms
        assert stored.span_id == step.span_id
        assert stored.parent_span_id == step.parent_span_id
    assert any(s.input_tokens for s in run.steps)
    assert (
        run.tokens
        == result.tokens
        == sum((s.input_tokens or 0) + (s.output_tokens or 0) for s in run.steps)
    )

    [check] = run.checks
    assert check.position == 0
    assert check.check == "tool_sequence"
    assert check.passed is True
    assert (
        check.details
        == result.check(tool_sequence(["lookup_order", "get_refund_policy", "issue_refund"]))[
            0
        ].details
    )


def test_checks_keep_their_order_and_details(store: Store, result: RunResult):
    checks = result.check(
        tool_sequence(["lookup_order", "get_refund_policy", "issue_refund"]),
        tool_sequence(["issue_refund"]),
    )
    run_id = store.save_run(
        result,
        checks,
        scenario="s",
        cassette_path="c",
        agent="a",
        passed=False,
        started_at=T0,
        duration_ms=1,
    )

    run = store.get_run(run_id)

    assert [c.position for c in run.checks] == [0, 1]
    assert [c.passed for c in run.checks] == [c.passed for c in checks]
    assert [c.message for c in run.checks] == [c.message for c in checks]
    assert [c.details for c in run.checks] == [c.details for c in checks]


def test_failed_run_stores_error(store: Store, cassette: Cassette):
    def crashing(client: object, task: str) -> object:
        raise KeyError("boom")

    failed = replay(cassette, crashing)
    run_id = store.save_run(
        failed,
        [],
        scenario="s",
        cassette_path="c",
        agent="a",
        passed=False,
        started_at=T0,
        duration_ms=1,
    )

    run = store.get_run(run_id)

    assert run.passed is False
    assert run.error_type == "KeyError"
    assert run.error_message is not None
    assert "boom" in run.error_message
    assert run.unconsumed_count == len(cassette.envelopes)


def test_step_error_round_trips(store: Store, cassette: Cassette):
    step = Step(
        index=0,
        kind=cassette.envelopes[0].kind,
        name="x",
        status=StepStatus.UNRECORDED,
        fingerprint="f",
        request={"a": [1, {"b": None}]},
        error=ErrorInfo(type="E", module="m", message="bad"),
    )
    result = RunResult(mode="full", task="t", trajectory=Trajectory(steps=(step,)))
    run_id = store.save_run(
        result,
        [],
        scenario="s",
        cassette_path="c",
        agent="a",
        passed=False,
        started_at=T0,
        duration_ms=0,
    )

    [stored] = store.get_run(run_id).steps

    assert stored.error == {"type": "E", "module": "m", "message": "bad"}
    assert stored.request == {"a": [1, {"b": None}]}
    assert stored.response is None


def test_list_runs_newest_first_with_filters(store: Store, result: RunResult):
    suite = store.start_suite(config_path="replay.toml")
    a = save(store, result, scenario="a", started_at=T0)
    b = save(store, result, scenario="b", started_at=T0 + timedelta(minutes=2), suite_id=suite)
    c = save(store, result, scenario="a", started_at=T0 + timedelta(minutes=1), suite_id=suite)

    assert [r.id for r in store.list_runs()] == [b, c, a]
    assert [r.id for r in store.list_runs(limit=2)] == [b, c]
    assert [r.id for r in store.list_runs(scenario="a")] == [c, a]
    assert [r.id for r in store.list_runs(suite_id=suite)] == [b, c]
    assert [r.id for r in store.list_runs(scenario="a", suite_id=suite)] == [c]
    assert store.list_runs(scenario="nope") == []
    assert all(r.started_at.tzinfo is not None for r in store.list_runs())


def test_suites_start_finish_and_run_counts(store: Store, result: RunResult):
    first = store.start_suite(config_path="a.toml", git_sha="abc123")
    [pending] = store.list_suites()
    assert pending.id == first
    assert pending.started_at == T0
    assert pending.finished_at is None
    assert pending.passed is None
    assert pending.git_sha == "abc123"
    assert pending.run_count == 0

    save(store, result, suite_id=first)
    save(store, result, suite_id=first)
    store.finish_suite(first, passed=False, scenario_count=2, failed_count=1)
    second = store.start_suite(config_path="b.toml")
    store.finish_suite(second, passed=True, scenario_count=0, failed_count=0)

    suites = {s.id: s for s in store.list_suites()}
    done = suites[first]
    assert done.finished_at == T0
    assert done.passed is False
    assert (done.scenario_count, done.failed_count, done.run_count) == (2, 1, 2)
    assert suites[second].passed is True
    assert suites[second].run_count == 0
    assert suites[second].git_sha is None
    assert len(store.list_suites(limit=1)) == 1


def test_finish_unknown_suite_names_the_fix(store: Store):
    with pytest.raises(StoreError, match="start_suite"):
        store.finish_suite("nope", passed=True, scenario_count=0, failed_count=0)


def test_suites_list_newest_first(tmp_path: Path):
    times = iter([T0, T0 + timedelta(hours=1)])
    with Store.open(f"sqlite:///{tmp_path / 'r.db'}", clock=lambda: next(times)) as s:
        old = s.start_suite(config_path="a")
        new = s.start_suite(config_path="b")
        assert [x.id for x in s.list_suites()] == [new, old]


def test_id_prefix_lookup(store: Store, result: RunResult):
    run_id = save(store, result)

    assert store.get_run(run_id[:6]).id == run_id
    assert store.get_run(run_id[:12]).id == run_id
    assert store.get_run(run_id.upper()[:6].lower()).id == run_id


def test_id_prefix_ambiguous_and_missing(store: Store, result: RunResult):
    ids = [save(store, result) for _ in range(40)]
    # Two ids sharing the first hex digit are certain among 40 ids; a 1-char prefix is too short.
    with pytest.raises(StoreError, match="at least 6"):
        store.get_run(ids[0][:3])

    # Force an ambiguity: rewrite two ids to share a 6-character prefix.
    with store._session.begin() as session:
        session.execute(text("PRAGMA foreign_keys=OFF"))
        session.execute(
            text("UPDATE runs SET id = :new WHERE id = :old"),
            {"new": "ffffff" + "0" * 26, "old": ids[0]},
        )
        session.execute(
            text("UPDATE runs SET id = :new WHERE id = :old"),
            {"new": "ffffff" + "1" * 26, "old": ids[1]},
        )
    with pytest.raises(StoreError, match="ambiguous"):
        store.get_run("ffffff")
    assert store.get_run("ffffff0").id == "ffffff" + "0" * 26

    with pytest.raises(StoreError, match="no run with id") as info:
        store.get_run("000000000000")
    assert "replay runs" in str(info.value)
    assert isinstance(info.value, ReplayError)


def test_like_wildcards_in_prefix_do_not_match(store: Store, result: RunResult):
    save(store, result)
    with pytest.raises(StoreError):
        store.get_run("%%%%%%")
    with pytest.raises(StoreError):
        store.get_run("______")


def test_env_url_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: RunResult):
    target = tmp_path / "from_env" / "env.db"
    monkeypatch.setenv("REPLAY_DB_URL", f"sqlite:///{target}")
    with Store.open() as s:
        run_id = save(s, result)
    assert target.exists()
    with Store.open() as again:
        assert again.get_run(run_id).id == run_id


def test_explicit_url_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("REPLAY_DB_URL", f"sqlite:///{tmp_path / 'env.db'}")
    explicit = tmp_path / "explicit.db"
    Store.open(f"sqlite:///{explicit}").close()
    assert explicit.exists()
    assert not (tmp_path / "env.db").exists()


def test_default_path_creates_replay_dir_under_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REPLAY_DB_URL", raising=False)

    Store.open().close()

    assert (tmp_path / ".replay" / "replay.db").is_file()


def test_foreign_keys_are_enforced(store: Store, result: RunResult):
    with pytest.raises(IntegrityError), store._session.begin() as session:
        session.execute(
            text(
                'INSERT INTO check_results (run_id, position, "check", passed, message, details)'
                " VALUES ('missing', 0, 'c', 1, 'm', '{}')"
            )
        )
    with pytest.raises(IntegrityError), store._session.begin() as session:
        session.execute(
            text(
                "INSERT INTO runs (id, suite_id, scenario, cassette_path, agent, mode, policy,"
                " task, passed, output, cost_usd, tokens, step_count, approximate_count,"
                " unconsumed_count, started_at, duration_ms)"
                " VALUES ('x', 'nosuite', 's', 'c', 'a', 'full', 'strict', 't', 1, 'null',"
                " 0, 0, 0, 0, 0, '2026-01-01', 0)"
            )
        )


def test_deleting_a_run_cascades(store: Store, result: RunResult):
    run_id = save(store, result)
    with store._session.begin() as session:
        session.execute(text("DELETE FROM runs WHERE id = :id"), {"id": run_id})
    with store._session() as session:
        assert session.execute(text("SELECT count(*) FROM steps")).scalar_one() == 0
        assert session.execute(text("SELECT count(*) FROM check_results")).scalar_one() == 0


def test_in_memory_sqlite_works(result: RunResult):
    with Store.open("sqlite://", clock=fixed_clock) as s:
        run_id = save(s, result)
        assert s.get_run(run_id).step_count == len(result.trajectory.steps)
        assert [r.id for r in s.list_runs()] == [run_id]


def test_failed_save_leaves_nothing_behind(store: Store, result: RunResult):
    save(store, result)
    with pytest.raises(IntegrityError):
        store.save_run(
            result,
            [],
            scenario="s",
            cassette_path="c",
            agent="a",
            passed=True,
            started_at=T0,
            duration_ms=0,
            suite_id="no-such-suite",
        )
    assert len(store.list_runs()) == 1
    with store._session() as session:
        n = session.execute(text("SELECT count(*) FROM steps")).scalar_one()
    assert n == len(result.trajectory.steps)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_unwritable_path_raises_store_error(tmp_path: Path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StoreError, match="REPLAY_DB_URL"):
            Store.open(f"sqlite:///{locked / 'sub' / 'r.db'}")
    finally:
        locked.chmod(0o700)


def test_path_under_a_file_raises_store_error(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(StoreError, match="REPLAY_DB_URL"):
        Store.open(f"sqlite:///{blocker / 'r.db'}")


def test_bad_url_raises_store_error():
    with pytest.raises(StoreError, match="REPLAY_DB_URL"):
        Store.open("not a url")
    with pytest.raises(StoreError, match="REPLAY_DB_URL"):
        Store.open("nosuchdialect://host/db")


def test_not_a_database_file_raises_store_error(tmp_path: Path):
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(StoreError, match="REPLAY_DB_URL"):
        Store.open(f"sqlite:///{junk}")


def test_advisory_flag_round_trips(store: Store, result: RunResult):
    checks = [
        CheckResult(check="judge", passed=False, message="judge said fail", advisory=True),
        CheckResult(check="rule:x", passed=True, message="ok"),
    ]
    run_id = store.save_run(
        result,
        checks,
        scenario="s",
        cassette_path="c.jsonl",
        agent="a:b",
        passed=True,
        started_at=T0,
        duration_ms=1.0,
    )
    stored = store.get_run(run_id).checks
    assert [c.advisory for c in stored] == [True, False]
