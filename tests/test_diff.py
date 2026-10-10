"""Trace diff: alignment costs, tie-breaking, properties, adapters and `replay diff`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import JsonValue
from typer.testing import CliRunner

from agent_replay import Cassette
from agent_replay.cassette import ErrorInfo, Kind
from agent_replay.checks import tool_sequence
from agent_replay.cli import app
from agent_replay.diff import (
    DiffStep,
    FieldChange,
    TraceDiff,
    argument_distance,
    diff_runs,
    diff_steps,
    diff_trajectories,
    steps_from_cassette,
    steps_from_record,
    steps_from_trajectory,
    to_json,
    to_markdown,
)
from agent_replay.runner import replay
from agent_replay.store import Store
from agent_replay.trajectory import Step, StepStatus, Trajectory
from examples.toy_agent import agent as toy

ROOT = Path(__file__).parents[1]
TOY = ROOT / "examples" / "toy_agent" / "cassettes"
RETAIL = ROOT / "examples" / "retail_agent" / "cassettes"
T0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
ENV = {"COLUMNS": "200"}

runner = CliRunner()


def tool(name: str, request: JsonValue = None, *, response: JsonValue = None) -> DiffStep:
    return DiffStep(
        index=0,
        kind="tool",
        name=name,
        status="exact",
        request={} if request is None else request,
        response=response,
    )


def model(*messages: str, response: JsonValue = None) -> DiffStep:
    return DiffStep(
        index=0,
        kind="model",
        name="anthropic.messages.create",
        status="exact",
        request={"model": "m", "messages": [{"role": "user", "content": m} for m in messages]},
        response=response,
    )


def indexed(steps: list[DiffStep]) -> list[DiffStep]:
    return [s.model_copy(update={"index": i}) for i, s in enumerate(steps)]


def ops(diff: TraceDiff) -> list[str]:
    return [r.op for r in diff.rows]


def test_identical_is_all_same_at_distance_zero() -> None:
    steps = indexed([model("hi"), tool("lookup", {"id": 1}), model("hi", "found")])
    diff = diff_steps(steps, steps)
    assert ops(diff) == ["same"] * 3
    assert diff.distance == 0
    assert diff.counts == {"same": 3, "changed": 0, "added": 0, "removed": 0}
    assert all(r.changes == [] for r in diff.rows)


def test_inserted_verification_call_is_one_added_row() -> None:
    base = indexed(
        [
            tool("lookup_order", {"order_id": "A1"}),
            tool("get_policy"),
            tool("issue_refund", {"order_id": "A1", "amount": 10}),
            tool("send_email", {"to": "a@example.com"}),
            tool("close_ticket", {"id": 7}),
        ]
    )
    head = indexed(
        [*base[:3], tool("verify_refund", {"order_id": "A1"}), *base[3:]]
    )  # inserted after step 3
    diff = diff_steps(base, head)
    assert ops(diff) == ["same", "same", "same", "added", "same", "same"]
    assert diff.rows[3].head is not None
    assert diff.rows[3].head.name == "verify_refund"
    assert diff.rows[3].base is None
    assert diff.distance == 1


def test_removed_step() -> None:
    base = indexed([tool("a"), tool("b"), tool("c")])
    head = indexed([tool("a"), tool("c")])
    diff = diff_steps(base, head)
    assert ops(diff) == ["same", "removed", "same"]
    assert diff.rows[1].base is not None
    assert diff.rows[1].base.name == "b"
    assert diff.rows[1].head is None
    assert diff.counts["removed"] == 1


def test_changed_argument_reports_the_field_and_costs_half_plus_half_d() -> None:
    base = [tool("issue_refund", {"order_id": "A1", "amount": 10})]
    head = [tool("issue_refund", {"order_id": "A1", "amount": 20})]
    diff = diff_steps(base, head)
    (row,) = diff.rows
    assert row.op == "changed"
    assert row.cost == pytest.approx(0.5 + 0.5 * (1 / 2))
    assert row.changes == [FieldChange(path="request.amount", base=10, head=20)]
    assert argument_distance(base[0], head[0]) == pytest.approx(0.5)


def test_added_key_counts_over_the_union_and_is_none_on_the_missing_side() -> None:
    base = [tool("t", {"a": 1})]
    head = [tool("t", {"a": 1, "b": 2, "c": 3})]
    (row,) = diff_steps(base, head).rows
    assert row.cost == pytest.approx(0.5 + 0.5 * (2 / 3))
    assert [(c.path, c.base, c.head) for c in row.changes] == [
        ("request.b", None, 2),
        ("request.c", None, 3),
    ]


def test_response_only_change_costs_half() -> None:
    base = [tool("t", {"a": 1}, response="ok")]
    head = [tool("t", {"a": 1}, response="denied")]
    (row,) = diff_steps(base, head).rows
    assert row.op == "changed"
    assert row.cost == 0.5
    assert row.changes == [FieldChange(path="response", base="ok", head="denied")]


def test_non_object_request_distance_is_all_or_nothing() -> None:
    assert argument_distance(tool("t", [1, 2]), tool("t", [1, 3])) == 1
    assert argument_distance(tool("t", [1, 2]), tool("t", [1, 2])) == 0
    (row,) = diff_steps([tool("t", [1])], [tool("t", "x")]).rows
    assert row.cost == 1
    assert row.changes == [FieldChange(path="request", base=[1], head="x")]


def test_model_steps_compare_only_the_last_message() -> None:
    base, head = model("hi", "lookup said A"), model("hi", "lookup said B")
    # Keys compared: model and messages[-1]; one of two differs.
    assert argument_distance(base, head) == pytest.approx(0.5)
    (row,) = diff_steps([base], [head]).rows
    assert row.cost == pytest.approx(0.75)
    assert [c.path for c in row.changes] == ["request.messages[-1]"]
    assert row.changes[0].head == {"role": "user", "content": "lookup said B"}


def test_model_steps_with_only_earlier_messages_different() -> None:
    base, head = model("hi", "done"), model("hello", "extra", "done")
    assert argument_distance(base, head) == 0
    (row,) = diff_steps([base], [head]).rows
    assert row.op == "changed"
    assert row.cost == 0.5
    assert row.changes == [
        FieldChange(path="request.messages", base="2 message(s)", head="3 message(s)")
    ]


def test_names_never_cross() -> None:
    a, b = tool("a", {"x": 1}), tool("b", {"x": 1})
    diff = diff_steps([a, b], [b, a])
    assert diff.distance == 2
    assert sorted(ops(diff)) == ["added", "removed", "same"]
    for row in diff.rows:
        if row.base is not None and row.head is not None:
            assert row.base.name == row.head.name


def test_different_kinds_never_substitute() -> None:
    t = tool("anthropic.messages.create", {"model": "m"})
    m = DiffStep(
        index=0, kind="model", name="anthropic.messages.create", status="exact", request={}
    )
    assert ops(diff_steps([t], [m])) == ["removed", "added"]


def test_tie_breaking_prefers_diagonal_then_deletion_then_insertion() -> None:
    a, b = tool("a"), tool("b")
    # Diagonal beats insertion: the earliest head copy is the one paired.
    diff = diff_steps([a], indexed([a, a]))
    assert ops(diff) == ["same", "added"]
    assert diff.rows[0].head is not None
    assert diff.rows[0].head.index == 0
    # Deletion beats insertion at the same cost.
    assert ops(diff_steps([a, b], [b, a])) == ["removed", "same", "added"]
    assert ops(diff_steps([a], [b])) == ["removed", "added"]
    # Changed (diagonal) beats remove + add.
    assert ops(diff_steps([tool("a", {"k": 1})], [tool("a", {"k": 2})])) == ["changed"]


def test_tie_breaking_is_deterministic() -> None:
    base = indexed([tool("a"), tool("b"), tool("a"), tool("b")])
    head = indexed([tool("b"), tool("a"), tool("b"), tool("a")])
    first = to_json(diff_steps(base, head))
    assert all(to_json(diff_steps(base, head)) == first for _ in range(5))


def test_empty_inputs() -> None:
    empty = diff_steps([], [])
    assert empty.rows == []
    assert empty.distance == 0
    assert empty.counts == {"same": 0, "changed": 0, "added": 0, "removed": 0}
    steps = indexed([tool("a"), tool("b")])
    assert ops(diff_steps([], steps)) == ["added", "added"]
    assert ops(diff_steps(steps, [])) == ["removed", "removed"]
    assert diff_steps(steps, []).distance == 2


def test_totals_of_cost_and_latency() -> None:
    base = [model("hi").model_copy(update={"cost_usd": 0.01, "latency_ms": 100.0}), tool("a")]
    head = [model("hi").model_copy(update={"cost_usd": 0.03, "latency_ms": 250.0})]
    diff = diff_steps(base, head)
    assert diff.base_cost_usd == pytest.approx(0.01)
    assert diff.head_cost_usd == pytest.approx(0.03)
    assert diff.base_latency_ms == 100
    assert diff.head_latency_ms == 250


def test_status_cost_and_latency_are_not_compared() -> None:
    base = tool("a", {"x": 1})
    head = base.model_copy(update={"status": "approximate", "cost_usd": 1.0, "latency_ms": 9.0})
    assert ops(diff_steps([base], [head])) == ["same"]


def test_markdown_and_json() -> None:
    base = indexed([tool("a", {"k": 1}), tool("b|c")])
    head = indexed([tool("a", {"k": 2}), tool("d")])
    diff = diff_steps(base, head, base_label="old", head_label="new")
    markdown = to_markdown(diff)
    assert markdown.startswith("## Trace diff: `old` → `new`")
    assert "| Op | Base | Head | Cost | Changes |" in markdown
    assert "| changed | `0 tool a` | `0 tool a` | 1 | `request.k: 1 → 2` |" in markdown
    assert "`1 tool b\\|c`" in markdown
    assert TraceDiff.model_validate_json(to_json(diff)) == diff


names = st.sampled_from(["a", "b", "c"])
requests = st.dictionaries(st.sampled_from(["x", "y", "z"]), st.integers(0, 2), max_size=3)
steps_strategy = st.lists(
    st.builds(lambda n, r, resp: tool(n, r, response=resp), names, requests, st.integers(0, 1)),
    max_size=8,
).map(indexed)


def content(steps: list[DiffStep]) -> list[tuple[object, ...]]:
    return [(s.kind, s.name, s.request, s.response, s.error) for s in steps]


@settings(max_examples=200, deadline=None)
@given(base=steps_strategy, head=steps_strategy)
def test_alignment_properties(base: list[DiffStep], head: list[DiffStep]) -> None:
    diff = diff_steps(base, head)
    assert diff.distance == pytest.approx(diff_steps(head, base).distance)
    assert (diff.distance == 0) == (content(base) == content(head))
    assert [r.base for r in diff.rows if r.base is not None] == base
    assert [r.head for r in diff.rows if r.head is not None] == head
    assert diff.distance == pytest.approx(sum(r.cost for r in diff.rows))
    assert diff.distance <= len(base) + len(head)
    assert sum(diff.counts.values()) == len(diff.rows)
    for row in diff.rows:
        if row.base is not None and row.head is not None:
            assert (row.base.kind, row.base.name) == (row.head.kind, row.head.name)
            assert (row.op == "same") == (row.cost == 0)
            assert 0.5 <= row.cost <= 1 or row.op == "same"


@pytest.fixture
def toy_cassette() -> Cassette:
    return Cassette.load(TOY / "refund_eligible.jsonl")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(f"sqlite:///{tmp_path / 'r.db'}", clock=lambda: T0) as s:
        yield s


def save_replay(store: Store, cassette: Cassette, cassette_path: Path) -> str:
    result = replay(cassette, toy.run)
    checks = result.check(tool_sequence(cassette.tool_names()))
    return store.save_run(
        result,
        checks,
        scenario="refund_eligible",
        cassette_path=str(cassette_path),
        agent="examples.toy_agent.agent:run",
        passed=True,
        started_at=T0,
        duration_ms=1.0,
    )


def test_trajectory_record_and_cassette_adapters_agree(
    toy_cassette: Cassette, store: Store
) -> None:
    result = replay(toy_cassette, toy.run)
    run = store.get_run(save_replay(store, toy_cassette, TOY / "refund_eligible.jsonl"))
    from_trajectory = steps_from_trajectory(result.trajectory)
    from_record = steps_from_record(run)
    from_cassette = steps_from_cassette(toy_cassette)
    assert content(from_trajectory) == content(from_record) == content(from_cassette)
    assert [s.index for s in from_cassette] == [s.index for s in from_record]
    assert {s.status for s in from_record} == {"exact"}
    assert diff_steps(from_cassette, from_record).distance == 0
    diff = diff_runs(run, run)
    assert diff.base_label == run.id
    assert diff.counts["same"] == len(run.steps)
    assert diff_trajectories(result.trajectory, result.trajectory).distance == 0


def test_nested_steps_are_left_out() -> None:
    def step(index: int, name: str, parent: str | None) -> Step:
        return Step(
            index=index,
            kind=Kind.TOOL,
            name=name,
            status=StepStatus.LIVE,
            fingerprint="sha256:0",
            request={},
            span_id=f"s{index}",
            parent_span_id=parent,
        )

    trajectory = Trajectory(steps=(step(0, "outer", None), step(1, "inner", "s0")))
    assert [s.name for s in steps_from_trajectory(trajectory)] == ["outer"]


def test_step_error_is_kept_as_json() -> None:
    step = Step(
        index=0,
        kind=Kind.TOOL,
        name="t",
        status=StepStatus.EXACT,
        fingerprint="sha256:0",
        request={},
        error=ErrorInfo(type="ValueError", message="bad"),
    )
    assert DiffStep.from_step(step).error == {
        "type": "ValueError",
        "module": None,
        "message": "bad",
    }


def invoke(*args: str):
    return runner.invoke(app, list(args), env=ENV)


@pytest.mark.parametrize(
    "path", [TOY / "refund_eligible.jsonl", RETAIL / "cancel_pending_laptop.jsonl"]
)
def test_cli_diff_cassette_against_itself(path: Path, tmp_path: Path) -> None:
    json_path, md_path = tmp_path / "d.json", tmp_path / "d.md"
    result = invoke(
        "diff", str(path), str(path), "--json", str(json_path), "--markdown", str(md_path)
    )
    assert result.exit_code == 0, result.output
    count = len(steps_from_cassette(Cassette.load(path)))
    assert f"distance 0 ({count} same, 0 changed, 0 added, 0 removed)" in result.output
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["distance"] == 0
    assert len(data["rows"]) == count
    assert md_path.read_text(encoding="utf-8").startswith("## Trace diff")


def test_cli_diff_two_scenarios_is_informational() -> None:
    result = invoke("diff", str(TOY / "refund_eligible.jsonl"), str(TOY / "refund_denied.jsonl"))
    assert result.exit_code == 0, result.output
    assert "removed" in result.output
    assert "changed" in result.output
    assert "request.order_id" in result.output
    assert "distance 4.167 (1 same, 3 changed, 0 added, 2 removed)" in result.output


def test_cli_diff_cassette_against_stored_run(tmp_path: Path, toy_cassette: Cassette) -> None:
    url = f"sqlite:///{tmp_path / 'r.db'}"
    with Store.open(url) as store:
        run_id = save_replay(store, toy_cassette, TOY / "refund_eligible.jsonl")
    result = invoke("diff", str(TOY / "refund_eligible.jsonl"), run_id[:8], "--db", url)
    assert result.exit_code == 0, result.output
    assert "distance 0 (6 same" in result.output
    assert run_id[:12] in result.output


def test_cli_diff_unknown_run_names_the_fix(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'r.db'}"
    result = invoke("diff", str(TOY / "refund_eligible.jsonl"), "deadbeef00", "--db", url)
    assert result.exit_code == 2
    assert "neither a cassette file nor a stored run" in result.output
    assert "replay runs" in result.output


def test_cli_diff_missing_cassette(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    result = invoke("diff", str(missing), str(TOY / "refund_eligible.jsonl"))
    assert result.exit_code == 2
    assert "does not exist" in result.output
