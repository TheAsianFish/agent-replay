"""Regression tests for edge cases: nesting, threads, error types, unsupported surfaces."""

from __future__ import annotations

import concurrent.futures
import inspect
from pathlib import Path

import anthropic
import pytest
from anthropic.types import Message
from pydantic import BaseModel
from typer.testing import CliRunner

from agent_replay import Cassette, RecordingSession, ReplayError, StepStatus, tool, wrap
from agent_replay._json import canonical_json, fingerprint, to_jsonable
from agent_replay.checks import tool_sequence
from agent_replay.cli import app
from agent_replay.fake import FakeAnthropic, Reply, scripted
from agent_replay.runner import record, replay

ROOT = Path(__file__).parents[1]
NO_MODEL = FakeAnthropic(scripted([]))  # these agents only call tools


@tool
def inner(x: int) -> int:
    return x * 2


@tool
def outer(x: int) -> int:
    return inner(x) + 1


@tool
def plain(x: int) -> int:
    return x


class Nullable(BaseModel):
    value: int | None


@tool
def nullable() -> Nullable:
    return Nullable(value=None)


@tool
def missing(key: str) -> str:
    return {"a": "1"}[key]


def run_pair(_client, _task):
    return [outer(1), plain(2)]


# --- nesting -----------------------------------------------------------------


def test_nested_tool_calls_record_a_loadable_cassette_with_parent_spans():
    _, cassette = record(run_pair, "t", client=NO_MODEL)
    loaded = Cassette.loads(cassette.dumps())
    by_name = {e.name: e for e in loaded.envelopes}
    assert [e.seq for e in loaded.envelopes] == [0, 1, 2]
    assert by_name["inner"].parent_span_id == by_name["outer"].span_id
    assert by_name["outer"].parent_span_id is None
    assert loaded.tool_names() == ["outer", "plain"]


def test_nested_calls_are_served_through_their_parent_on_replay():
    _, cassette = record(run_pair, "t", client=NO_MODEL)
    result = replay(cassette, run_pair)
    assert result.ok
    assert result.output == [3, 2]
    assert result.trajectory.tool_names() == ["outer", "plain"]
    assert result.unconsumed == ()
    assert result.check(tool_sequence(cassette.tool_names()))[0].passed


# --- threads -----------------------------------------------------------------


def threaded(_client, _task):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(plain, range(8)))


def test_tools_run_in_a_thread_pool_are_recorded_and_served():
    _, cassette = record(threaded, "t", client=NO_MODEL)
    assert sorted(e.request["x"] for e in cassette.envelopes) == list(range(8))  # type: ignore[index]
    assert [e.seq for e in Cassette.loads(cassette.dumps()).envelopes] == list(range(8))

    calls: list[int] = []
    original = plain.fn
    plain.fn = lambda x: calls.append(x) or x  # type: ignore[assignment]
    try:
        result = replay(cassette, threaded)
    finally:
        plain.fn = original
    assert result.ok
    assert calls == []  # nothing ran for real
    assert len(result.trajectory.tool_steps()) == 8


# --- values ------------------------------------------------------------------


def test_required_nullable_fields_round_trip():
    def agent(_c, _t):
        return nullable().value

    _, cassette = record(agent, "t", client=NO_MODEL)
    assert cassette.envelopes[0].response == {"value": None}
    result = replay(cassette, agent)
    assert result.ok, result.error


def test_replayed_exception_is_an_instance_of_the_recorded_type():
    def agent(_c, _t):
        try:
            return missing("b")
        except KeyError:
            return "handled"

    recorded, cassette = record(agent, "t", client=NO_MODEL)
    assert recorded.output == "handled"
    assert cassette.envelopes[0].error is not None
    assert cassette.envelopes[0].error.module == "builtins"
    assert replay(cassette, agent).output == "handled"


def test_swallowed_unrecorded_call_still_fails_the_run():
    _, cassette = record(lambda c, t: plain(1), "t", client=NO_MODEL)

    def swallows(_c, _t):
        plain(1)
        try:
            plain(2)
        except Exception:
            return "carried on"
        return "unreachable"

    result = replay(cassette, swallows)
    assert result.error is not None
    assert result.error.type == "UnrecordedCallError"
    assert result.trajectory.steps[-1].status is StepStatus.UNRECORDED


def test_unrecordable_tool_argument_fails_the_run_with_a_clear_error():
    @tool
    def takes(obj: object) -> int:
        return 1

    def agent(_c, _t):
        try:
            return takes(object())
        except Exception:
            return "swallowed"

    result, cassette = record(agent, "t", client=NO_MODEL)
    assert result.error is not None
    assert "cannot be recorded as JSON" in result.error.message
    assert cassette.footer.status == "error"


def test_unrecordable_tool_result_fails_the_run():
    @tool
    def returns_object() -> object:
        return object()

    result, _ = record(lambda c, t: returns_object(), "t", client=NO_MODEL)
    assert result.error is not None
    assert "cannot be recorded as JSON" in result.error.message


# --- decoration is lazy ---------------------------------------------------------


class Handle:
    pass


def test_decorating_never_fails_for_unschematisable_or_unresolvable_annotations():
    @tool
    def connect(dsn: str) -> Handle:
        return Handle()

    @tool
    def later(x: NotDefinedYet) -> int:  # type: ignore[name-defined]  # noqa: F821
        return 1

    assert isinstance(connect("x"), Handle)  # no session: plain call
    with RecordingSession(), pytest.raises(ReplayError, match="cannot be rebuilt"):
        connect("x")
    with pytest.raises(ReplayError, match="type hints"):
        later.to_anthropic()


# --- unsupported client surfaces --------------------------------------------------


def _client():
    return wrap(FakeAnthropic(scripted([Reply(text="hi")])))


@pytest.mark.parametrize("attr", ["parse", "with_raw_response"])
def test_unrecorded_message_methods_raise_inside_a_session(attr):
    client = _client()
    with RecordingSession() as session, pytest.raises(ReplayError, match="not recorded yet"):
        getattr(client.messages, attr)(model="m", max_tokens=1, messages=[])
    assert session.fatal is not None


def test_beta_namespace_raises_inside_a_session():
    client = wrap(anthropic.Anthropic(api_key="test-key"))
    with RecordingSession(), pytest.raises(ReplayError, match=r"client\.beta"):
        client.beta.messages.create(model="m", max_tokens=1, messages=[])
    assert client.beta is not None  # fine outside a session


def test_async_clients_get_an_async_proxy():
    client = wrap(anthropic.AsyncAnthropic(api_key="test-key"))
    assert inspect.iscoroutinefunction(client.messages.create)


def test_sdk_sentinels_are_dropped_from_the_request():
    client = _client()
    with RecordingSession() as session:
        reply = client.messages.create(
            model="m",
            max_tokens=10,
            messages=[{"role": "user", "content": "x"}],
            system=anthropic.omit,
            temperature=anthropic.NOT_GIVEN,
        )
    assert isinstance(reply, Message)
    request = session.finish().envelopes[0].request
    assert isinstance(request, dict)
    assert "system" not in request
    assert "temperature" not in request


# --- CLI error paths --------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def in_root(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_agent_module_with_a_syntax_error_is_a_usage_error(in_root, monkeypatch):
    pkg = in_root / "brokenpkg"
    pkg.mkdir()
    _write(pkg / "__init__.py", "")
    _write(pkg / "agent.py", "def run(:\n")
    monkeypatch.syspath_prepend(str(in_root))
    out = runner.invoke(app, ["record", "brokenpkg.agent:run", "-t", "x", "-o", str(in_root / "c")])
    assert out.exit_code == 2
    assert "SyntaxError" in out.output


def test_client_factory_that_raises_is_a_usage_error(in_root, monkeypatch):
    pkg = in_root / "factorypkg"
    pkg.mkdir()
    _write(pkg / "__init__.py", "def make():\n    raise RuntimeError('no config')\n")
    monkeypatch.syspath_prepend(str(in_root))
    out = runner.invoke(
        app,
        [
            "record",
            "examples.toy_agent.agent:run",
            "-t",
            "x",
            "-o",
            str(in_root / "c"),
            "--client",
            "factorypkg:make",
        ],
    )
    assert out.exit_code == 2
    assert "no config" in out.output


def test_bad_log_level_is_a_usage_error(in_root):
    out = runner.invoke(app, ["show", "x"], env={"REPLAY_LOG_LEVEL": "LOUD"})
    assert out.exit_code == 2


def test_cassette_without_task_needs_task_option(in_root):
    session = RecordingSession(agent="examples.toy_agent.agent:run")
    path = in_root / "notask.jsonl"
    session.finish().dump(path)
    out = runner.invoke(app, ["run", str(path)])
    assert out.exit_code == 2
    assert "--task" in out.output


def test_strict_fails_when_steps_were_approximate(in_root):
    cassette = ROOT / "examples/toy_agent/cassettes/refund_eligible.jsonl"
    args = ["run", str(cassette), "--agent", "examples.toy_agent.mutants:first_tool_only"]
    lenient = runner.invoke(app, [*args, "--expect", "lookup_order,issue_refund"])
    strict = runner.invoke(app, [*args, "--expect", "lookup_order,issue_refund", "--strict"])
    assert lenient.exit_code == 0
    assert strict.exit_code == 1
    assert "strict" in strict.output


# --- values JSON cannot carry exactly -------------------------------------------


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_floats_cannot_be_converted(value):
    with pytest.raises(TypeError, match="not finite"):
        to_jsonable({"x": [1, value]})


def test_non_finite_tool_argument_fails_the_run():
    @tool
    def scale(factor: float) -> float:
        return factor

    result, cassette = record(lambda c, t: scale(float("inf")), "t", client=NO_MODEL)
    assert result.error is not None
    assert result.error.type == "ReplayError"
    assert "not finite" in result.error.message
    assert cassette.footer.status == "error"
    assert Cassette.loads(cassette.dumps()) == cassette


def test_non_finite_tool_result_fails_the_run():
    @tool
    def ratio() -> float:
        return float("nan")

    result, cassette = record(lambda c, t: ratio(), "t", client=NO_MODEL)
    assert result.error is not None
    assert "not finite" in result.error.message
    assert Cassette.loads(cassette.dumps()) == cassette


def test_non_finite_agent_output_fails_the_run():
    result, cassette = record(lambda c, t: {"score": float("inf")}, "t", client=NO_MODEL)
    assert result.error is not None
    assert "not finite" in result.error.message
    assert cassette.footer.output is None
    assert Cassette.loads(cassette.dumps()) == cassette


def test_sets_are_recorded_in_a_deterministic_order():
    request = {"tags": {"zeta", "alpha", "mu"}, "ids": frozenset({3, 1, 2})}
    assert to_jsonable(request) == {"tags": ["alpha", "mu", "zeta"], "ids": [1, 2, 3]}
    assert fingerprint("tool", "t", request) == fingerprint(
        "tool", "t", {"tags": ["alpha", "mu", "zeta"], "ids": [1, 2, 3]}
    )


@pytest.mark.parametrize("value", ["\udcff", {"\udcff": 1}, ["ok", "\ud800"]])
def test_lone_surrogates_are_a_type_error(value):
    with pytest.raises(TypeError, match="valid Unicode"):
        fingerprint("tool", "t", {"x": value})
    with pytest.raises(TypeError, match="valid Unicode"):
        canonical_json(value)


def test_lone_surrogate_in_a_tool_argument_fails_the_run():
    @tool
    def echo(text: str) -> str:
        return text

    def agent(_c, _t):
        try:
            return echo("bad \udcff byte")
        except Exception:
            return "swallowed"

    with RecordingSession() as session:
        output = agent(None, "t")
    assert output == "swallowed"
    assert isinstance(session.fatal, ReplayError)
    assert "valid Unicode" in str(session.fatal)
    assert session.finish(output=output).footer.status == "error"


def test_nested_sdk_sentinels_are_dropped_from_the_request():
    client = _client()
    with RecordingSession() as session:
        reply = client.messages.create(
            model="m",
            max_tokens=10,
            messages=[{"role": "user", "content": "x"}],
            output_config={"effort": anthropic.omit, "format": anthropic.NOT_GIVEN},
            metadata={"user_id": anthropic.omit},
        )
    assert isinstance(reply, Message)
    cassette = session.finish()
    assert cassette.footer.status == "ok"
    [envelope] = cassette.envelopes
    assert isinstance(envelope.request, dict)
    assert envelope.request["output_config"] == {}
    assert envelope.request["metadata"] == {}


# --- unrecorded surfaces fail on call, not on attribute access --------------------


def test_probing_unrecorded_surfaces_does_not_fail_the_session():
    client = wrap(anthropic.Anthropic(api_key="test-key"))
    with RecordingSession() as session:
        assert hasattr(client.messages, "stream")
        assert hasattr(client.messages, "parse")
        assert hasattr(client, "beta")
        assert hasattr(client, "with_raw_response")
    assert session.fatal is None
    assert session.finish().footer.status == "ok"


def test_calling_through_an_unrecorded_namespace_raises():
    client = wrap(anthropic.Anthropic(api_key="test-key"))
    with RecordingSession() as session, pytest.raises(ReplayError, match=r"client\.beta"):
        client.beta.messages.create(model="m", max_tokens=1, messages=[])
    assert session.fatal is not None


def test_wrapped_client_is_a_context_manager():
    inner = anthropic.Anthropic(api_key="test-key")
    with wrap(inner) as client:
        assert client.messages.create is not None
    assert inner.is_closed()


# --- exit codes -------------------------------------------------------------------


def test_record_with_a_client_that_cannot_be_wrapped_is_a_usage_error(in_root):
    out = runner.invoke(
        app,
        [
            "record",
            "examples.toy_agent.agent:run",
            "-t",
            "x",
            "-o",
            str(in_root / "c.jsonl"),
            "--client",
            "builtins:dict",
        ],
    )
    assert out.exit_code == 2, out.output
    assert "cannot wrap dict" in out.output
    assert not (in_root / "c.jsonl").exists()


def test_record_function_rejects_an_unwrappable_client_before_running_the_agent():
    ran: list[str] = []
    with pytest.raises(TypeError, match="cannot wrap"):
        record(lambda c, t: ran.append(t), "t", client={})
    assert ran == []


def test_record_to_an_unwritable_path_is_a_usage_error(in_root):
    blocker = in_root / "file.txt"
    _write(blocker, "not a directory")
    target = blocker / "x.jsonl"
    out = runner.invoke(
        app,
        [
            "record",
            "examples.toy_agent.agent:run",
            "-t",
            "Order A1001 arrived damaged. Can I get a refund?",
            "-o",
            str(target),
            "--client",
            "examples.toy_agent.fake_model:make_client",
        ],
    )
    assert out.exit_code == 2, out.output
    assert "cannot write" in out.output
    assert "Traceback" not in out.output


# --- tool_sequence reports the first differing position ----------------------------


def test_exact_mismatch_reports_the_first_differing_call():
    def agent(_c, _t):
        return [plain(1), inner(1), inner(1), inner(1), plain(2), outer(1)]

    _, cassette = record(agent, "t", client=NO_MODEL)
    result = replay(cassette, agent)
    [check] = result.check(tool_sequence(["plain", "outer"]))
    assert not check.passed
    assert check.message.startswith("tool call #2 differs")
