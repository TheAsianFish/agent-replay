"""The Ollama provider: fake client, adapter, and the toy agent ported to Ollama."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path

import ollama
import pytest
from ollama import ChatResponse, Message

from agent_replay import Cassette, Kind, StepStatus, tool, wrap
from agent_replay.checks import tool_sequence
from agent_replay.fake import FakeAnthropic, Reply, ToolCall, scripted
from agent_replay.fake_ollama import FakeAsyncOllama, FakeOllama, FakeOllamaRequest
from agent_replay.record import WrappedClient, adapter_for_calls
from agent_replay.record.anthropic import ANTHROPIC
from agent_replay.record.ollama import OLLAMA
from agent_replay.runner import load_agent, record, replay
from examples.toy_agent import ollama_agent
from examples.toy_agent.ollama_fake import make_client

COMMITTED = Path(__file__).parents[1] / "examples/toy_agent/cassettes/ollama_refund_eligible.jsonl"


def _two_turns(request: FakeOllamaRequest) -> Reply:
    if request.latest_result("add") is not None:
        return Reply(text="The sum is three.")
    return Reply(text="Adding.", tool_calls=[ToolCall("add", {"a": 1, "b": 2})])


# --- fake client ----------------------------------------------------------------


def test_fake_is_a_real_ollama_client_and_returns_chat_responses():
    client = FakeOllama(_two_turns)
    assert isinstance(client, ollama.Client)

    response = client.chat("llama3.2", [{"role": "user", "content": "1 + 2?"}])

    assert isinstance(response, ChatResponse)
    assert response.done is True
    assert response.model == "llama3.2"
    assert response.message.content == "Adding."
    assert response.prompt_eval_count
    assert response.eval_count
    [call] = response.message.tool_calls or []
    assert isinstance(call, Message.ToolCall)
    assert (call.function.name, dict(call.function.arguments)) == ("add", {"a": 1, "b": 2})


def test_fake_request_reads_ollama_tool_messages():
    seen: list[FakeOllamaRequest] = []

    def spy(request: FakeOllamaRequest) -> Reply:
        seen.append(request)
        return Reply(text="ok")

    FakeOllama(spy).chat(
        model="m",
        messages=[
            Message(role="system", content="be brief"),
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_name": "add", "content": "3"},
            {"role": "tool", "tool_name": "div", "content": "error: division by zero"},
        ],
    )

    [request] = seen
    assert request.system == "be brief"
    assert request.first_user_text() == "hello"
    assert [(r.name, r.content, r.is_error) for r in request.tool_results()] == [
        ("add", "3", False),
        ("div", "error: division by zero", True),
    ]
    assert request.latest_result("div") is not None


def test_fake_streams_chunks_that_add_up_to_the_whole_response():
    messages = [{"role": "user", "content": "1 + 2?"}]
    whole = FakeOllama(_two_turns).chat("m", messages)
    chunks = list(FakeOllama(_two_turns).chat("m", messages, stream=True))

    assert len(chunks) > 1
    assert all(isinstance(c, ChatResponse) for c in chunks)
    assert [c.done for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert "".join(c.message.content or "" for c in chunks) == whole.message.content
    assert chunks[-1].message.tool_calls == whole.message.tool_calls
    assert chunks[-1].eval_count == whole.eval_count


def test_async_fake_chats_and_streams():
    client = FakeAsyncOllama(_two_turns)
    assert isinstance(client, ollama.AsyncClient)

    async def go() -> tuple[ChatResponse, list[ChatResponse]]:
        messages = [{"role": "user", "content": "1 + 2?"}]
        whole = await client.chat("m", messages)
        stream = await client.chat("m", messages, stream=True)
        return whole, [c async for c in stream]

    whole, chunks = asyncio.run(go())
    assert whole.message.content == "Adding."
    assert "".join(c.message.content or "" for c in chunks) == "Adding."


def test_fake_refuses_everything_but_chat():
    with pytest.raises(NotImplementedError, match="only implements chat"):
        FakeOllama(_two_turns).generate(model="m", prompt="hi")


# --- adapter ------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
      a: The first number
      b: The second number
    """
    return a + b


def _agent(client, task):
    """Two turns through every request shape: positional model, Message objects, a tool."""
    messages = [Message(role="user", content=task)]
    first = client.chat("llama3.2", messages, tools=[add], keep_alive="5m")
    messages.append(first.message)
    for call in first.message.tool_calls or []:
        result = add(**call.function.arguments)
        messages.append({"role": "tool", "tool_name": "add", "content": str(result)})
    second = client.chat(model="llama3.2", messages=messages, options={"temperature": 0})
    return second.message.content


def _streaming_agent(client, task):
    chunks = list(client.chat("llama3.2", [{"role": "user", "content": task}], stream=True))
    assert all(isinstance(c, ChatResponse) for c in chunks)
    return "".join(c.message.content or "" for c in chunks)


async def _async_agent(client, task):
    first = await client.chat("llama3.2", [{"role": "user", "content": task}])
    stream = await client.chat(
        "llama3.2",
        [
            {"role": "user", "content": task},
            first.message,
            {"role": "tool", "tool_name": "add", "content": "3"},
        ],
        stream=True,
    )
    return "".join([c.message.content or "" async for c in stream])


@pytest.mark.parametrize(
    ("agent", "make"),
    [
        (_agent, lambda: FakeOllama(_two_turns)),
        (_streaming_agent, lambda: FakeOllama(_two_turns)),
        (_async_agent, lambda: FakeAsyncOllama(_two_turns)),
    ],
    ids=["sync", "stream", "async-and-async-stream"],
)
def test_record_then_full_replay_is_exact(agent, make):
    result, cassette = record(agent, "1 + 2?", client=make())

    assert result.ok, result.error
    assert {e.name for e in cassette.envelopes if e.kind is Kind.MODEL} == {"ollama.chat"}
    replayed = replay(Cassette.loads(cassette.dumps()), agent)
    assert replayed.ok, replayed.error
    assert replayed.output == result.output
    assert {s.status for s in replayed.trajectory.steps} == {StepStatus.EXACT}
    assert replayed.unconsumed == ()


def test_request_is_recorded_in_the_form_ollama_sends():
    _, cassette = record(_agent, "1 + 2?", client=FakeOllama(_two_turns))
    first, second = (e for e in cassette.envelopes if e.kind is Kind.MODEL)

    assert isinstance(first.request, dict)
    assert first.request["model"] == "llama3.2"  # positional, recorded as a keyword
    assert first.request["messages"] == [{"role": "user", "content": "1 + 2?"}]
    # The Python function, converted by ollama's own helper.
    assert first.request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "required": ["a", "b"],
                    "properties": {
                        "a": {"type": "integer", "description": "The first number"},
                        "b": {"type": "integer", "description": "The second number"},
                    },
                },
            },
        }
    ]
    assert first.request["keep_alive"] == "5m"
    assert isinstance(second.request, dict)
    assert second.request["options"] == {"temperature": 0}
    assert first.usage is not None
    assert first.usage.input_tokens > 0
    assert first.cost_usd is None  # local models are never priced


def test_volatile_fields_stay_out_of_the_fingerprint_but_options_do_not():
    def fingerprint_with(**extra):
        def agent(client, task):
            reply = client.chat("llama3.2", [{"role": "user", "content": task}], **extra)
            return len(list(reply)) if extra.get("stream") else reply.message.content

        _, cassette = record(agent, "1 + 2?", client=FakeOllama(_two_turns))
        return cassette.envelopes[0].fingerprint

    plain = fingerprint_with()
    assert fingerprint_with(keep_alive="10m") == plain
    assert fingerprint_with(stream=True) == plain
    assert fingerprint_with(options={"temperature": 0.5}) != plain


def test_model_live_replay_serves_tools_and_runs_the_fake_model():
    _, cassette = record(_agent, "1 + 2?", client=FakeOllama(_two_turns))

    replayed = replay(cassette, _agent, mode="model-live", client=FakeOllama(_two_turns))

    assert replayed.ok, replayed.error
    statuses = {s.kind: s.status for s in replayed.trajectory.steps}
    assert statuses == {Kind.MODEL: StepStatus.LIVE, Kind.TOOL: StepStatus.EXACT}


def test_ollama_cassettes_pick_the_ollama_offline_client():
    assert adapter_for_calls(["ollama.chat"]) is OLLAMA
    assert adapter_for_calls(["anthropic.messages.create"]) is ANTHROPIC
    assert OLLAMA.accepts(OLLAMA.offline_client(is_async=False))
    assert OLLAMA.accepts(OLLAMA.offline_client(is_async=True))


def test_offline_stand_in_names_the_fix_under_the_fallback_policy():
    _, cassette = record(_streaming_agent, "1 + 2?", client=FakeOllama(_two_turns))

    def asks_twice(client, task):
        _streaming_agent(client, task)
        return client.chat("llama3.2", [{"role": "user", "content": "again"}]).message.content

    replayed = replay(cassette, asks_twice, policy="fallback")

    assert replayed.error is not None
    assert replayed.error.type == "ReplayError"
    assert "no live Ollama client" in replayed.error.message
    assert "fallback policy" in replayed.error.message
    assert replayed.trajectory.steps[-1].status is StepStatus.FALLBACK


def test_wrap_is_idempotent_and_passes_through_without_a_session():
    wrapped = wrap(FakeOllama(_two_turns))
    assert isinstance(wrapped, WrappedClient)
    assert wrap(wrapped) is wrapped
    assert wrapped.chat("m", [{"role": "user", "content": "hi"}]).message.content == "Adding."
    assert isinstance(wrap(FakeAsyncOllama(_two_turns)), WrappedClient)


def test_anthropic_and_ollama_accept_disjoint_clients():
    ollama_clients = [ollama.Client(), ollama.AsyncClient(), FakeOllama(_two_turns)]
    anthropic_clients = [FakeAnthropic(scripted([])), ANTHROPIC.offline_client(is_async=False)]

    assert not any(ANTHROPIC.accepts(c) for c in ollama_clients)
    assert all(OLLAMA.accepts(c) for c in ollama_clients)
    assert not any(OLLAMA.accepts(c) for c in anthropic_clients)
    assert all(ANTHROPIC.accepts(c) for c in anthropic_clients)
    with pytest.raises(TypeError, match="not an ollama client"):
        OLLAMA.wrap(object())


def test_unrecorded_client_methods_fail_the_run_even_if_swallowed():
    def generates(client, task):
        assert callable(client.generate)  # looking it up is harmless
        try:
            client.generate(model="llama3.2", prompt=task)
        except Exception:
            return "carried on"
        return "unreachable"

    result, _ = record(generates, "hi", client=FakeOllama(_two_turns))

    assert result.error is not None
    assert result.error.type == "ReplayError"
    assert "client.generate is not recorded" in result.error.message


def test_importing_agent_replay_does_not_import_ollama():
    code = "import sys, agent_replay.runner; assert 'ollama' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_adapter_module_imports_without_ollama(monkeypatch):
    module = importlib.import_module("agent_replay.record.ollama")
    monkeypatch.setitem(sys.modules, "ollama", None)
    try:
        reloaded = importlib.reload(module)
        assert not reloaded.OLLAMA.accepts(FakeOllama(_two_turns))
        assert reloaded.OLLAMA.accepts(reloaded.OLLAMA.offline_client(is_async=False))
    finally:
        monkeypatch.undo()
        importlib.reload(module)


# --- the toy agent on Ollama ----------------------------------------------------------


def test_committed_ollama_cassette_replays_offline_without_a_client():
    cassette = Cassette.load(COMMITTED)
    assert cassette.header.agent == "examples.toy_agent.ollama_agent:run"

    # No client: the runner picks the Ollama offline stand-in from the envelope names.
    result = replay(cassette, load_agent(cassette.header.agent))

    assert result.ok, result.error
    assert result.output == "Done: I refunded $42.50 for order A1001 (reference RF-A1001)."
    assert {s.status for s in result.trajectory.steps} == {StepStatus.EXACT}
    assert result.unconsumed == ()
    [check] = result.check(tool_sequence(["lookup_order", "get_refund_policy", "issue_refund"]))
    assert check.passed, check.message


def test_committed_ollama_cassette_matches_a_fresh_recording():
    """Fails when the agent or fake model changed without re-recording (`make cassettes`)."""
    committed = Cassette.load(COMMITTED)
    assert committed.header.task is not None

    _, fresh = record(ollama_agent.run, committed.header.task, client=make_client())

    def comparable(c: Cassette) -> list[dict[str, object]]:
        return [e.model_dump(exclude={"latency_ms"}) for e in c.envelopes]

    assert comparable(fresh) == comparable(committed), "re-record with `make cassettes`"
    assert fresh.footer == committed.footer


def test_ollama_toy_agent_runs_unchanged_without_a_session():
    output = ollama_agent.run(make_client(), "I'd like a refund for order A1002, please.")
    assert output.startswith("Sorry, order A1002 was delivered 45 days ago")


def _answer(client, task, *, stream):
    reply = client.chat("llama3.2", [{"role": "user", "content": task}], stream=stream)
    if stream:
        return "".join(chunk.message.content or "" for chunk in reply)
    return reply.message.content


@pytest.mark.parametrize("recorded_stream", [True, False])
def test_a_recording_serves_both_streaming_and_plain_calls(recorded_stream):
    def recorded(client, task):
        return _answer(client, task, stream=recorded_stream)

    def switched(client, task):
        return _answer(client, task, stream=not recorded_stream)

    first, cassette = record(recorded, "1 + 2?", client=FakeOllama(_two_turns))
    result = replay(Cassette.loads(cassette.dumps()), switched)

    assert result.ok, result.error
    assert result.output == first.output
    assert [s.status for s in result.trajectory.steps] == [StepStatus.EXACT]
