"""The Ollama provider: fake client, adapter, and the toy agent ported to Ollama."""

from __future__ import annotations

import asyncio

import ollama
import pytest
from ollama import ChatResponse, Message

from agent_replay.fake import Reply, ToolCall
from agent_replay.fake_ollama import FakeAsyncOllama, FakeOllama, FakeOllamaRequest


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
