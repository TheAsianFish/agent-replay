"""Tests for the judge: rubric, transcript, providers, cache, calibration gate and the check.

Everything runs offline: the Anthropic provider talks to ``FakeAnthropic``, the
Ollama provider to ``FakeOllama``, and suites use the retail example's fake judge.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import ollama
import pytest
import yaml
from pydantic import JsonValue
from structlog.testing import capture_logs

from agent_replay import Cassette, Kind, RunResult, Step, StepStatus, Trajectory, replay
from agent_replay.cassette import ErrorInfo
from agent_replay.errors import ConfigError, JudgeError
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply
from agent_replay.fake_ollama import FakeOllama
from agent_replay.judge import (
    PROMPT_VERSION,
    AnthropicJudge,
    CacheEntry,
    CalibrationReport,
    CriterionVerdict,
    FakeJudge,
    JudgeCache,
    JudgeInput,
    JudgeOutput,
    OllamaJudge,
    Rubric,
    gating,
    make_provider,
)
from agent_replay.judge.cache import cache_key
from agent_replay.judge.calibration import resolve_min_kappa
from agent_replay.judge.transcript import clip
from agent_replay.pricing import cost_usd
from examples.retail_agent import agent as retail_agent
from examples.retail_agent import tools as retail_tools

ROOT = Path(__file__).parents[1]
RETAIL = ROOT / "examples" / "retail_agent"

RUBRIC_DATA: dict[str, Any] = {
    "name": "refunds",
    "version": 1,
    "description": "Refund handling.",
    "instructions": "Be strict about policy.",
    "criteria": [
        {"id": "resolved", "question": "Was the request resolved?", "pass_when": "it was"},
        {"id": "policy", "question": "Was the policy followed?", "fail_when": "it was not"},
    ],
}


def write_yaml(tmp_path: Path, data: object, name: str = "rubric.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def make_rubric(**changes: Any) -> Rubric:
    return Rubric.model_validate({**RUBRIC_DATA, **changes})


# --- rubric ---------------------------------------------------------------------


def test_a_rubric_loads_from_yaml(tmp_path: Path):
    rubric = Rubric.load(write_yaml(tmp_path, RUBRIC_DATA))

    assert rubric.name == "refunds"
    assert rubric.ids == ("resolved", "policy")
    assert rubric.criteria[0].pass_when == "it was"
    assert rubric.fingerprint.startswith("sha256:")
    text = rubric.render()
    assert "# Rubric: refunds (version 1)" in text
    assert "Be strict about policy." in text
    assert "### policy" in text
    assert "Fail when: it was not" in text


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"name": "Bad Name"}, "name:"),
        ({"version": 0}, "version:"),
        ({"criteria": []}, "criteria:"),
        ({"criteria": [{"id": f"c{i}", "question": "q?"} for i in range(11)]}, "criteria:"),
        ({"criteria": [{"id": "Bad", "question": "q?"}]}, "criteria.0.id:"),
        ({"criteria": [{"id": "a", "question": "  "}]}, "criteria.0.question:"),
        ({"criteria": [{"id": "a"}]}, "criteria.0.question: Field required"),
        (
            {"criteria": [{"id": "a", "question": "q?"}, {"id": "a", "question": "r?"}]},
            "repeated: a",
        ),
        (
            {"criteria": [{"id": "a", "question": "q?", "weight": 2}]},
            "criteria.0.weight: unknown field; use id, question",
        ),
        ({"extra": 1}, "extra: unknown field; use name, version"),
    ],
)
def test_rubric_validation_names_the_file_and_field(tmp_path: Path, changes, fragment):
    path = write_yaml(tmp_path, {**RUBRIC_DATA, **changes})

    with pytest.raises(ConfigError) as info:
        Rubric.load(path)

    assert str(path) in str(info.value)
    assert fragment in str(info.value)


def test_a_missing_or_malformed_rubric_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="does not exist"):
        Rubric.load(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Rubric.load(bad)
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        Rubric.load(write_yaml(tmp_path, ["a", "list"]))


def test_any_edit_changes_the_rubric_fingerprint():
    base = make_rubric()
    edited_question = make_rubric(
        criteria=[
            {"id": "resolved", "question": "Was it fully resolved?", "pass_when": "it was"},
            RUBRIC_DATA["criteria"][1],
        ]
    )

    assert make_rubric().fingerprint == base.fingerprint
    assert make_rubric(version=2).fingerprint != base.fingerprint
    assert make_rubric(instructions="Be lenient.").fingerprint != base.fingerprint
    assert edited_question.fingerprint != base.fingerprint


# --- transcript -----------------------------------------------------------------


def replay_retail(task_id: str, agent: Any = None) -> RunResult:
    cassette = Cassette.load(RETAIL / "cassettes" / f"{task_id}.jsonl")
    retail_tools.reset_db()
    try:
        return replay(cassette, agent or retail_agent.run)
    finally:
        retail_tools.reset_db()


def test_the_transcript_of_a_replayed_retail_run():
    run = replay_retail("cancel_pending_laptop")
    judge_input = JudgeInput.from_run(run)

    assert judge_input.task == run.task
    assert judge_input.output == run.output
    assert judge_input.error is None
    assert [c.name for c in judge_input.tool_calls] == run.trajectory.tool_names()
    first = judge_input.tool_calls[0]
    assert first.input == {"first_name": "Emma", "last_name": "Smith", "zip": "10192"}
    assert first.result == "emma_smith_8564"
    assert judge_input.tool_call_count == len(run.trajectory.tool_steps())
    assert judge_input.model_call_count == len(run.trajectory.model_steps())
    text = judge_input.render()
    assert "<task>\n" + run.task in text
    assert "## Call 1: find_user_id_by_name_zip" in text
    assert '"first_name": "Emma"' in text  # pretty-printed JSON
    assert "<final_output>\nDone: order #W2417020 is now cancelled.\n</final_output>" in text
    # The same run gives the same transcript, so a cached verdict applies.
    again = JudgeInput.from_run(replay_retail("cancel_pending_laptop"))
    assert again == judge_input
    assert again.fingerprint == judge_input.fingerprint


def test_long_values_are_clipped_with_a_visible_marker():
    run = replay_retail("exchange_waterproof_boots")
    judge_input = JudgeInput.from_run(run)

    product = next(c for c in judge_input.tool_calls if c.name == "get_product_details")
    assert isinstance(product.result, str)
    assert product.result.endswith("…[truncated 891 chars]")
    assert len(product.result) == 2000 + len("…[truncated 891 chars]")
    roomy = JudgeInput.from_run(run, max_chars=10_000)
    roomy_product = next(c for c in roomy.tool_calls if c.name == "get_product_details")
    assert isinstance(roomy_product.result, dict)
    assert roomy.fingerprint != judge_input.fingerprint


def test_clip():
    assert clip("abcdef", 3) == "abc…[truncated 3 chars]"
    assert clip("abc", 3) == "abc"
    assert clip({"a": 1}, 100) == {"a": 1}
    assert clip([1, 2, 3], 4) == "[1, …[truncated 5 chars]"
    with pytest.raises(ValueError, match="max_chars"):
        JudgeInput.from_run(make_run(), max_chars=0)


Call = tuple[str, JsonValue, JsonValue, ErrorInfo | None]


def make_run(
    tools: tuple[Call, ...] = (),
    *,
    output: JsonValue = "done",
    error: ErrorInfo | None = None,
    task: str = "Refund order A1",
) -> RunResult:
    steps = [
        Step(
            index=i,
            kind=Kind.TOOL,
            name=name,
            status=StepStatus.EXACT,
            fingerprint="fp",
            request=request,
            response=response,
            error=step_error,
        )
        for i, (name, request, response, step_error) in enumerate(tools)
    ]
    return RunResult(
        mode="full",
        task=task,
        output=output,
        error=error,
        trajectory=Trajectory(steps=tuple(steps)),
    )


def test_errors_appear_in_the_transcript():
    run = make_run(
        (("lookup", {"id": "A1"}, None, ErrorInfo(type="ValueError", message="no such order")),),
        output=None,
        error=ErrorInfo(type="RuntimeError", message="gave up"),
    )
    judge_input = JudgeInput.from_run(run)

    assert judge_input.tool_calls[0].error is not None
    assert judge_input.tool_calls[0].result is None
    text = judge_input.render()
    assert "Error (ValueError): no such order" in text
    assert "<agent_error>\nThe agent did not finish: RuntimeError: gave up" in text
    assert "(the agent made no tool calls)" in JudgeInput.from_run(make_run()).render()


# --- providers ------------------------------------------------------------------

JUDGE_ENV = (
    "REPLAY_JUDGE_PROVIDER",
    "REPLAY_JUDGE_MODEL",
    "REPLAY_JUDGE_EFFORT",
    "REPLAY_JUDGE_MIN_KAPPA",
    "REPLAY_JUDGE_CACHE",
    "REPLAY_JUDGE_OFFLINE",
)


@pytest.fixture(autouse=True)
def _clean_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in JUDGE_ENV:
        monkeypatch.delenv(name, raising=False)


def answer(*verdicts: tuple[str, str], reasoning: str = "because") -> str:
    return json.dumps(
        {"criteria": [{"id": i, "reasoning": reasoning, "verdict": v} for i, v in verdicts]}
    )


GOOD_ANSWER = answer(("resolved", "pass"), ("policy", "fail"))


class SpyAnthropic:
    """FakeAnthropic behind ``messages.create`` that records requests and can tweak replies.

    The fake reports its model as ``fake:<model>`` (never priced); ``priced``
    restores the requested model so the cost path runs.
    """

    def __init__(
        self,
        text: str | Callable[[FakeRequest], str],
        *,
        priced: bool = True,
        stop: str | None = None,
    ) -> None:
        policy = text if callable(text) else (lambda _req: text)
        self._fake = FakeAnthropic(lambda req: Reply(text=policy(req)))
        self.requests: list[dict[str, Any]] = []
        self.priced = priced
        self.stop = stop
        self.messages = self

    def create(self, **kwargs: Any) -> anthropic.types.Message:
        self.requests.append(kwargs)
        message = self._fake.messages.create(**kwargs)
        update: dict[str, Any] = {}
        if self.priced:
            update["model"] = kwargs["model"]
        if self.stop is not None:
            update["stop_reason"] = self.stop
        return message.model_copy(update=update)


def a_transcript() -> JudgeInput:
    return JudgeInput.from_run(make_run((("lookup", {"id": "A1"}, {"status": "ok"}, None),)))


def test_the_anthropic_judge_request():
    client = SpyAnthropic(GOOD_ANSWER)
    judge = AnthropicJudge(client)
    rubric, judge_input = make_rubric(), a_transcript()

    judge.evaluate(rubric, judge_input)

    [request] = client.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 8000
    assert request["output_config"]["effort"] == "low"
    fmt = request["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "criteria" in fmt["schema"]["properties"]
    assert fmt["schema"]["additionalProperties"] is False
    [system] = request["system"]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert system["text"].startswith("You are an impartial evaluator")
    assert rubric.render() in system["text"]  # the stable content, cached across runs
    assert request["messages"] == [{"role": "user", "content": judge_input.render()}]
    for forbidden in ("thinking", "budget_tokens", "tool_choice", "tools", "output_format"):
        assert forbidden not in request
    assert judge.judge_id == "anthropic:claude-opus-5-5:effort=low"


def test_the_anthropic_judge_parses_and_prices_the_verdict():
    ticks = iter([1.0, 1.25])
    client = SpyAnthropic(GOOD_ANSWER)
    judge = AnthropicJudge(
        client, model="claude-opus-5-5", effort="high", timer=lambda: next(ticks)
    )

    response = judge.evaluate(make_rubric(), a_transcript())

    assert [(v.id, v.verdict) for v in response.verdicts] == [
        ("resolved", "pass"),
        ("policy", "fail"),
    ]
    assert response.verdicts[0].reasoning == "because"
    assert response.latency_ms == 250.0
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.cost_usd == cost_usd("claude-opus-5-5", response.usage)
    assert response.cost_usd is not None
    assert response.cost_usd > 0
    assert client.requests[0]["output_config"]["effort"] == "high"
    assert judge.judge_id == "anthropic:claude-opus-5-5:effort=high"


def test_an_unpriced_judge_model_has_no_cost():
    response = AnthropicJudge(SpyAnthropic(GOOD_ANSWER, priced=False)).evaluate(
        make_rubric(), a_transcript()
    )

    assert response.cost_usd is None


def test_verdicts_come_back_in_rubric_order():
    reordered = answer(("policy", "pass"), ("resolved", "fail"))

    response = AnthropicJudge(SpyAnthropic(reordered)).evaluate(make_rubric(), a_transcript())

    assert [v.id for v in response.verdicts] == ["resolved", "policy"]


@pytest.mark.parametrize(
    ("text", "stop", "fragment"),
    [
        (GOOD_ANSWER, "refusal", "refused"),
        (GOOD_ANSWER[:40], "max_tokens", "max_tokens=8000"),
        (GOOD_ANSWER, "model_context_window_exceeded", "context window"),
        ("not json", None, "does not match the verdict schema"),
        (json.dumps({"criteria": [{"id": "resolved", "verdict": "pass"}]}), None, "reasoning"),
        (answer(("resolved", "pass")), None, "missing policy"),
        (
            answer(("resolved", "pass"), ("policy", "pass"), ("policy", "fail")),
            None,
            "repeated policy",
        ),
        (answer(("resolved", "pass"), ("policy", "pass"), ("tone", "pass")), None, "unknown tone"),
        (answer(("resolved", "maybe"), ("policy", "pass")), None, "verdict"),
    ],
)
def test_bad_judge_answers_are_judge_errors(text, stop, fragment):
    judge = AnthropicJudge(SpyAnthropic(text, stop=stop))

    with pytest.raises(JudgeError, match=fragment):
        judge.evaluate(make_rubric(), a_transcript())


def test_sdk_errors_become_judge_errors():
    class Failing:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **_kwargs: Any) -> None:
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request)

    with pytest.raises(JudgeError, match="APIConnectionError") as info:
        AnthropicJudge(Failing()).evaluate(make_rubric(), a_transcript())
    assert "REPLAY_JUDGE_OFFLINE" in str(info.value)


def test_a_missing_api_key_is_a_judge_error_on_first_use(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    judge = AnthropicJudge()  # constructing never needs a key

    with pytest.raises(JudgeError, match="ANTHROPIC_API_KEY"):
        judge.evaluate(make_rubric(), a_transcript())


def test_anthropic_judge_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REPLAY_JUDGE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("REPLAY_JUDGE_EFFORT", "medium")
    judge = AnthropicJudge(SpyAnthropic(GOOD_ANSWER))

    assert judge.judge_id == "anthropic:claude-sonnet-5-5:effort=medium"
    assert AnthropicJudge(model="claude-opus-5-5", effort="max").judge_id.endswith("effort=max")
    monkeypatch.setenv("REPLAY_JUDGE_EFFORT", "extreme")
    with pytest.raises(ConfigError, match=r"REPLAY_JUDGE_EFFORT='extreme'.*low, medium"):
        AnthropicJudge()
    with pytest.raises(ConfigError, match="effort='none'"):
        AnthropicJudge(effort="none")
    with pytest.raises(ConfigError, match="max_tokens"):
        AnthropicJudge(effort="low", max_tokens=0)


class SpyOllama(FakeOllama):
    def __init__(self, text: str) -> None:
        super().__init__(lambda _req: Reply(text=text))
        self.requests: list[dict[str, Any]] = []

    def chat(self, model: str = "", messages: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        self.requests.append({"model": model, "messages": messages, **kwargs})
        return super().chat(model, messages, **kwargs)


def test_the_ollama_judge():
    client = SpyOllama(GOOD_ANSWER)
    judge = OllamaJudge(client)
    rubric, judge_input = make_rubric(), a_transcript()

    response = judge.evaluate(rubric, judge_input)

    assert judge.judge_id == "ollama:qwen3:8b"
    assert [v.verdict for v in response.verdicts] == ["pass", "fail"]
    assert response.cost_usd == 0.0
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    [request] = client.requests
    assert request["model"] == "qwen3:8b"
    assert request["format"] == JudgeOutput.model_json_schema()
    assert request["options"] == {"temperature": 0}
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert rubric.render() in request["messages"][0]["content"]
    assert request["messages"][1]["content"] == judge_input.render()


def test_ollama_judge_failures_name_the_fix():
    class Down(FakeOllama):
        def chat(self, *_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
            raise ConnectionError("Failed to connect to Ollama")

    class NoModel(FakeOllama):
        def chat(self, *_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
            raise ollama.ResponseError("model 'qwen3:8b' not found", 404)

    with pytest.raises(JudgeError, match="ollama serve"):
        OllamaJudge(Down(lambda _r: Reply())).evaluate(make_rubric(), a_transcript())
    with pytest.raises(JudgeError, match="ollama pull qwen3:8b"):
        OllamaJudge(NoModel(lambda _r: Reply())).evaluate(make_rubric(), a_transcript())
    with pytest.raises(JudgeError, match="verdict schema"):
        OllamaJudge(SpyOllama("{}")).evaluate(make_rubric(), a_transcript())


def test_the_ollama_judge_without_the_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "ollama", None)

    with pytest.raises(JudgeError, match=re.escape("pip install 'agent-replay[ollama]'")):
        OllamaJudge(model="llama3").evaluate(make_rubric(), a_transcript())


def test_the_fake_judge():
    def decide(rubric: Rubric, judge_input: JudgeInput) -> dict[str, bool | tuple[bool, str]]:
        return {"resolved": judge_input.output == "done", "policy": (False, "no lookup")}

    judge = FakeJudge(decide, name="rules")
    response = judge.evaluate(make_rubric(), a_transcript())

    assert judge.judge_id == "fake:rules"
    assert judge.calls == 1
    assert response.verdicts == (
        CriterionVerdict(id="resolved", reasoning="decided by the fake judge", verdict="pass"),
        CriterionVerdict(id="policy", reasoning="no lookup", verdict="fail"),
    )
    assert response.cost_usd == 0.0
    assert all(v.passed for v in FakeJudge().evaluate(make_rubric(), a_transcript()).verdicts)
    with pytest.raises(JudgeError, match="missing policy"):
        FakeJudge(lambda _r, _i: {"resolved": True}).evaluate(make_rubric(), a_transcript())


def a_provider() -> FakeJudge:
    return FakeJudge(name="from-factory")


def a_failing_factory() -> FakeJudge:
    raise RuntimeError("boom")


def test_make_provider(monkeypatch: pytest.MonkeyPatch):
    assert isinstance(make_provider(), AnthropicJudge)
    assert isinstance(make_provider("ollama", model="llama3"), OllamaJudge)
    assert make_provider("fake").judge_id == "fake:default"
    assert make_provider("tests.test_judge:a_provider").judge_id == "fake:from-factory"
    assert make_provider("anthropic", model="claude-sonnet-5-5", effort="high").judge_id == (
        "anthropic:claude-sonnet-5-5:effort=high"
    )
    monkeypatch.setenv("REPLAY_JUDGE_PROVIDER", "ollama")
    assert isinstance(make_provider(), OllamaJudge)
    monkeypatch.setenv("REPLAY_JUDGE_PROVIDER", "openai")
    with pytest.raises(
        ConfigError, match=r"REPLAY_JUDGE_PROVIDER='openai'.*anthropic, ollama, fake"
    ):
        make_provider()


@pytest.mark.parametrize(
    ("spec", "kwargs", "fragment"),
    [
        ("gpt", {}, "provider='gpt' is not a judge provider"),
        ("ollama", {"effort": "low"}, "effort applies to the anthropic judge only"),
        ("fake", {"model": "x"}, "model applies to the anthropic and ollama judges"),
        ("tests.nope:factory", {}, "cannot load judge provider"),
        ("tests.test_judge:RUBRIC_DATA", {}, "is not callable"),
        ("tests.test_judge:make_rubric", {}, "returned Rubric"),
        ("tests.test_judge:a_failing_factory", {}, "raised RuntimeError: boom"),
    ],
)
def test_make_provider_errors(spec, kwargs, fragment):
    with pytest.raises(ConfigError, match=re.escape(fragment)):
        make_provider(spec, **kwargs)


# --- cache ----------------------------------------------------------------------


def test_a_cache_hit_skips_the_provider(tmp_path: Path):
    cache = JudgeCache(tmp_path / "cache")
    provider = FakeJudge(lambda _r, _i: {"resolved": True, "policy": (False, "no lookup")})
    rubric, judge_input = make_rubric(), a_transcript()

    first, hit_first = cache.verdict(provider, rubric, judge_input)
    second, hit_second = cache.verdict(provider, rubric, judge_input)

    assert (hit_first, hit_second) == (False, True)
    assert provider.calls == 1
    assert second == first
    key = cache_key(provider.judge_id, rubric, judge_input)
    path = tmp_path / "cache" / key[:2] / f"{key}.json"
    assert cache.path(key) == path
    entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
    assert entry.judge_id == "fake:default"
    assert entry.rubric_fingerprint == rubric.fingerprint
    assert entry.input_fingerprint == judge_input.fingerprint
    assert entry.prompt_version == PROMPT_VERSION
    assert [v.verdict for v in entry.verdicts] == ["pass", "fail"]
    assert list(tmp_path.glob("cache/*/*.tmp")) == []  # written atomically


def test_the_cache_key_covers_judge_rubric_and_transcript(tmp_path: Path):
    cache = JudgeCache(tmp_path)
    provider = FakeJudge()
    rubric, judge_input = make_rubric(), a_transcript()
    other_input = JudgeInput.from_run(make_run(output="something else"))

    for args in [
        (provider, rubric, judge_input),
        (FakeJudge(name="other"), rubric, judge_input),
        (provider, make_rubric(version=2), judge_input),
        (provider, rubric, other_input),
    ]:
        assert cache.verdict(*args)[1] is False
    assert cache.verdict(provider, rubric, judge_input)[1] is True
    assert provider.calls == 3


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"key": "x"}',
        None,  # a valid entry whose verdicts do not cover the rubric
    ],
)
def test_a_corrupt_cache_file_is_a_logged_miss(tmp_path: Path, content: str | None):
    cache = JudgeCache(tmp_path)
    provider = FakeJudge()
    rubric, judge_input = make_rubric(), a_transcript()
    cache.verdict(provider, rubric, judge_input)
    path = cache.path(cache_key(provider.judge_id, rubric, judge_input))
    if content is None:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["verdicts"] = data["verdicts"][:1]
        content = json.dumps(data)
    path.write_text(content, encoding="utf-8")

    with capture_logs() as logs:
        _, hit = cache.verdict(provider, rubric, judge_input)

    assert hit is False
    assert provider.calls == 2
    assert [entry["event"] for entry in logs] == ["judge.cache_corrupt"]
    assert cache.verdict(provider, rubric, judge_input)[1] is True  # rewritten


def test_an_unwritable_cache_still_returns_the_verdict(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = JudgeCache(blocker / "cache")

    with capture_logs() as logs:
        response, hit = cache.verdict(FakeJudge(), make_rubric(), a_transcript())

    assert hit is False
    assert len(response.verdicts) == 2
    assert "judge.cache_write_failed" in [entry["event"] for entry in logs]


def test_offline_mode_judges_only_from_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider = FakeJudge()
    rubric, judge_input = make_rubric(), a_transcript()
    offline = JudgeCache(tmp_path, offline=True)

    with pytest.raises(JudgeError, match=r"not in the judge cache.*commit the cache directory"):
        offline.verdict(provider, rubric, judge_input)
    assert provider.calls == 0

    JudgeCache(tmp_path).verdict(provider, rubric, judge_input)
    assert offline.verdict(provider, rubric, judge_input)[1] is True
    assert provider.calls == 1

    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "true")
    assert JudgeCache(tmp_path).offline is True
    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "0")
    assert JudgeCache(tmp_path).offline is False
    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "sometimes")
    with pytest.raises(ConfigError, match="REPLAY_JUDGE_OFFLINE='sometimes' is not a boolean"):
        JudgeCache(tmp_path)


def test_the_cache_directory_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert JudgeCache().directory == Path(".replay/judge-cache")
    monkeypatch.setenv("REPLAY_JUDGE_CACHE", str(tmp_path / "shared"))
    assert JudgeCache().directory == tmp_path / "shared"


# --- calibration gate -----------------------------------------------------------

FAKE_ID = "fake:default"


def calibration(
    rubric: Rubric | None = None,
    *,
    judge_id: str = FAKE_ID,
    ci_low: dict[str, float] | None = None,
) -> CalibrationReport:
    rubric = rubric or make_rubric()
    lows = ci_low if ci_low is not None else dict.fromkeys(rubric.ids, 0.75)
    return CalibrationReport.model_validate(
        {
            "rubric": rubric.name,
            "rubric_fingerprint": rubric.fingerprint,
            "judge_id": judge_id,
            "created_at": "2026-10-09T00:00:00Z",
            "n_items": 120,
            "method": "Cohen's kappa, 95% bootstrap CI (2000 resamples)",
            "criteria": {
                i: {
                    "kappa": min(1.0, low + 0.05),
                    "ci_low": low,
                    "ci_high": min(1.0, low + 0.1),
                    "agreement": 0.9,
                    "n": 120,
                }
                for i, low in lows.items()
            },
        }
    )


def write_calibration(tmp_path: Path, report: CalibrationReport) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_without_a_report_every_criterion_is_advisory():
    gates = gating(None, make_rubric(), FAKE_ID, 0.6)

    assert set(gates) == {"resolved", "policy"}
    assert not any(g.gating for g in gates.values())
    assert "calibrate the judge against human labels first" in gates["resolved"].reason


def test_a_report_for_another_rubric_version_or_judge_gates_nothing():
    rubric = make_rubric()

    other_version = gating(calibration(make_rubric(version=2)), rubric, FAKE_ID, 0.6)
    other_judge = gating(calibration(judge_id="fake:other"), rubric, FAKE_ID, 0.6)

    assert not any(g.gating for g in other_version.values())
    assert "another version of rubric 'refunds'" in other_version["policy"].reason
    assert not any(g.gating for g in other_judge.values())
    assert "for judge fake:other, not fake:default" in other_judge["policy"].reason


@pytest.mark.parametrize(
    ("ci_low", "gates_resolved"),
    [(0.59, False), (0.6, True), (0.61, True)],
)
def test_a_criterion_gates_when_its_kappa_lower_bound_reaches_the_threshold(ci_low, gates_resolved):
    gates = gating(
        calibration(ci_low={"resolved": ci_low, "policy": 0.9}), make_rubric(), FAKE_ID, 0.6
    )

    assert gates["resolved"].gating is gates_resolved
    assert gates["policy"].gating is True
    if not gates_resolved:
        assert "kappa lower bound 0.59 < 0.60 (n=120)" in gates["resolved"].reason
    else:
        assert gates["resolved"].reason.startswith("calibrated: kappa lower bound")


def test_mixed_gating_and_advisory_criteria():
    gates = gating(calibration(ci_low={"resolved": 0.8}), make_rubric(), FAKE_ID, 0.6)

    assert gates["resolved"].gating is True
    assert gates["policy"].gating is False
    assert "not in the calibration report" in gates["policy"].reason


def test_calibration_reports_load_and_validate(tmp_path: Path):
    report = calibration()
    path = write_calibration(tmp_path, report)

    assert CalibrationReport.load(path) == report
    with pytest.raises(ConfigError, match="does not exist"):
        CalibrationReport.load(tmp_path / "missing.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    data["criteria"]["resolved"]["ci_low"] = 1.5
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        CalibrationReport.load(path)
    assert "schema_version" in str(info.value)
    assert "criteria.resolved.ci_low" in str(info.value)


def test_min_kappa_comes_from_the_argument_or_the_environment(monkeypatch: pytest.MonkeyPatch):
    assert resolve_min_kappa(None) == 0.6
    assert resolve_min_kappa(0.8) == 0.8
    monkeypatch.setenv("REPLAY_JUDGE_MIN_KAPPA", "0.7")
    assert resolve_min_kappa(None) == 0.7
    assert resolve_min_kappa(0.0) == 0.0
    monkeypatch.setenv("REPLAY_JUDGE_MIN_KAPPA", "high")
    with pytest.raises(ConfigError, match="REPLAY_JUDGE_MIN_KAPPA='high' is not a number"):
        resolve_min_kappa(None)
    monkeypatch.setenv("REPLAY_JUDGE_MIN_KAPPA", "1.5")
    with pytest.raises(ConfigError, match=r"REPLAY_JUDGE_MIN_KAPPA=1\.5 is out of range"):
        resolve_min_kappa(None)
    with pytest.raises(ConfigError, match=r"min_kappa=-0\.1 is out of range"):
        resolve_min_kappa(-0.1)
    with pytest.raises(ConfigError, match="out of range"):
        resolve_min_kappa(float("nan"))
