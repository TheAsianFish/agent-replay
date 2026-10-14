"""Tests for the pairwise judge: request shape, parsing, stop reasons, the fake and the cache.

Offline: the Anthropic judge talks to ``FakeAnthropic`` behind a spy that records requests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from agent_replay.errors import ConfigError, JudgeCacheMiss, JudgeError
from agent_replay.fake import FakeAnthropic, Reply
from agent_replay.judge.pairwise import (
    PAIRWISE_INSTRUCTIONS,
    PAIRWISE_PROMPT_VERSION,
    AnthropicPairwiseJudge,
    FakePairwiseJudge,
    PairwiseCache,
    PairwiseJudge,
    PairwiseVerdict,
    make_pairwise_judge,
    pairwise_cache_key,
    render_pair,
)
from agent_replay.pricing import cost_usd

GOOD = json.dumps({"reasoning": "(b) does what was asked", "winner": "b"})


class SpyAnthropic:
    """``FakeAnthropic`` answering ``text`` that records each request; ``priced`` reports the
    requested model so the cost path runs, ``stop`` overrides the stop reason."""

    def __init__(self, text: str, *, priced: bool = True, stop: str | None = None) -> None:
        self._fake = FakeAnthropic(lambda _req: Reply(text=text))
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("REPLAY_JUDGE_MODEL", "REPLAY_JUDGE_EFFORT", "REPLAY_JUDGE_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


def test_the_request_shape():
    client = SpyAnthropic(GOOD)
    judge = AnthropicPairwiseJudge(client)

    judge.compare("Say hi.", "Hello there, friend!", "hi")

    [request] = client.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 8000
    assert request["output_config"]["effort"] == "low"
    fmt = request["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"reasoning", "winner"}
    assert fmt["schema"]["properties"]["winner"]["enum"] == ["a", "b"]
    assert fmt["schema"]["required"] == ["reasoning", "winner"]  # reasoning first
    assert fmt["schema"]["additionalProperties"] is False
    [system] = request["system"]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert system["text"] == PAIRWISE_INSTRUCTIONS  # stable across pairs, so it caches
    assert request["messages"] == [
        {"role": "user", "content": render_pair("Say hi.", "Hello there, friend!", "hi")}
    ]
    for forbidden in ("thinking", "budget_tokens", "tool_choice", "tools", "output_format"):
        assert forbidden not in request
    assert judge.judge_id == "anthropic:claude-opus-5-5:effort=low"


def test_the_pair_is_fenced_in_order():
    rendered = render_pair("I", "first", "second")

    assert rendered.index("<instruction>") < rendered.index("<output_a>\nfirst\n</output_a>")
    assert rendered.index("</output_a>") < rendered.index("<output_b>\nsecond\n</output_b>")


def test_the_verdict_is_parsed_and_priced():
    ticks = iter([2.0, 2.5])
    client = SpyAnthropic(GOOD)
    judge = AnthropicPairwiseJudge(client, effort="medium", timer=lambda: next(ticks))

    verdict = judge.compare("I", "a text", "b text")

    assert verdict.winner == "b"
    assert verdict.reasoning == "(b) does what was asked"
    assert verdict.latency_ms == 500.0
    assert verdict.usage is not None
    assert verdict.usage.input_tokens > 0
    assert verdict.cost_usd == cost_usd("claude-opus-5-5", verdict.usage)
    assert verdict.cost_usd is not None
    assert verdict.cost_usd > 0
    assert client.requests[0]["output_config"]["effort"] == "medium"
    assert judge.judge_id == "anthropic:claude-opus-5-5:effort=medium"


def test_an_unpriced_model_has_no_cost():
    verdict = AnthropicPairwiseJudge(SpyAnthropic(GOOD, priced=False)).compare("I", "a", "b")

    assert verdict.cost_usd is None


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REPLAY_JUDGE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("REPLAY_JUDGE_EFFORT", "high")

    assert AnthropicPairwiseJudge().judge_id == "anthropic:claude-sonnet-5-5:effort=high"
    with pytest.raises(ConfigError, match="effort='none'"):
        AnthropicPairwiseJudge(effort="none")
    with pytest.raises(ConfigError, match="max_tokens"):
        AnthropicPairwiseJudge(max_tokens=0)


@pytest.mark.parametrize(
    ("text", "stop", "fragment"),
    [
        (GOOD, "refusal", "refused"),
        (GOOD[:20], "max_tokens", "max_tokens=8000"),
        (GOOD, "model_context_window_exceeded", "context window"),
        ("not json", None, "pairwise verdict schema"),
        (json.dumps({"winner": "a"}), None, "reasoning"),
        (json.dumps({"reasoning": "r", "winner": "tie"}), None, "winner"),
    ],
)
def test_bad_answers_are_judge_errors(text: str, stop: str | None, fragment: str):
    with pytest.raises(JudgeError, match=fragment):
        AnthropicPairwiseJudge(SpyAnthropic(text, stop=stop)).compare("I", "a", "b")


def test_sdk_errors_become_judge_errors():
    class Failing:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **_kwargs: Any) -> None:
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request)

    with pytest.raises(JudgeError, match="APIConnectionError"):
        AnthropicPairwiseJudge(Failing()).compare("I", "a", "b")


def test_a_missing_api_key_is_a_judge_error_on_first_use(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    judge = AnthropicPairwiseJudge()  # constructing never needs a key

    with pytest.raises(JudgeError, match="ANTHROPIC_API_KEY"):
        judge.compare("I", "a", "b")


def test_the_fake_judge():
    default = FakePairwiseJudge()
    by_length = FakePairwiseJudge(
        lambda _i, a, b: ("a", "longer") if len(a) > len(b) else "b", name="longer"
    )

    assert default.compare("I", "x", "y").winner == "a"
    assert default.judge_id == "fake:first"
    long_first = by_length.compare("I", "long", "s")
    assert (long_first.winner, long_first.reasoning) == ("a", "longer")
    assert by_length.compare("I", "s", "long").winner == "b"
    assert (default.calls, by_length.calls) == (1, 2)
    with pytest.raises(JudgeError, match="return 'a' or 'b'"):
        FakePairwiseJudge(lambda *_: "c").compare("I", "x", "y")  # type: ignore[arg-type,return-value]


def test_make_pairwise_judge():
    assert isinstance(make_pairwise_judge("anthropic"), AnthropicPairwiseJudge)
    assert make_pairwise_judge("anthropic", effort="high").judge_id.endswith("effort=high")
    assert make_pairwise_judge("fake").judge_id == "fake:first"
    loaded = make_pairwise_judge("agent_replay.judge.pairwise:FakePairwiseJudge")
    assert loaded.judge_id == "fake:first"
    with pytest.raises(ConfigError, match="not a pairwise judge"):
        make_pairwise_judge("nope")
    with pytest.raises(ConfigError, match="model and effort"):
        make_pairwise_judge("fake", model="m")
    with pytest.raises(ConfigError, match="cannot load"):
        make_pairwise_judge("no_such_module_xyz:make")
    with pytest.raises(ConfigError, match="compare"):
        make_pairwise_judge("agent_replay.judge.pairwise:PairwiseCache")


# --- cache -------------------------------------------------------------------------------

NOW = datetime(2026, 10, 14, tzinfo=UTC)


def a_cache(tmp_path: Path, *, offline: bool = False) -> PairwiseCache:
    return PairwiseCache(tmp_path / "cache", offline=offline, clock=lambda: NOW)


def test_a_miss_calls_the_judge_and_a_hit_does_not(tmp_path: Path):
    cache = a_cache(tmp_path)
    judge = FakePairwiseJudge()

    first, hit_first = cache.verdict(judge, "I", "x", "y")
    second, hit_second = cache.verdict(judge, "I", "x", "y")

    assert (hit_first, hit_second) == (False, True)
    assert judge.calls == 1
    assert first == second
    key = pairwise_cache_key("fake:first", "I", "x", "y")
    stored = json.loads(cache.path(key).read_text(encoding="utf-8"))
    assert stored["prompt_version"] == PAIRWISE_PROMPT_VERSION
    assert stored["created_at"] == "2026-10-14T00:00:00Z"
    assert "x" not in stored.values()  # the pair is fingerprinted, not stored
    assert not list(cache.directory.rglob("*.tmp"))


def test_the_key_covers_order_judge_and_texts(tmp_path: Path):
    cache = a_cache(tmp_path)
    judge = FakePairwiseJudge()
    cache.verdict(judge, "I", "x", "y")

    assert cache.get("fake:first", "I", "y", "x") is None  # swapped order is another question
    assert cache.get("fake:other", "I", "x", "y") is None
    assert cache.get("fake:first", "J", "x", "y") is None
    keys = {
        pairwise_cache_key("j", "I", "x", "y"),
        pairwise_cache_key("j", "I", "y", "x"),
        pairwise_cache_key("j", "Ix", "", "y"),
        pairwise_cache_key("k", "I", "x", "y"),
    }
    assert len(keys) == 4


def test_offline_mode_raises_on_a_miss_and_serves_hits(tmp_path: Path):
    judge = FakePairwiseJudge()
    a_cache(tmp_path).verdict(judge, "I", "x", "y")
    offline = a_cache(tmp_path, offline=True)

    assert offline.verdict(judge, "I", "x", "y")[1] is True
    with pytest.raises(JudgeCacheMiss, match="offline mode is on"):
        offline.verdict(judge, "I", "y", "x")
    assert judge.calls == 1


def test_offline_defaults_to_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "1")

    assert PairwiseCache(tmp_path).offline is True


def test_corrupt_or_misplaced_entries_are_misses(tmp_path: Path):
    cache = a_cache(tmp_path)
    judge = FakePairwiseJudge()
    cache.verdict(judge, "I", "x", "y")
    cache.verdict(judge, "I", "y", "x")
    key_xy = pairwise_cache_key("fake:first", "I", "x", "y")
    key_yx = pairwise_cache_key("fake:first", "I", "y", "x")

    # An entry copied under another pair's key must not answer for that pair.
    cache.path(key_yx).write_text(cache.path(key_xy).read_text(encoding="utf-8"), "utf-8")
    assert cache.get("fake:first", "I", "y", "x") is None
    cache.path(key_xy).write_text("{not json", encoding="utf-8")
    assert cache.get("fake:first", "I", "x", "y") is None

    verdict, hit = cache.verdict(judge, "I", "x", "y")  # re-judged and rewritten
    assert (verdict.winner, hit, judge.calls) == ("a", False, 3)
    assert cache.get("fake:first", "I", "x", "y") is not None


def test_a_protocol_implementation_is_accepted():
    class Custom:
        judge_id = "custom"

        def compare(self, instruction: str, output_a: str, output_b: str, /) -> PairwiseVerdict:
            return PairwiseVerdict(reasoning=instruction, winner="b")

    judge: PairwiseJudge = Custom()
    assert judge.compare("why", "a", "b").winner == "b"
