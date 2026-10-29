"""The M16 harness (``eval/pydantic_ai.py --long``), offline against the fake model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from anthropic import AsyncAnthropic

from agent_replay.fake import FakeAsyncAnthropic, FakeRequest, Reply, ToolCall
from eval import pydantic_ai
from examples.pydantic_ai_agent import fake_model
from examples.pydantic_ai_agent.tasks import task_for
from examples.pydantic_ai_agent.tools import (
    CHAIN,
)

FAKE = "examples.pydantic_ai_agent.fake_model:make_client"
HERE = "tests.test_pydantic_ai_long"
EXAMPLE = pydantic_ai.EXAMPLE


def _miscopied_link(request: FakeRequest) -> Reply:
    """The fake model, except that it copies the fourth link's reference in lower case:
    the same booking to the tool, another entity to the suite's identify keys."""
    reply = fake_model.policy(request)
    if task_for(request.first_user_text()).name != pydantic_ai.LONG_SCENARIO:
        return reply
    ref = CHAIN[3]
    calls = [
        ToolCall(c.name, {"booking_ref": ref.lower()})
        if c.name == "get_booking" and c.input == {"booking_ref": ref}
        else c
        for c in reply.tool_calls
    ]
    return Reply(text=reply.text, tool_calls=calls)


def miscopied_link_client() -> AsyncAnthropic:
    return FakeAsyncAnthropic(_miscopied_link).as_anthropic()


def _long(tmp_path: Path, client: str = FAKE, runs: int = 2) -> dict[str, Any]:
    out = tmp_path / "long.json"
    result = pydantic_ai.run_long(runs, out, client=client, environment={})
    assert json.loads(out.read_text(encoding="utf-8")) == result
    return result


def _arm(result: dict[str, Any], arm: str) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = result["arms"][arm]["scenarios"]
    (scenario,) = scenarios
    assert scenario["name"] == pydantic_ai.LONG_SCENARIO
    return scenario


def test_the_fake_model_walks_the_chain_exactly_under_both_arms(tmp_path: Path) -> None:
    result = _long(tmp_path)
    assert result["recorded_tool_calls"] == {"convert_currency": 1, "get_booking": 8}
    assert list(result["arms"]) == ["strict", "fallback"]
    for arm in ("strict", "fallback"):
        s = _arm(result, arm)
        assert (s["runs"], s["passes"]) == (2, 2)
        long = s["long"]
        assert long["served_calls_per_run"] == [9, 9]
        assert long["runs_with_a_world_step"] == long["identity_firings"] == 0
        assert long["world_endings"]["runs"] == 0
        assert long["clustering"]["measurable"] is False
        assert [p["survival"] for p in long["survival"]] == [1.0] * 9
        assert long["survival"][-1]["independent"] == round((1 - 3 / 143) ** 9, 4)
        for row in s["outcomes"]:
            assert row["statuses"] == ["exact"] * 9
            assert row["first_world_step"] is None


def test_a_miscopied_link_ends_strict_by_identity_and_runs_the_tool_under_fallback(
    tmp_path: Path,
) -> None:
    result = _long(tmp_path, f"{HERE}:miscopied_link_client")
    strict = _arm(result, "strict")
    assert strict["causes"]["unrecorded"] == strict["unrecorded_by_identity"] == 2
    for row in strict["outcomes"]:
        assert row["statuses"] == ["exact"] * 3 + ["unrecorded"]
        assert (row["first_world_step"], row["identity_firings"]) == (4, 1)
    long = strict["long"]
    assert long["world_endings"]["runs"] == 2
    assert long["identity_firings"] == 2
    assert [p["at_risk"] for p in long["survival"]] == [2, 2, 2, 2]
    assert long["survival"][3]["survival"] == 0.0

    fallback = _arm(result, "fallback")
    # The real tool upper-cases the reference, so the run goes on and passes; the rest of
    # the chain is served exact, since nothing consumed the recording of link four.
    assert fallback["passes"] == fallback["runs_with_a_fallback_step"] == 2
    for row in fallback["outcomes"]:
        assert row["statuses"] == ["exact"] * 3 + ["fallback"] + ["exact"] * 5
        assert row["identity_firings"] == 1
    assert fallback["long"]["world_endings"]["runs"] == 0
    assert fallback["long"]["clustering"]["world_steps"] == 2


def test_identity_firings_count_only_calls_with_recordings_left() -> None:
    def calls(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
        return [{"name": n, "status": s} for n, s in pairs]

    recorded = Counter({"get_booking": 2, "convert_currency": 1})
    # A wrong reference while recordings of get_booking remain: the keys turned it away.
    assert pydantic_ai.identity_firings(calls(("get_booking", "fallback")), recorded) == 1
    # A third get_booking after both recordings were served is off the path by count.
    path = calls(("get_booking", "exact"), ("get_booking", "exact"), ("get_booking", "fallback"))
    assert pydantic_ai.identity_firings(path, recorded) == 0
    # A tool the cassette never recorded is not an identity firing either.
    assert pydantic_ai.identity_firings(calls(("search_flights", "fallback")), recorded) == 0
    # An approximate step was served (the keys matched), so it is not a firing.
    assert pydantic_ai.identity_firings(calls(("convert_currency", "approximate")), recorded) == 0


def test_survival_leaves_runs_that_ended_out_of_the_risk_set() -> None:
    sequences = [
        ["exact", "unrecorded"],  # world step at 2
        ["exact"],  # finished after 1 call, no world step
        ["exact", "exact", "exact"],
        ["fallback", "exact", "exact"],  # world step at 1
    ]
    curve = pydantic_ai.survival_curve(sequences, per_call=0.1)
    assert [(p["at_risk"], p["world_steps"]) for p in curve] == [(4, 1), (2, 1), (1, 0)]
    assert [p["survival"] for p in curve] == [0.75, 0.375, 0.375]
    # The plain share counts the run that finished early as a survivor.
    assert [p["share_without_world_step"] for p in curve] == [0.75, 0.5, 0.5]
    assert [p["independent"] for p in curve] == [0.9, 0.81, 0.729]


def test_clustering_compares_steps_after_a_world_step_with_all_steps() -> None:
    clustered = [["exact"] * 8 + ["fallback", "fallback", "fallback"]] + [["exact"] * 10] * 4
    c = pydantic_ai.clustering(clustered, minimum=3)
    assert (c["tool_steps"], c["world_steps"]) == (51, 3)
    assert (c["steps_after_a_world_step"], c["world_steps_after_a_world_step"]) == (2, 2)
    assert c["base_rate"] == round(3 / 51, 4)
    assert c["conditional_rate"] == 1.0
    assert c["measurable"] is True
    assert c["clusters"] is True
    assert c["ratio"] == 17.0
    spread = [["fallback"] + ["exact"] * 9 for _ in range(5)]
    s = pydantic_ai.clustering(spread)
    assert (s["world_steps"], s["world_steps_after_a_world_step"], s["ratio"]) == (5, 0, 0.0)
    assert s["clusters"] is False
    # Below the minimum the ratio is not measurable, not refuted.
    few = pydantic_ai.clustering(clustered)
    assert (few["measurable"], few["ratio"], few["clusters"]) == (False, None, None)


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    # Clopper-Pearson, as scipy.stats.binomtest(k, n).proportion_ci("exact") gives it.
    [(0, 20, (0.0, 0.1684)), (3, 20, (0.0321, 0.3789)), (20, 20, (0.8316, 1.0))],
)
def test_exact_interval(k: int, n: int, expected: tuple[float, float]) -> None:
    assert pydantic_ai.exact_interval(k, n) == pytest.approx(expected, abs=1e-4)


def test_exact_interval_against_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for n in (15, 20):
        for k in range(n + 1):
            ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
            got = pydantic_ai.exact_interval(k, n)
            assert got == pytest.approx((ci.low, ci.high), abs=1e-4)


def test_binomial_range_covers_the_prediction() -> None:
    p = 1 - (1 - 3 / 143) ** 9
    assert round(p * 20, 2) == 3.47
    assert pydantic_ai.binomial_range(20, p) == (1, 7)


def test_runs_per_arm_drop_to_fifteen_when_the_recording_was_slow(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    assert pydantic_ai.long_runs(log) == 20
    attempts = [{"passed": False, "wall_s": 400.0}, {"passed": True, "wall_s": 90.0}]
    log.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    assert pydantic_ai.long_runs(log) == 20  # the kept attempt decides
    attempts[1]["wall_s"] = 151.0
    log.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    assert pydantic_ai.long_runs(log) == 15
    log.write_text(json.dumps({"attempts": attempts[:1]}), encoding="utf-8")
    assert pydantic_ai.long_runs(log) == 15  # none passed: the slowest attempt decides


def test_runs_must_be_positive(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pydantic_ai.main(["--long", "--runs", "0"])
    assert "--runs must be at least 1" in capsys.readouterr().err


def test_long_takes_no_other_arms(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pydantic_ai.main(["--long", "--swap"])
    assert "--long runs its own arms" in capsys.readouterr().err


def test_long_without_ollama_is_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pydantic_ai, "ollama_model", lambda model=None: None)
    monkeypatch.setattr(pydantic_ai, "run_long", lambda *a, **k: pytest.fail("ran"))
    assert pydantic_ai.main(["--long"]) == 0
    assert "skipped the model-live runs" in capsys.readouterr().out
