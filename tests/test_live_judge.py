"""The judge against the real Anthropic API. Deselected by default; run `make test-live`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_replay import Cassette, replay
from agent_replay.judge import AnthropicJudge, JudgeInput, Rubric
from examples.retail_agent import agent as retail_agent
from examples.retail_agent import tools as retail_tools

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"),
]

RETAIL = Path(__file__).parents[1] / "examples" / "retail_agent"


def test_the_opus_judge_grades_a_replayed_retail_run():
    import anthropic  # noqa: PLC0415

    retail_tools.reset_db()
    try:
        run = replay(
            Cassette.load(RETAIL / "cassettes" / "cancel_pending_laptop.jsonl"), retail_agent.run
        )
    finally:
        retail_tools.reset_db()
    assert run.ok, run.error
    rubric = Rubric.load(RETAIL / "rubric.yaml")

    with anthropic.Anthropic() as client:  # closed, so no socket outlives the test
        judge = AnthropicJudge(client, model="claude-opus-5-5", effort="low")
        response = judge.evaluate(rubric, JudgeInput.from_run(run))

    assert [v.id for v in response.verdicts] == list(rubric.ids)
    assert all(v.verdict in ("pass", "fail") and v.reasoning.strip() for v in response.verdicts)
    assert response.usage is not None
    assert response.usage.output_tokens > 0
    assert response.cost_usd is not None
    assert response.cost_usd > 0
    print(f"\njudge {judge.judge_id}: cost ${response.cost_usd:.6f}, {response.latency_ms} ms")
    print(f"usage {response.usage.model_dump()}")
    for v in response.verdicts:
        print(f"{v.id}: {v.verdict} - {v.reasoning}")
