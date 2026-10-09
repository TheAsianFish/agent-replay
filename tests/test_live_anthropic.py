"""Live checks against the real Anthropic API. Deselected by default; run `make test-live`."""

from __future__ import annotations

import os

import pytest

from agent_replay import Cassette, Kind, StepStatus
from agent_replay.checks import tool_sequence
from agent_replay.runner import record, replay
from examples.toy_agent import agent as toy

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"),
]


def test_real_recording_replays_offline_exactly():
    import anthropic  # noqa: PLC0415

    result, cassette = record(
        toy.run, "Order A1001 arrived damaged. Can I get a refund?", client=anthropic.Anthropic()
    )

    assert result.ok, result.error
    model_calls = [e for e in cassette.envelopes if e.kind is Kind.MODEL]
    assert model_calls
    assert all(e.usage is not None and e.usage.input_tokens > 0 for e in model_calls)
    assert all(e.cost_usd is not None and e.cost_usd > 0 for e in model_calls)
    # Whatever the model chose, it must have looked the order up before refunding.
    assert result.check(tool_sequence(["lookup_order", "issue_refund"], mode="in_order"))[0].passed

    replayed = replay(Cassette.loads(cassette.dumps()), toy.run)
    assert replayed.ok, replayed.error
    assert replayed.output == result.output
    assert {s.status for s in replayed.trajectory.steps} == {StepStatus.EXACT}
