"""The judge smoke eval replays offline from the committed verdict cache."""

from __future__ import annotations

import pytest

from eval import judge_smoke


def test_every_run_is_judged_from_the_committed_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_JUDGE_OFFLINE", "1")
    rows = judge_smoke.run()

    assert len(rows) == 24
    assert all(r["judge_answered"] and r["cached"] for r in rows)
    originals = [r for r in rows if r["variant"] == "agent"]
    assert all(r["deterministic_pass"] and r["judge_pass"] for r in originals)
    agree = sum(1 for r in rows if r["judge_pass"] == r["deterministic_pass"])
    assert agree == 23
