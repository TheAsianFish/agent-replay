"""eval/record_golden.py with the fake model: D1's harness runs offline end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_replay.calibrate.golden import OVERALL, GoldenSet
from eval import record_golden

FAKE = "examples.retail_agent.fake_model:make_client"


def test_records_builds_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "d1"

    assert record_golden.main(["--runs", "1", "--agent", "opus", "--out", str(out),
                               "--client", FAKE]) == 0  # fmt: skip

    golden = GoldenSet.load(out)
    assert len(list((out / "cassettes").glob("*__opus__00.jsonl"))) == 6
    assert len(golden.items) == 6
    assert {lb.criterion for lb in golden.labels} == {OVERALL}
    assert all(lb.source == "objective" for lb in golden.labels)
    assert {s.variant for i in golden.items for s in i.sources} == {"opus"}
    labels = (out / "labels.jsonl").read_bytes()
    assert "6 of 6 runs to record" in capsys.readouterr().out

    assert record_golden.main(["--runs", "1", "--agent", "opus", "--out", str(out),
                               "--client", FAKE]) == 0  # fmt: skip
    assert "0 of 6 runs to record" in capsys.readouterr().out
    assert (out / "labels.jsonl").read_bytes() == labels


def test_the_dry_run_estimates_from_measured_costs(capsys: pytest.CaptureFixture[str]):
    assert record_golden.main(["--dry-run", "--runs", "10"]) == 0
    assert "120 of 120 runs to record; estimated cost $2.19" in capsys.readouterr().out
