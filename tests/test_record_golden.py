"""eval/record_golden.py with the fake model: D1's harness runs offline end to end."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_replay.calibrate.golden import OVERALL, GoldenSet
from eval import record_golden

FAKE = "examples.retail_agent.fake_model:make_client"


class _Unauthorised:
    """A model client whose every call fails, as the real one does without credentials."""

    class messages:  # mirrors the SDK's attribute
        @staticmethod
        def create(**_kwargs: Any) -> Any:
            raise PermissionError("synthetic: could not resolve authentication method")


def make_unauthorised_client() -> _Unauthorised:
    return _Unauthorised()


def test_records_builds_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The fake model plays the live one. On ``modify_address_suite`` it calls a tool the
    Opus recording never served, so that run ends in an error: it is not kept, the harness
    exits 1, and the next invocation has that one run left to record."""
    out = tmp_path / "d1"

    assert record_golden.main(["--runs", "1", "--agent", "opus", "--out", str(out),
                               "--client", FAKE]) == 1  # fmt: skip

    golden = GoldenSet.load(out)
    recorded = sorted(p.name for p in (out / "cassettes").glob("*__opus__00.jsonl"))
    assert len(recorded) == 5
    assert "modify_address_suite__opus__00.jsonl" not in recorded
    assert len(golden.items) == 5
    assert {lb.criterion for lb in golden.labels} == {OVERALL}
    assert all(lb.source == "objective" for lb in golden.labels)
    assert all(lb.note is None or "agent error" not in lb.note for lb in golden.labels)
    assert {s.variant for i in golden.items for s in i.sources} == {"opus"}
    labels = (out / "labels.jsonl").read_bytes()
    captured = capsys.readouterr()
    assert "6 of 6 runs to record" in captured.out
    assert "modify_address_suite__opus__00.jsonl: agent error UnrecordedCallError" in captured.err

    assert record_golden.main(["--runs", "1", "--agent", "opus", "--out", str(out),
                               "--client", FAKE]) == 1  # fmt: skip
    assert "1 of 6 runs to record" in capsys.readouterr().out
    assert (out / "labels.jsonl").read_bytes() == labels


def test_the_dry_run_estimates_from_measured_costs(capsys: pytest.CaptureFixture[str]):
    assert record_golden.main(["--dry-run", "--runs", "10"]) == 0
    assert "120 of 120 runs to record; estimated cost $2.19" in capsys.readouterr().out


def test_without_a_key_nothing_is_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "d1"

    code = record_golden.main(["--runs", "1", "--agent", "haiku", "--out", str(out)])

    assert code == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert not out.exists()


def test_runs_that_end_in_an_agent_error_are_not_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "d1"
    client = "tests.test_record_golden:make_unauthorised_client"

    code = record_golden.main(["--runs", "1", "--agent", "haiku", "--out", str(out),
                               "--client", client])  # fmt: skip

    err = capsys.readouterr().err
    assert code == 1
    assert "cassette discarded" in err
    assert "the first 3 runs all failed" in err
    assert list((out / "cassettes").glob("*.jsonl")) == []  # nothing to "resume" over
    assert not (out / "items.jsonl").exists()
    # The next invocation has everything still to do.
    assert record_golden.main(["--runs", "1", "--agent", "haiku", "--out", str(out),
                               "--dry-run"]) == 0  # fmt: skip
    assert "6 of 6 runs to record" in capsys.readouterr().out
