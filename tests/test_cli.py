"""Tests for the ``replay`` command line."""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pytest
from typer.testing import CliRunner

from agent_replay import __version__
from agent_replay.cassette import Cassette
from agent_replay.cli import app

ROOT = Path(__file__).parents[1]
AGENT = "examples.toy_agent.agent:run"
CLIENT = "examples.toy_agent.fake_model:make_client"
TASK = "Order A1001 arrived damaged. Can I get a refund?"
ENV = {"COLUMNS": "200"}

runner = CliRunner()


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def invoke(*args: str):
    return runner.invoke(app, list(args), env=ENV)


def record_args(out: Path, *extra: str) -> list[str]:
    return ["record", AGENT, "--task", TASK, "--output", str(out), "--client", CLIENT, *extra]


@pytest.fixture
def cassette_path(tmp_path: Path) -> Path:
    path = tmp_path / "c.jsonl"
    result = invoke(*record_args(path))
    assert result.exit_code == 0, result.output
    return path


def test_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_record_writes_loadable_cassette(tmp_path: Path) -> None:
    out = tmp_path / "rec.jsonl"
    result = invoke(*record_args(out))
    assert result.exit_code == 0, result.output
    assert "envelope" in result.output
    assert "unpriced)" in result.output  # the fake model has no price: not "$0.0000"
    assert "$0.0000" not in result.output
    cassette = Cassette.load(out)
    assert cassette.header.agent == AGENT
    assert cassette.header.task == TASK
    assert cassette.tool_names()


def test_record_refuses_overwrite_without_force(tmp_path: Path) -> None:
    out = tmp_path / "rec.jsonl"
    assert invoke(*record_args(out)).exit_code == 0
    before = out.read_text()
    result = invoke(*record_args(out))
    assert result.exit_code == 2
    assert "--force" in result.output
    assert out.read_text() == before
    assert invoke(*record_args(out, "--force")).exit_code == 0


def test_record_unimportable_agent(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    result = invoke("record", "no_such_pkg.mod:fn", "-t", "t", "-o", str(out), "--client", CLIENT)
    assert result.exit_code == 2
    assert "error" in result.output
    assert "no_such_pkg" in result.output
    assert not out.exists()


def test_record_malformed_agent_ref(tmp_path: Path) -> None:
    result = invoke(
        "record", "no_colon_here", "-t", "t", "-o", str(tmp_path / "x.jsonl"), "--client", CLIENT
    )
    assert result.exit_code == 2
    assert "module:function" in result.output


def test_record_anthropic_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        anthropic.Anthropic()
    except anthropic.AnthropicError:
        pass
    else:

        def boom(*_: object, **__: object) -> object:
            raise anthropic.AnthropicError("no key")

        monkeypatch.setattr(anthropic, "Anthropic", boom)
    result = invoke(
        "record", AGENT, "-t", TASK, "-o", str(tmp_path / "x.jsonl"), "--client", "anthropic"
    )
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output
    assert not (tmp_path / "x.jsonl").exists()


def test_run_passes_on_fresh_cassette(cassette_path: Path) -> None:
    result = invoke("run", str(cassette_path))
    assert result.exit_code == 0, result.output
    assert "PASS tool_sequence" in result.output


def test_run_mutant_fails(cassette_path: Path) -> None:
    result = invoke(
        "run", str(cassette_path), "--agent", "examples.toy_agent.mutants:first_tool_only"
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_run_json_report(cassette_path: Path) -> None:
    ok = invoke("run", str(cassette_path), "--json")
    assert ok.exit_code == 0
    report = json.loads(ok.output)
    assert report["passed"] is True
    assert report["steps"]
    assert report["checks"][0]["check"] == "tool_sequence"
    assert report["unconsumed"] == []

    bad = invoke(
        "run", str(cassette_path), "--json", "--agent", "examples.toy_agent.mutants:first_tool_only"
    )
    assert bad.exit_code == 1
    bad_report = json.loads(bad.output)
    assert bad_report["passed"] is False
    assert bad_report["checks"][0]["passed"] is False


def test_run_fails_when_the_agent_raises_even_if_checks_pass(cassette_path: Path) -> None:
    agent = "examples.toy_agent.mutants:crash_after_answer"
    result = invoke("run", str(cassette_path), "--agent", agent)
    assert result.exit_code == 1
    assert "FAIL run: agent raised RuntimeError: crashed after answering" in result.output
    assert "PASS tool_sequence" in result.output

    # The warning log goes to stderr; the report alone is on stdout.
    report = json.loads(invoke("run", str(cassette_path), "--json", "--agent", agent).stdout)
    assert report["passed"] is False
    assert report["error"]["type"] == "RuntimeError"
    assert report["checks"][0]["passed"] is True


def test_run_expect_override_mismatch(cassette_path: Path) -> None:
    result = invoke("run", str(cassette_path), "--expect", "lookup_order")
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_run_in_order_subset(cassette_path: Path) -> None:
    result = invoke(
        "run", str(cassette_path), "--expect", "lookup_order", "--sequence-mode", "in_order"
    )
    assert result.exit_code == 0, result.output
    assert "PASS tool_sequence" in result.output


def test_run_missing_cassette(tmp_path: Path) -> None:
    result = invoke("run", str(tmp_path / "missing.jsonl"))
    assert result.exit_code == 2
    assert "replay record" in result.output


def test_run_corrupt_cassette(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("this is not json\n")
    result = invoke("run", str(bad))
    assert result.exit_code == 2
    assert "error" in result.output


def test_show(cassette_path: Path) -> None:
    result = invoke("show", str(cassette_path))
    assert result.exit_code == 0, result.output
    assert AGENT in result.output
    assert TASK in result.output
    assert "lookup_order" in result.output
    assert "anthropic.messages.create" in result.output


def test_show_missing_file(tmp_path: Path) -> None:
    result = invoke("show", str(tmp_path / "nope.jsonl"))
    assert result.exit_code == 2
