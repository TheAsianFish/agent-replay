"""Tests for the price table and its ``REPLAY_PRICES_FILE`` override."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.cassette import Usage
from agent_replay.cli import app
from agent_replay.errors import ConfigError
from agent_replay.pricing import PRICES, PRICES_CAPTURED, ModelPrice, cost_usd, prices

USAGE = Usage(
    input_tokens=1000,
    output_tokens=500,
    cache_read_input_tokens=2000,
    cache_creation_input_tokens=100,
)


def write_prices(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPLAY_PRICES_FILE", raising=False)


def test_without_the_variable_the_builtin_table_is_used() -> None:
    assert prices() is PRICES
    assert PRICES_CAPTURED == "2026-10-08"
    assert cost_usd("my-model", USAGE) is None


def test_the_file_prices_a_new_model_and_overrides_a_known_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_prices(
        tmp_path / "prices.json",
        {
            "my-model": {"input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 3.0},
            "claude-opus-5-5": {
                "input": 2.0,
                "output": 10.0,
                "cache_read": 0.1,
                "cache_write": 2.5,
            },
        },
    )
    monkeypatch.setenv("REPLAY_PRICES_FILE", str(path))

    table = prices()
    assert table["my-model"] == ModelPrice(input=1.0, output=2.0, cache_read=0.5, cache_write=3.0)
    assert table["claude-sonnet-5-5"] == PRICES["claude-sonnet-5-5"]  # kept
    assert cost_usd("my-model", USAGE) == pytest.approx(
        (1000 * 1.0 + 500 * 2.0 + 2000 * 0.5 + 100 * 3.0) / 1e6
    )
    assert cost_usd("claude-opus-5-5", USAGE) == pytest.approx(
        (1000 * 2.0 + 500 * 10.0 + 2000 * 0.1 + 100 * 2.5) / 1e6
    )
    assert PRICES["claude-opus-5-5"].input == 4.0  # the built-in table is not mutated


def test_an_edited_file_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_prices(
        tmp_path / "prices.json",
        {"m": {"input": 1.0, "output": 1.0, "cache_read": 1.0, "cache_write": 1.0}},
    )
    monkeypatch.setenv("REPLAY_PRICES_FILE", str(path))
    assert prices()["m"].input == 1.0
    write_prices(path, {"m": {"input": 7.5, "output": 1.0, "cache_read": 1.0, "cache_write": 1.0}})
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert prices()["m"].input == 7.5


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "is invalid"),
        ('{"m": {"input": 1, "output": 1, "cache_read": 1}}', r"m\.cache_write: Field required"),
        (
            '{"m": {"input": -1, "output": 1, "cache_read": 1, "cache_write": 1}}',
            r"m\.input: Input should be greater than or equal to 0",
        ),
        (
            '{"m": {"input": 1, "output": 1, "cache_read": 1, "cache_write": 1, "x": 1}}',
            r"m\.x: Extra inputs",
        ),
    ],
)
def test_an_invalid_file_is_a_config_error_naming_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str, message: str
) -> None:
    path = tmp_path / "prices.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("REPLAY_PRICES_FILE", str(path))
    with pytest.raises(ConfigError, match="REPLAY_PRICES_FILE") as info:
        prices()
    assert info.match(message)
    with pytest.raises(ConfigError, match="REPLAY_PRICES_FILE"):
        cost_usd("claude-opus-5-5", USAGE)


def test_a_missing_file_is_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_PRICES_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(ConfigError, match=r"REPLAY_PRICES_FILE=.*unset it"):
        prices()


def test_the_cli_rejects_a_bad_prices_file_before_doing_anything(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text("{", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["show", "examples/toy_agent/cassettes/refund_eligible.jsonl"],
        env={"REPLAY_PRICES_FILE": str(path), "COLUMNS": "200"},
    )
    assert result.exit_code == 2
    assert "REPLAY_PRICES_FILE" in result.output
