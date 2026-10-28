"""xz-compressed cassettes (``*.jsonl.xz``), chosen by suffix (PLAN.md, M13)."""

from __future__ import annotations

import lzma
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_replay.cassette import Cassette, is_cassette_path
from agent_replay.ci import run_suite
from agent_replay.cli import app
from agent_replay.errors import CassetteError

ROOT = Path(__file__).parents[1]
TOY = ROOT / "examples" / "toy_agent" / "cassettes" / "refund_eligible.jsonl"
LIVE = ROOT / "examples" / "retail_agent" / "cassettes" / "live" / "cancel_pending_laptop.jsonl"
AGENT = "examples.toy_agent.agent:run"
CLIENT = "examples.toy_agent.fake_model:make_client"
TASK = "Order A1001 arrived damaged. Can I get a refund?"
ENV = {"COLUMNS": "200"}

runner = CliRunner()


def invoke(*args: str):  # type: ignore[no-untyped-def]  # click's Result
    return runner.invoke(app, list(args), env=ENV)


def test_xz_round_trips_and_is_plain_xz_of_the_jsonl(tmp_path: Path) -> None:
    cassette = Cassette.load(LIVE)
    path = tmp_path / "run.jsonl.xz"
    cassette.dump(path)
    assert lzma.decompress(path.read_bytes()).decode("utf-8") == cassette.dumps()
    assert Cassette.load(path) == cassette
    assert path.stat().st_size * 10 < len(cassette.dumps())  # even a 5-turn run shrinks 10x


def test_a_file_compressed_by_another_xz_writer_loads(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl.xz"
    path.write_bytes(lzma.compress(TOY.read_bytes(), preset=9 | lzma.PRESET_EXTREME))
    assert Cassette.load(path) == Cassette.load(TOY)


def test_compressed_output_is_deterministic(tmp_path: Path) -> None:
    cassette = Cassette.load(TOY)
    cassette.dump(tmp_path / "a.jsonl.xz")
    cassette.dump(tmp_path / "b.jsonl.xz")
    assert (tmp_path / "a.jsonl.xz").read_bytes() == (tmp_path / "b.jsonl.xz").read_bytes()


def test_suffix_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "C.JSONL.XZ"
    Cassette.load(TOY).dump(path)
    assert path.read_bytes().startswith(b"\xfd7zXZ\x00")
    assert Cassette.load(path) == Cassette.load(TOY)


def test_an_xz_file_without_the_suffix_names_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_bytes(lzma.compress(TOY.read_bytes()))
    with pytest.raises(CassetteError, match=r"xz-compressed .* rename it to c\.jsonl\.xz"):
        Cassette.load(path)


def test_a_plain_file_named_xz_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl.xz"
    shutil.copy(TOY, path)
    with pytest.raises(CassetteError, match="not a valid xz file"):
        Cassette.load(path)


def test_a_truncated_xz_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl.xz"
    Cassette.load(TOY).dump(path)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(CassetteError, match="not a valid xz file"):
        Cassette.load(path)


@pytest.mark.parametrize("suffix", [".gz", ".bz2", ".zst"])
def test_other_compressions_are_refused_both_ways(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"c.jsonl{suffix}"
    with pytest.raises(CassetteError, match=r"not a supported compression; name it c\.jsonl\.xz"):
        Cassette.load(TOY).dump(path)
    assert not path.exists()
    path.write_bytes(b"anything")
    with pytest.raises(CassetteError, match="not a supported compression"):
        Cassette.load(path)


def test_a_missing_compressed_cassette_says_so(tmp_path: Path) -> None:
    with pytest.raises(CassetteError, match="does not exist"):
        Cassette.load(tmp_path / "nope.jsonl.xz")


def test_cassette_path_names() -> None:
    assert is_cassette_path("a/run.jsonl")
    assert is_cassette_path("run.jsonl.xz")
    assert not is_cassette_path("run.json")
    assert not is_cassette_path("01HXYZ")


def test_record_out_xz_then_run_and_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    out = tmp_path / "rec.jsonl.xz"
    result = invoke("record", AGENT, "--task", TASK, "--out", str(out), "--client", CLIENT)
    assert result.exit_code == 0, result.output
    assert out.read_bytes().startswith(b"\xfd7zXZ\x00")
    assert Cassette.load(out).tool_names()
    assert invoke("run", str(out)).exit_code == 0
    assert "status: ok" in invoke("show", str(out)).output


def test_diff_takes_a_compressed_cassette(tmp_path: Path) -> None:
    packed = tmp_path / "base.jsonl.xz"
    Cassette.load(TOY).dump(packed)
    result = invoke("diff", str(packed), str(TOY))
    assert result.exit_code == 0, result.output
    # named like a cassette, so a missing one is a missing file, not an unknown run id
    db = f"sqlite:///{tmp_path / 'store.db'}"
    missing = invoke("diff", str(tmp_path / "gone.jsonl.xz"), str(TOY), "--db", db)
    assert missing.exit_code == 2
    assert "does not exist" in missing.output
    assert not (tmp_path / "store.db").exists()


def test_replay_ci_reads_a_compressed_cassette(tmp_path: Path) -> None:
    packed = tmp_path / "eligible.jsonl.xz"
    Cassette.load(TOY).dump(packed)
    suite = tmp_path / "replay.toml"
    suite.write_text(
        '[defaults]\nchecks = [{ type = "tool_sequence" }]\n\n'
        '[[scenario]]\nname = "eligible"\ncassette = "eligible.jsonl.xz"\n',
        encoding="utf-8",
    )
    report = run_suite(suite)
    assert [s.passed for s in report.scenarios] == [True]
