"""Partial cassettes: a recording that writes as it goes, and survives being killed (M14)."""

from __future__ import annotations

import json
import lzma
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest
import structlog
from anthropic import Anthropic
from typer.testing import CliRunner

from agent_replay import cassette as cassette_module
from agent_replay import wrap
from agent_replay.cassette import (
    INTERRUPTED_ERROR,
    Cassette,
    CassetteWriter,
    Envelope,
    is_cassette_path,
    partial_path,
)
from agent_replay.ci import Overrides, run_suite
from agent_replay.cli import app
from agent_replay.diff import diff_steps, steps_from_cassette
from agent_replay.errors import CassetteError, ReplayError, UnrecordedCallError
from agent_replay.fake import FakeAnthropic, FakeRequest, Reply, ToolCall
from agent_replay.runner import record, replay
from agent_replay.session import RecordingSession
from agent_replay.trajectory import StepStatus
from eval import bench_scale as bs

ROOT = Path(__file__).resolve().parents[1]
TURNS = 8
"""8 model calls and 7 tool calls: 15 envelopes."""
AGENT = "eval.bench_scale:run"
SUFFIXES = (".jsonl", ".jsonl.xz")


def _policy(kill_before_model_call: int | None = None) -> Any:
    def policy(request: FakeRequest) -> Reply:
        done = len(request.tool_results())
        if done == kill_before_model_call:
            os.kill(os.getpid(), signal.SIGKILL)
        text = bs.words(f"reply-{done}", bs.REPLY_CHARS)
        if done + 1 >= TURNS:
            return Reply(text=text)
        return Reply(text=text, tool_calls=[ToolCall("read_record", {"record_id": done})])

    return policy


def make_client() -> Anthropic:
    return FakeAnthropic(_policy()).as_anthropic()


def _record(out: Path | None, **kwargs: Any) -> Cassette:
    result, cassette = record(
        bs.run, bs.TASK, client=make_client(), agent_ref=AGENT, out=out, **kwargs
    )
    assert result.error is None
    return cassette


def _comparable(envelope: Envelope) -> dict[str, Any]:
    """An envelope without what differs between two recordings (timing)."""
    return envelope.model_dump(exclude={"latency_ms"})


# --- the writer ----------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_finished_file_is_byte_identical_to_dump(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"run{suffix}"
    cassette = _record(out)
    dumped = tmp_path / f"dumped{suffix}"
    cassette.dump(dumped)
    assert out.read_bytes() == dumped.read_bytes()
    assert not partial_path(out).exists()
    assert Cassette.load(out) == cassette


def test_an_xz_partial_is_plain_jsonl_while_the_run_goes(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl.xz"
    seen: list[bytes] = []

    def policy(request: FakeRequest) -> Reply:
        seen.append(partial_path(out).read_bytes())
        return _policy()(request)

    with RecordingSession(agent=AGENT, task=bs.TASK, out=out) as session:
        output = bs.run(cast(Anthropic, wrap(FakeAnthropic(policy))), bs.TASK)
    session.finish(output=output)
    # Before model call m the partial holds the header and 2m envelopes, complete lines.
    assert [blob.count(b"\n") for blob in seen] == [1 + 2 * m for m in range(TURNS)]
    assert all(blob.endswith(b"\n") for blob in seen)
    assert lzma.decompress(out.read_bytes()).startswith(seen[-1])


def test_the_streamed_xz_bytes_do_not_depend_on_chunking() -> None:
    text = bs.record_run(20).dumps().encode()
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, preset=cassette_module.XZ_PRESET)
    streamed = b"".join(compressor.compress(line + b"\n") for line in text.split(b"\n")[:-1])
    streamed += compressor.flush()
    assert streamed == lzma.compress(text, format=lzma.FORMAT_XZ, preset=cassette_module.XZ_PRESET)


def test_an_earlier_partial_is_kept_unless_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    partial_path(out).write_text("kept\n")
    with pytest.raises(CassetteError, match=r"interrupted earlier recording.*--force"):
        _record(out)
    assert partial_path(out).read_text() == "kept\n"
    _record(out, overwrite=True)
    assert out.exists()
    assert not partial_path(out).exists()


def test_the_writer_refuses_other_compressions_before_writing(tmp_path: Path) -> None:
    with pytest.raises(CassetteError, match=r"\.gz is not a supported compression"):
        RecordingSession(out=tmp_path / "run.jsonl.gz")
    assert list(tmp_path.iterdir()) == []


def test_the_writer_checks_the_footer_count(tmp_path: Path) -> None:
    cassette = bs.record_run(2)
    writer = CassetteWriter(tmp_path / "x.jsonl", cassette.header)
    writer.append(cassette.envelopes[0])
    with pytest.raises(CassetteError, match="out of order"):
        writer.append(cassette.envelopes[2])
    with pytest.raises(CassetteError, match="footer counts 3 envelopes, 1 were written"):
        writer.finish(cassette.footer)
    writer.close()


def test_a_base_exception_in_the_agent_leaves_an_interrupted_partial(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"

    def agent(client: Anthropic, task: str) -> str:
        client.messages.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": task}]
        )
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        record(agent, "t", client=make_client(), out=out)
    assert not out.exists()
    cassette = Cassette.load(partial_path(out))
    assert cassette.interrupted
    assert len(cassette.envelopes) == 1


def test_an_exception_out_of_the_block_can_still_be_finished(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    session = RecordingSession(task="t", out=out)
    try:
        with session:
            wrap(make_client()).messages.create(
                model="m", max_tokens=1, messages=[{"role": "user", "content": "t"}]
            )
            raise ValueError("boom")
    except ValueError as exc:
        cassette = session.finish(error=exc)
    assert cassette.footer.status == "error"
    assert Cassette.load(out) == cassette
    assert not partial_path(out).exists()


# --- loading a partial ---------------------------------------------------------------------


def _partial(tmp_path: Path, keep: int, *, torn: bytes = b"") -> tuple[Path, Cassette]:
    """A ``.partial`` with the header and the first ``keep`` envelopes of a finished run."""
    finished = bs.record_run(TURNS)
    lines = finished.dumps().encode().split(b"\n")[: 1 + keep]
    path = tmp_path / "run.jsonl.partial"
    path.write_bytes(b"\n".join(lines) + b"\n" + torn)
    return path, finished


def test_a_footer_less_partial_loads_as_interrupted(tmp_path: Path) -> None:
    path, finished = _partial(tmp_path, 5)
    cassette = Cassette.load(path)
    assert cassette.interrupted
    assert cassette.envelopes == finished.envelopes[:5]
    assert cassette.header == finished.header
    assert cassette.footer.status == "error"
    assert cassette.footer.error is not None
    assert cassette.footer.error.type == INTERRUPTED_ERROR
    assert cassette.footer.envelope_count == 5
    assert "interrupted recording: 5 envelope(s)" in cassette.describe_interrupted(path)


def test_a_torn_last_line_is_dropped_and_counted(tmp_path: Path) -> None:
    torn = '{"type":"envelope","seq":3,"request":"café'.encode()[:-1]  # inside the é
    path, finished = _partial(tmp_path, 3, torn=torn)
    cassette = Cassette.load(path)
    assert cassette.envelopes == finished.envelopes[:3]
    assert cassette.torn_bytes == len(torn)
    assert f"torn last line of {len(torn)} bytes" in cassette.describe_interrupted(path)


def test_a_partial_with_only_its_header_has_no_envelopes(tmp_path: Path) -> None:
    path, _ = _partial(tmp_path, 0)
    cassette = Cassette.load(path)
    assert cassette.interrupted
    assert cassette.envelopes == ()


def test_an_empty_partial_names_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl.partial"
    path.write_bytes(b'{"type":"hea')
    with pytest.raises(CassetteError, match=r"stopped before its header.*replay record"):
        Cassette.load(path)


def test_a_partial_that_got_its_footer_is_a_finished_cassette(tmp_path: Path) -> None:
    """Killed between the footer and the rename: nothing is missing."""
    finished = bs.record_run(TURNS)
    path = tmp_path / "run.jsonl.partial"
    path.write_text(finished.dumps())
    cassette = Cassette.load(path)
    assert not cassette.interrupted
    assert cassette == finished


def test_a_footer_less_finished_name_is_still_an_error(tmp_path: Path) -> None:
    path, _ = _partial(tmp_path, 5)
    renamed = path.rename(tmp_path / "run.jsonl")
    with pytest.raises(CassetteError, match="cut short"):
        Cassette.load(renamed)


def test_an_xz_partial_is_refused_with_the_fix(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl.xz.partial"
    path.write_bytes(lzma.compress(b"{}\n"))
    with pytest.raises(CassetteError, match="partial cassette is plain JSONL"):
        Cassette.load(path)


def test_an_interrupted_cassette_is_written_only_as_a_partial(tmp_path: Path) -> None:
    path, _ = _partial(tmp_path, 4)
    cassette = Cassette.load(path)
    assert cassette.dumps() == path.read_text()
    with pytest.raises(CassetteError, match=r"no footer; write it to x\.jsonl\.partial"):
        cassette.dump(tmp_path / "x.jsonl")
    cassette.dump(tmp_path / "x.jsonl.partial")
    assert Cassette.load(tmp_path / "x.jsonl.partial") == cassette


def test_partials_are_named_like_cassettes() -> None:
    assert is_cassette_path("a/run.jsonl.partial")
    assert is_cassette_path("run.jsonl.xz.partial")
    assert not is_cassette_path("notes.txt.partial")


# --- replaying, diffing and refusing a partial ---------------------------------------------


def test_replay_serves_the_recorded_envelopes_then_fails(tmp_path: Path) -> None:
    path, _ = _partial(tmp_path, 6)
    result = replay(Cassette.load(path), bs.run)
    statuses = [s.status for s in result.trajectory.steps]
    assert statuses[:6] == [StepStatus.EXACT] * 6
    assert statuses[6] is StepStatus.UNRECORDED
    assert result.error is not None
    assert result.error.type == UnrecordedCallError.__name__


def _suite(tmp_path: Path, cassette: str) -> Path:
    suite = tmp_path / "replay.toml"
    suite.write_text(
        textwrap.dedent(
            f"""
            [defaults]
            agent = "{AGENT}"

            [[scenario]]
            name = "long"
            cassette = "{cassette}"
            """
        )
    )
    return suite


def test_replay_ci_refuses_an_interrupted_cassette(tmp_path: Path) -> None:
    _partial(tmp_path, 6)
    with structlog.testing.capture_logs():
        report = run_suite(_suite(tmp_path, "run.jsonl.partial"), overrides=Overrides())
    assert not report.passed
    scenario = report.scenarios[0]
    assert not scenario.ran
    assert scenario.error is not None
    assert "interrupted recording: 6 envelope(s)" in scenario.error.message
    assert "replay record --out run.jsonl --force" in scenario.error.message


def test_replay_ci_accepts_a_partial_that_finished(tmp_path: Path) -> None:
    (tmp_path / "run.jsonl.partial").write_text(bs.record_run(TURNS).dumps())
    with structlog.testing.capture_logs():
        report = run_suite(_suite(tmp_path, "run.jsonl.partial"), overrides=Overrides())
    assert report.passed


def test_replay_run_warns_on_an_interrupted_cassette(tmp_path: Path) -> None:
    path, _ = _partial(tmp_path, 6)
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code != 0
    assert "interrupted recording: 6 envelope(s)" in result.output


def test_replay_record_keeps_the_partial_on_ctrl_c(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "record",
            "tests.test_partial_cassettes:interrupted_agent",
            "-t",
            "t",
            "-o",
            str(out),
            "--client",
            "tests.test_partial_cassettes:make_client",
        ],
    )
    assert result.exit_code == 1
    assert "interrupted" in result.output
    assert "replay run" in result.output
    assert Cassette.load(partial_path(out)).interrupted


def interrupted_agent(client: Anthropic, task: str) -> str:
    client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": task}])
    raise KeyboardInterrupt


def test_replay_record_refuses_an_earlier_partial(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    partial_path(out).write_text("kept\n")
    args = [
        "record",
        AGENT,
        "-t",
        bs.TASK,
        "-o",
        str(out),
        "--client",
        "tests.test_partial_cassettes:make_client",
    ]
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0
    assert "interrupted earlier recording" in refused.output
    assert CliRunner().invoke(app, [*args, "--force"]).exit_code == 0
    assert not partial_path(out).exists()


# --- SIGKILL in a subprocess ---------------------------------------------------------------

KILL_SCRIPT = """
import sys
from tests.test_partial_cassettes import AGENT, _policy
from agent_replay.fake import FakeAnthropic
from agent_replay.runner import record
from eval import bench_scale as bs

out, kill_before = sys.argv[1], int(sys.argv[2])
record(bs.run, bs.TASK, client=FakeAnthropic(_policy(kill_before)), agent_ref=AGENT, out=out)
raise SystemExit("not killed")
"""


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("kill_before", [0, 3, TURNS - 1])
def test_a_recording_killed_after_envelope_k_replays_those_k(
    tmp_path: Path, suffix: str, kill_before: int
) -> None:
    """SIGKILL inside model call m: the 2m envelopes before it replay in ``full`` and diff."""
    out = tmp_path / f"killed{suffix}"
    proc = subprocess.run(
        [sys.executable, "-c", KILL_SCRIPT, str(out), str(kill_before)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "REPLAY_LOG_LEVEL": "ERROR"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    assert not out.exists()
    k = 2 * kill_before
    partial = Cassette.load(partial_path(out))
    assert partial.interrupted
    assert partial.torn_bytes == 0
    assert len(partial.envelopes) == k

    finished = _record(tmp_path / f"finished{suffix}")
    assert [_comparable(e) for e in partial.envelopes] == [
        _comparable(e) for e in finished.envelopes[:k]
    ]

    result = replay(partial, bs.run)
    statuses = [s.status for s in result.trajectory.steps]
    assert statuses == [StepStatus.EXACT] * k + [StepStatus.UNRECORDED]
    assert result.error is not None

    diff = diff_steps(steps_from_cassette(partial), steps_from_cassette(finished))
    assert diff.counts == {
        "same": k,
        "changed": 0,
        "added": len(finished.envelopes) - k,
        "removed": 0,
    }
    assert [r.op for r in diff.rows[:k]] == ["same"] * k


def test_the_partial_is_valid_json_lines_at_every_step(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    snapshots: list[str] = []

    def policy(request: FakeRequest) -> Reply:
        snapshots.append(partial_path(out).read_text())
        return _policy()(request)

    with RecordingSession(agent=AGENT, task=bs.TASK, out=out) as session:
        output = bs.run(cast(Anthropic, wrap(FakeAnthropic(policy))), bs.TASK)
    session.finish(output=output)
    for snapshot in snapshots:
        lines = snapshot.split("\n")
        assert lines[-1] == ""
        types = [json.loads(x)["type"] for x in lines[:-1]]
        assert types[0] == "header"
        assert set(types[1:]) <= {"envelope"}


# --- the partial failing to write ----------------------------------------------------------


def test_a_failed_partial_write_never_reaches_the_agent_or_loses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run.jsonl"
    real_append = CassetteWriter.append
    writes = {"n": 0}

    def disk_full_on_the_third(self: CassetteWriter, envelope: Envelope) -> None:
        writes["n"] += 1
        if writes["n"] == 3:
            raise OSError(28, "No space left on device")
        real_append(self, envelope)

    monkeypatch.setattr(CassetteWriter, "append", disk_full_on_the_third)
    cassette = _record(out)
    assert len(cassette.envelopes) == 2 * TURNS - 1
    assert Cassette.load(out) == cassette  # written whole by dump instead
    assert not partial_path(out).exists()
    assert writes["n"] == 3  # nothing more was sent to the broken writer


def test_a_session_that_cannot_start_leaves_no_partial(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    with RecordingSession(task="outer"):
        inner = RecordingSession(task="inner", out=out)
        with pytest.raises(ReplayError, match="already active"):
            inner.__enter__()
    assert not partial_path(out).exists()
    _record(out)  # and the next recording is not refused
