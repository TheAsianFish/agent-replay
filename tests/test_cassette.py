"""Tests for agent_replay.cassette."""

from __future__ import annotations

import itertools
import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_replay.cassette import Cassette
from agent_replay.errors import CassetteError
from agent_replay.record.tools import tool
from agent_replay.session import RecordingSession

CLOCK = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)


def make_session() -> RecordingSession:
    counter = itertools.count()
    return RecordingSession(
        agent="agent:run",
        task="do it",
        metadata={"k": "v"},
        clock=lambda: CLOCK,
        timer=lambda: next(counter) * 0.01,
    )


@tool
def lookup(order_id: str) -> dict[str, Any]:
    """Look up an order."""
    return {"id": order_id, "status": "shipped"}


@tool
def cancel(order_id: str, reason: str = "none") -> str:
    """Cancel an order."""
    return f"cancelled {order_id} ({reason})"


def build() -> Cassette:
    session = make_session()
    with session:
        lookup("A1")
        cancel("A1", reason="late")
        lookup("B2")
    return session.finish(output={"answer": "ünï"})


@pytest.fixture
def cassette() -> Cassette:
    return build()


def lines_of(cassette: Cassette) -> list[str]:
    return cassette.dumps().splitlines()


def test_recorded_shape(cassette: Cassette) -> None:
    assert cassette.header.created_at == CLOCK
    assert cassette.header.agent == "agent:run"
    assert len(cassette.envelopes) == 3
    assert cassette.footer.envelope_count == 3
    assert cassette.footer.status == "ok"
    assert [e.seq for e in cassette.envelopes] == [0, 1, 2]


def test_tool_names(cassette: Cassette) -> None:
    assert cassette.tool_names() == ["lookup", "cancel", "lookup"]


def test_dumps_loads_round_trip(cassette: Cassette) -> None:
    text = cassette.dumps()
    assert text.endswith("\n")
    assert len(text.splitlines()) == 5
    assert Cassette.loads(text) == cassette


def test_dump_load_round_trip_and_atomic(tmp_path: Path, cassette: Cassette) -> None:
    path = tmp_path / "a" / "b" / "run.jsonl"
    cassette.dump(path)
    assert path.exists()
    assert sorted(p.name for p in path.parent.iterdir()) == ["run.jsonl"]
    assert Cassette.load(path) == cassette
    cassette.dump(path)  # overwrite
    assert sorted(p.name for p in path.parent.iterdir()) == ["run.jsonl"]


def test_dump_uses_ordinary_file_permissions(tmp_path: Path, cassette: Cassette) -> None:
    path = tmp_path / "run.jsonl"
    cassette.dump(path)
    plain = tmp_path / "plain.txt"
    plain.write_text("x")  # what a regular open() gives under the current umask
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


def test_load_missing_path(tmp_path: Path) -> None:
    with pytest.raises(CassetteError, match=r"replay record"):
        Cassette.load(tmp_path / "nope.jsonl")


@pytest.mark.parametrize("text", ["", "\n  \n"])
def test_empty_text(text: str) -> None:
    with pytest.raises(CassetteError, match=r"header and a footer"):
        Cassette.loads(text)


def test_single_line_rejected(cassette: Cassette) -> None:
    with pytest.raises(CassetteError, match=r"replay record"):
        Cassette.loads(lines_of(cassette)[0])


def test_future_schema_version_rejected(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    header = json.loads(lines[0])
    header["schema_version"] = 3
    lines[0] = json.dumps(header)
    with pytest.raises(CassetteError, match=r"upgrade") as info:
        Cassette.loads("\n".join(lines), source="x.jsonl")
    assert "x.jsonl" in str(info.value)
    assert "3" in str(info.value)


def test_truncated_file_missing_footer(cassette: Cassette) -> None:
    lines = lines_of(cassette)[:-1]
    with pytest.raises(CassetteError, match=r"footer.*cut short"):
        Cassette.loads("\n".join(lines))


def test_footer_count_mismatch(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    del lines[2]
    with pytest.raises(CassetteError, match=r"footer says 3 envelopes, found 2"):
        Cassette.loads("\n".join(lines))


def test_out_of_order_seq(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    lines[1], lines[2] = lines[2], lines[1]
    with pytest.raises(CassetteError, match=r"seq"):
        Cassette.loads("\n".join(lines))


def test_garbage_json_line(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    lines[2] = "{not json"
    with pytest.raises(CassetteError, match=r":3: invalid envelope"):
        Cassette.loads("\n".join(lines))


def test_garbage_header(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    lines[0] = "garbage"
    with pytest.raises(CassetteError, match=r":1: invalid header"):
        Cassette.loads("\n".join(lines))


def test_unknown_field_in_envelope(cassette: Cassette) -> None:
    lines = lines_of(cassette)
    env = json.loads(lines[1])
    env["surprise"] = 1
    lines[1] = json.dumps(env)
    with pytest.raises(CassetteError, match=r"(?s)invalid envelope.*surprise"):
        Cassette.loads("\n".join(lines))


def test_blank_lines_are_ignored(cassette: Cassette) -> None:
    text = "\n\n".join(lines_of(cassette)) + "\n\n"
    assert Cassette.loads(text) == cassette


@pytest.mark.parametrize("error_case", [True, False])
def test_finish_with_error(error_case: bool) -> None:
    session = make_session()
    with session:
        lookup("A")
    exc = ValueError("boom") if error_case else None
    cassette = session.finish(error=exc)
    assert cassette.footer.status == ("error" if error_case else "ok")
    if error_case:
        assert cassette.footer.error is not None
        assert cassette.footer.error.type == "ValueError"
    assert Cassette.loads(cassette.dumps()) == cassette


# Any text JSON can carry; lone surrogates are not valid UTF-8.
SAFE_CHARS = st.characters(blacklist_categories=("Cs",))


@pytest.mark.parametrize("char", ["\x85", "\u2028", "\u2029"])
def test_unicode_line_separators_survive_round_trip(char: str) -> None:
    @tool
    def echo(text: str) -> str:
        return text

    session = make_session()
    with session:
        echo(f"a{char}b")
    cassette = session.finish()
    assert Cassette.loads(cassette.dumps()) == cassette


json_args = st.dictionaries(
    st.text(alphabet=SAFE_CHARS, min_size=1, max_size=5),
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**31), max_value=2**31)
    | st.text(alphabet=SAFE_CHARS, max_size=10)
    | st.lists(st.integers(min_value=-100, max_value=100), max_size=3),
    max_size=3,
)


@settings(max_examples=50, deadline=None)
@given(calls=st.lists(st.tuples(st.sampled_from(["a", "b", "c"]), json_args), max_size=6))
def test_round_trip_random_cassettes(calls: list[tuple[str, dict[str, Any]]]) -> None:
    @tool
    def echo(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return args

    session = make_session()
    with session:
        for name, args in calls:
            echo(name, args)
    cassette = session.finish(output=calls)
    assert Cassette.loads(cassette.dumps()) == cassette
    assert cassette.tool_names() == ["echo"] * len(calls)


# --- schema versions ----------------------------------------------------------

LIVE_V1 = Path(__file__).parents[1] / "examples/toy_agent/cassettes/live_opus_refund_eligible.jsonl"


def as_v1(cassette: Cassette) -> str:
    lines = lines_of(cassette)
    header = json.loads(lines[0])
    header["schema_version"] = 1
    return "\n".join([json.dumps(header), *lines[1:]])


def test_v1_cassette_is_migrated_to_v2_on_load() -> None:
    raw_header = json.loads(LIVE_V1.read_text(encoding="utf-8").split("\n")[0])
    assert (
        raw_header["schema_version"] == 1
    )  # committed in v1 on purpose: keeps the migration honest

    cassette = Cassette.load(LIVE_V1)

    assert cassette.header.schema_version == 2
    assert cassette.header.cassette_id == raw_header["cassette_id"]
    assert [e.kind.value for e in cassette.envelopes] == [
        "model", "tool", "tool", "model", "tool", "model"
    ]  # fmt: skip


def test_v1_round_trip_keeps_every_field(cassette: Cassette) -> None:
    migrated = Cassette.loads(as_v1(cassette))
    assert migrated == cassette


def test_migrated_cassette_is_written_as_v2(cassette: Cassette) -> None:
    migrated = Cassette.loads(as_v1(cassette))
    assert json.loads(migrated.dumps().split("\n")[0])["schema_version"] == 2


def test_v1_cassette_cannot_contain_v2_kinds(cassette: Cassette) -> None:
    lines = as_v1(cassette).split("\n")
    envelope = json.loads(lines[1])
    envelope["kind"] = "clock"
    lines[1] = json.dumps(envelope)
    with pytest.raises(CassetteError, match=r":2: invalid v1 envelope"):
        Cassette.loads("\n".join(lines))


def test_v1_header_errors_name_the_version(cassette: Cassette) -> None:
    lines = as_v1(cassette).split("\n")
    header = json.loads(lines[0])
    header["surprise"] = True
    lines[0] = json.dumps(header)
    with pytest.raises(CassetteError, match=r":1: invalid v1 header"):
        Cassette.loads("\n".join(lines))


@pytest.mark.parametrize("line", ["[1, 2]", '"text"', "{not json"])
def test_header_must_be_a_json_object(cassette: Cassette, line: str) -> None:
    lines = lines_of(cassette)
    lines[0] = line
    with pytest.raises(CassetteError, match=r":1: invalid header"):
        Cassette.loads("\n".join(lines))


def test_in_start_order_includes_nested_envelopes(cassette: Cassette) -> None:
    assert [e.span_id for e in cassette.in_start_order()] == ["s0", "s1", "s2"]
