"""The garbage collector is paused while a cassette is parsed, and only then (M14)."""

from __future__ import annotations

import gc
from typing import Any, cast

import pytest

from agent_replay import cassette as cassette_module
from agent_replay.cassette import Cassette
from agent_replay.errors import CassetteError
from eval import bench_scale as bs


def _spy_on_loads(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    seen: list[bool] = []
    inner = Cassette._loads.__func__  # type: ignore[attr-defined]

    def spy(cls: type[Cassette], text: str, **kwargs: Any) -> Cassette:
        seen.append(gc.isenabled())
        return cast(Cassette, inner(cls, text, **kwargs))

    monkeypatch.setattr(Cassette, "_loads", classmethod(spy))
    return seen


def test_the_collector_is_paused_only_inside_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _spy_on_loads(monkeypatch)
    assert gc.isenabled()
    Cassette.loads(bs.record_run(3).dumps())
    assert seen == [False]
    assert gc.isenabled()


def test_the_collector_is_restored_when_loading_fails() -> None:
    with pytest.raises(CassetteError):
        Cassette.loads("not json\n{}\n")
    assert gc.isenabled()


def test_a_collector_that_was_off_stays_off() -> None:
    gc.disable()
    try:
        Cassette.loads(bs.record_run(3).dumps())
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_nested_pauses_re_enable_once_at_the_end() -> None:
    with cassette_module._collector_paused():
        with cassette_module._collector_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled()
