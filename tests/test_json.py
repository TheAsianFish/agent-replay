"""Tests for agent_replay._json."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from anthropic.types import TextBlock, ToolUseBlock
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_replay._json import canonical_json, fingerprint, to_jsonable

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=6), children, max_size=4)
    ),
    max_leaves=15,
)


def shuffled(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = [(k, shuffled(v, rng)) for k, v in value.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(value, list):
        return [shuffled(v, rng) for v in value]
    return value


class TestCanonicalJson:
    def test_sorts_keys_without_whitespace(self) -> None:
        assert canonical_json({"b": 1, "a": [1, 2], "c": {"z": 1, "y": 2}}) == (
            '{"a":[1,2],"b":1,"c":{"y":2,"z":1}}'
        )

    def test_keeps_non_ascii(self) -> None:
        assert canonical_json({"k": "héllo ☃"}) == '{"k":"héllo ☃"}'

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, bad: float) -> None:
        with pytest.raises(TypeError, match="not finite"):
            canonical_json({"x": bad})


class TestToJsonable:
    def test_pydantic_models_drop_none_fields(self) -> None:
        block = TextBlock(type="text", text="hi", citations=None)
        assert to_jsonable(block) == {"type": "text", "text": "hi"}
        tool_use = ToolUseBlock(type="tool_use", id="t1", name="f", input={"a": None})
        # None is kept inside plain dicts that are data, not model fields.
        assert to_jsonable(tool_use) == {
            "type": "tool_use",
            "id": "t1",
            "name": "f",
            "input": {"a": None},
        }

    def test_plain_dict_keeps_none(self) -> None:
        assert to_jsonable({"a": None, "b": [None]}) == {"a": None, "b": [None]}

    def test_tuples_become_lists(self) -> None:
        result = to_jsonable({"t": (1, (2, 3))})
        assert result == {"t": [1, [2, 3]]}
        assert isinstance(result, dict)
        assert isinstance(result["t"], list)


class TestFingerprint:
    def test_prefix(self) -> None:
        fp = fingerprint("tool", "f", {"a": 1})
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 64

    def test_key_order_irrelevant(self) -> None:
        a = fingerprint("tool", "f", {"a": 1, "b": {"x": 1, "y": 2}})
        b = fingerprint("tool", "f", {"b": {"y": 2, "x": 1}, "a": 1})
        assert a == b

    def test_differs_by_kind_name_request(self) -> None:
        base = fingerprint("tool", "f", {"a": 1})
        assert fingerprint("model", "f", {"a": 1}) != base
        assert fingerprint("tool", "g", {"a": 1}) != base
        assert fingerprint("tool", "f", {"a": 2}) != base

    def test_ignore_drops_top_level_fields(self) -> None:
        ignore = frozenset({"metadata"})
        a = fingerprint("model", "m", {"x": 1, "metadata": {"id": "1"}}, ignore=ignore)
        b = fingerprint("model", "m", {"x": 1, "metadata": {"id": "2"}}, ignore=ignore)
        c = fingerprint("model", "m", {"x": 1}, ignore=ignore)
        assert a == b == c

    def test_without_ignore_metadata_matters(self) -> None:
        a = fingerprint("model", "m", {"x": 1, "metadata": {"id": "1"}})
        b = fingerprint("model", "m", {"x": 1, "metadata": {"id": "2"}})
        assert a != b

    def test_ignore_only_top_level(self) -> None:
        ignore = frozenset({"metadata"})
        a = fingerprint("model", "m", {"x": {"metadata": 1}}, ignore=ignore)
        b = fingerprint("model", "m", {"x": {"metadata": 2}}, ignore=ignore)
        assert a != b


@settings(max_examples=100, deadline=None)
@given(value=st.dictionaries(st.text(max_size=6), json_values, max_size=5), seed=st.integers())
def test_fingerprint_invariant_to_key_order(value: dict[str, Any], seed: int) -> None:
    other = shuffled(value, random.Random(seed))
    assert fingerprint("tool", "f", value) == fingerprint("tool", "f", other)
    assert canonical_json(value) == canonical_json(other)


@settings(max_examples=100, deadline=None)
@given(value=json_values)
def test_canonical_json_round_trips(value: Any) -> None:
    assert json.loads(canonical_json(value)) == value
