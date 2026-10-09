"""Stubs for the synth policy: every stub must validate back into its type."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agent_replay.synth import SynthError, stub_from_schema


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Line(BaseModel):
    sku: str
    quantity: Annotated[int, Field(ge=1)]


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(alias="orderId")
    status: Literal["pending", "delivered"]
    color: Color
    placed_at: datetime
    due: date
    ref: UUID
    total: Decimal
    lines: list[Line] = Field(min_length=1)
    note: str | None
    tags: dict[str, int]
    ratio: Annotated[float, Field(gt=0, lt=0.5)]
    score: Annotated[int, Field(gt=-10, lt=-2)]
    pair: tuple[int, str]


VALID = [
    str,
    int,
    float,
    bool,
    type(None),
    datetime,
    date,
    UUID,
    Decimal,
    Color,
    Literal["a", "b"],
    list[int],
    dict[str, Line],
    Order,
    int | str,
    Annotated[str, Field(min_length=3)],
    Annotated[float, Field(ge=2.5)],
    Annotated[int, Field(le=-4)],
]


@pytest.mark.parametrize("hint", VALID, ids=repr)
def test_stub_validates_into_the_type(hint: Any) -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(hint)
    stub = stub_from_schema(adapter.json_schema(mode="validation"))
    adapter.validate_python(stub)


def test_order_stub_is_the_smallest_valid_order() -> None:
    stub = stub_from_schema(TypeAdapter(Order).json_schema(mode="validation"))
    assert isinstance(stub, dict)
    assert stub["orderId"] == ""
    assert stub["status"] == "pending"
    assert stub["lines"] == [{"sku": "", "quantity": 1}]
    assert stub["note"] is None
    assert 0 < stub["ratio"] < 0.5
    assert stub["score"] == -3


@pytest.mark.parametrize(
    "hint",
    [
        Annotated[str, Field(pattern=r"^[A-Z]{3}$")],
        Annotated[int, Field(gt=0, lt=1)],
        Annotated[set[int], Field(min_length=2)],
    ],
    ids=repr,
)
def test_types_without_a_safe_stub_are_refused(hint: Any) -> None:
    with pytest.raises(SynthError):
        stub_from_schema(TypeAdapter(hint).json_schema(mode="validation"))


def test_recursive_type_without_a_base_case_is_refused() -> None:
    class Node(BaseModel):
        child: Node

    Node.model_rebuild()
    with pytest.raises(SynthError, match="recursive"):
        stub_from_schema(TypeAdapter(Node).json_schema())
