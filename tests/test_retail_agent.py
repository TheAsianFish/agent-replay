"""The retail example: τ-bench tools, tasks and the expected-database reward."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from examples.retail_agent import tools
from examples.retail_agent.tasks import TASKS, expected_db_hash


@pytest.fixture(autouse=True)
def fresh_db() -> Iterator[None]:
    tools.reset_db()
    yield
    tools.reset_db()


def test_six_tasks_cover_every_kind_of_request():
    kinds = {a.name for t in TASKS.values() for a in t.expected_actions}
    assert kinds == tools.WRITE_TOOLS | tools.TERMINAL_TOOLS
    assert [t.id for t in TASKS.values() if not t.expected_actions] == ["lookup_latest_order_total"]
    assert {t.tau_bench_index for t in TASKS.values()} == {10, 17, 27, 53, 67, 69}


def test_expected_db_hash_differs_per_write_and_resets_first():
    first = expected_db_hash(TASKS["cancel_pending_laptop"])
    again = expected_db_hash(TASKS["cancel_pending_laptop"])
    other = expected_db_hash(TASKS["return_damaged_bicycle"])
    assert first == again != other


def test_cancel_refunds_a_gift_card_immediately():
    before = tools.get_user_details("emma_smith_8564")["payment_methods"]["gift_card_8541487"]
    order = tools.cancel_pending_order("#W2417020", "no longer needed")
    after = tools.get_user_details("emma_smith_8564")["payment_methods"]["gift_card_8541487"]
    assert order["status"] == "cancelled"
    assert after["balance"] == round(before["balance"] + 2674.4, 2)


def test_cancelling_a_non_pending_order_fails_like_tau_bench():
    with pytest.raises(ValueError, match=r"^non-pending order cannot be cancelled$"):
        tools.cancel_pending_order("#W3916020", "no longer needed")


def test_invalid_cancel_reason_is_rejected():
    with pytest.raises(ValueError, match=r"^invalid reason$"):
        tools.cancel_pending_order("#W2417020", "found it cheaper")  # type: ignore[arg-type]


def test_unknown_email_is_user_not_found():
    with pytest.raises(ValueError, match=r"^user not found$"):
        tools.find_user_id_by_email("nobody@example.com")


def test_email_lookup_is_case_insensitive():
    assert tools.find_user_id_by_email("MIA.Garcia2723@example.com") == "mia_garcia_4516"


def test_return_to_another_card_violates_policy():
    with pytest.raises(ValueError, match="original payment method or a gift card"):
        tools.return_delivered_order_items("#W5490111", ["4579334072"], "paypal_9497703")


def test_exchange_to_an_unavailable_variant_fails():
    with pytest.raises(ValueError, match=r"^new item 4694984344 not found or available$"):
        tools.exchange_delivered_order_items(
            "#W7181492", ["8118291112"], ["4694984344"], "paypal_3024827"
        )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("652.61 - 642.72", "9.89"), ("(1 + 2) * 3 / 4", "2.25"), ("-2 ** 2", "-4.0")],
)
def test_calculate_matches_tau_bench(expression: str, expected: str):
    assert tools.calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os')", "invalid characters in expression"),
        ("1 / 0", "division by zero"),
        ("9 ** 9 ** 9", "exponent 387420489 is too large"),
    ],
)
def test_calculate_errors(expression: str, message: str):
    with pytest.raises(ValueError, match=message):
        tools.calculate(expression)


def test_reset_db_discards_changes_in_place():
    db = tools.DB
    initial = tools.db_hash()
    tools.cancel_pending_order("#W2417020", "no longer needed")
    assert tools.db_hash() != initial
    tools.reset_db()
    assert tools.DB is db
    assert tools.db_hash() == initial
