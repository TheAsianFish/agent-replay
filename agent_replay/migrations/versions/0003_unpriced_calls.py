"""Unpriced model calls (their model has no price) beside each stored cost, so a run,
flake report or flake scenario whose cost is unknown is not shown as free.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TABLES = ("runs", "flake_reports", "flake_scenarios")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(
            table,
            sa.Column("unpriced_calls", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    for table in reversed(TABLES):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("unpriced_calls")
