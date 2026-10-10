"""The run store as it was before migrations: suites, runs, steps and check results.

Stores created by ``create_all`` before Alembic arrived have these tables but no
``alembic_version``; :func:`agent_replay.store.migrate` brings their columns up
to this revision and stamps it instead of running it.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "suites",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config_path", sa.String(), nullable=False),
        sa.Column("git_sha", sa.String(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("scenario_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("suite_id", sa.String(32), sa.ForeignKey("suites.id"), nullable=True),
        sa.Column("scenario", sa.String(), nullable=False),
        sa.Column("cassette_path", sa.String(), nullable=False),
        sa.Column("cassette_id", sa.String(), nullable=True),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("policy", sa.String(), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("output", sa.JSON(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("approximate_count", sa.Integer(), nullable=False),
        sa.Column("unconsumed_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
    )
    op.create_index("ix_runs_suite_id", "runs", ["suite_id"])
    op.create_index("ix_runs_scenario", "runs", ["scenario"])
    op.create_index("ix_runs_started_at", "runs", ["started_at"])
    op.create_table(
        "steps",
        sa.Column(
            "run_id",
            sa.String(32),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("index", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cache_creation_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("span_id", sa.String(), nullable=True),
        sa.Column("parent_span_id", sa.String(), nullable=True),
        sa.Column("envelope_seq", sa.Integer(), nullable=True),
    )
    op.create_table(
        "check_results",
        sa.Column(
            "run_id",
            sa.String(32),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("check", sa.String(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("advisory", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_table("check_results")
    op.drop_table("steps")
    op.drop_table("runs")
    op.drop_table("suites")
