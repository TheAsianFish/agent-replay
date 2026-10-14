"""Flake reports (one per ``replay flake``, with each scenario's estimate) and calibration
reports, so the dashboard can chart pass rates over time and show judge agreement.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flake_reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("suite_id", sa.String(32), sa.ForeignKey("suites.id"), nullable=True),
        sa.Column("suite", sa.String(), nullable=False),
        sa.Column("stats_impl", sa.String(), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("git_sha", sa.String(), nullable=True),
        sa.Column("scenario_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("flaky_count", sa.Integer(), nullable=False),
        sa.Column("run_count", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
    )
    op.create_index("ix_flake_reports_suite_id", "flake_reports", ["suite_id"])
    op.create_index("ix_flake_reports_started_at", "flake_reports", ["started_at"])
    op.create_table(
        "flake_scenarios",
        sa.Column(
            "report_id",
            sa.String(32),
            sa.ForeignKey("flake_reports.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("cassette", sa.String(), nullable=False),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("policy", sa.String(), nullable=False),
        sa.Column("classification", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("runs", sa.Integer(), nullable=False),
        sa.Column("passes", sa.Integer(), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("ci_low", sa.Float(), nullable=False),
        sa.Column("ci_high", sa.Float(), nullable=False),
        sa.Column("failures", sa.JSON(), nullable=False),
        sa.Column("executed", sa.Integer(), nullable=False),
        sa.Column("concurrency", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("outcomes", sa.JSON(), nullable=False),
    )
    op.create_index("ix_flake_scenarios_name", "flake_scenarios", ["name"])
    op.create_table(
        "calibration_reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("rubric", sa.String(), nullable=False),
        sa.Column("rubric_fingerprint", sa.String(), nullable=False),
        sa.Column("judge_id", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("report", sa.JSON(), nullable=False),
    )
    op.create_index("ix_calibration_reports_rubric", "calibration_reports", ["rubric"])


def downgrade() -> None:
    op.drop_table("calibration_reports")
    op.drop_table("flake_scenarios")
    op.drop_table("flake_reports")
