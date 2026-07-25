"""Initial schema — users, projects, rounds_history tables.

Revision ID: 001
Revises:
Create Date: 2026-06-16
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",                sa.String(36),  primary_key=True),
        sa.Column("username",          sa.String(120), nullable=False),
        sa.Column("password_hash",     sa.String(200), nullable=False),
        sa.Column("hospital_name",     sa.String(200), nullable=False),
        sa.Column("contact_email",     sa.String(200), nullable=False),
        sa.Column("approved_projects", sa.JSON(),      nullable=False, server_default="[]"),
        sa.Column("pending_projects",  sa.JSON(),      nullable=False, server_default="[]"),
        sa.Column("role",              sa.String(20),  nullable=False, server_default="client"),
        sa.Column("created_at",        sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("last_active",       sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    # ── projects ──────────────────────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("proj_id",              sa.String(36),   primary_key=True),
        sa.Column("name",                 sa.String(200),  nullable=False),
        sa.Column("description",          sa.Text(),       nullable=False, server_default=""),
        sa.Column("current_round",        sa.Integer(),    nullable=False, server_default="0"),
        sa.Column("round_state",          sa.String(20),   nullable=False, server_default="idle"),
        sa.Column("global_model_path",    sa.String(500),  nullable=False, server_default=""),
        sa.Column("recommended_depth",    sa.Integer(),    nullable=False, server_default="4"),
        sa.Column("accepting_clients",    sa.Boolean(),    nullable=False, server_default="true"),
        sa.Column("fedprox_mu",           sa.Float(),      nullable=False, server_default="0.01"),
        sa.Column("momentum_beta",        sa.Float(),      nullable=False, server_default="0.9"),
        sa.Column("min_clients_per_round", sa.Integer(),   nullable=False, server_default="1"),
        sa.Column("connected_clients",    sa.JSON(),       nullable=False, server_default="[]"),
        sa.Column("pending_clients",      sa.JSON(),       nullable=False, server_default="[]"),
        sa.Column("required_schema",      sa.JSON(),       nullable=True),
        sa.Column("schema_version",       sa.String(20),   nullable=False, server_default="1.0"),
        sa.Column("created_at",           sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at",           sa.DateTime(timezone=True), nullable=True),
    )

    # ── rounds_history ────────────────────────────────────────────────────────
    op.create_table(
        "rounds_history",
        sa.Column("id",                  sa.Integer(),    primary_key=True, autoincrement=True),
        sa.Column("proj_id",             sa.String(36),   nullable=False),
        sa.Column("round",               sa.Integer(),    nullable=False),
        sa.Column("global_val_rmse",     sa.Float(),      nullable=True),
        sa.Column("global_tox_accuracy", sa.Float(),      nullable=True),
        sa.Column("global_auc",          sa.Float(),      nullable=True),
        sa.Column("num_clients",         sa.Integer(),    nullable=True),
        sa.Column("active_depth",        sa.Integer(),    nullable=True),
        sa.Column("extra",               sa.JSON(),       nullable=True),
        sa.Column("timestamp",           sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ix_rounds_history_proj_id", "rounds_history", ["proj_id"])


def downgrade() -> None:
    op.drop_index("ix_rounds_history_proj_id", table_name="rounds_history")
    op.drop_table("rounds_history")
    op.drop_table("projects")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
