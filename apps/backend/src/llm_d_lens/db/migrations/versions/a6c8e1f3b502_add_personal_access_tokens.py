"""Add independent, hashed personal access tokens.

Revision ID: a6c8e1f3b502
Revises: e6b8d0f2a4c7
"""

import sqlalchemy as sa
from alembic import op

revision = "a6c8e1f3b502"
down_revision = "e6b8d0f2a4c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "personal_access_tokens",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("token_prefix", sa.String(20), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_personal_access_tokens_hash", "personal_access_tokens", ["token_hash"], unique=True)
    op.create_index("ix_personal_access_tokens_user", "personal_access_tokens", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_personal_access_tokens_user", table_name="personal_access_tokens")
    op.drop_index("uq_personal_access_tokens_hash", table_name="personal_access_tokens")
    op.drop_table("personal_access_tokens")
