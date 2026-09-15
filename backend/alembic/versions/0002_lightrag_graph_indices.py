"""Add versioned LightRAG graph index tracking.

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
    bind = op.get_bind()
    # Revision 0001 historically used current metadata.create_all(). On a
    # brand-new installation it may already have created this table.
    if sa.inspect(bind).has_table("graph_indices"):
        return
    op.create_table(
        "graph_indices",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("book_id", sa.String(length=36), nullable=False),
        sa.Column("index_version", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("track_ids", sa.JSON(), nullable=False),
        sa.Column("document_ids", sa.JSON(), nullable=False),
        sa.Column("file_sources", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_graph_indices_book_id", "graph_indices", ["book_id"])
    op.create_index("ix_graph_indices_index_version", "graph_indices", ["index_version"])
    op.create_index("ix_graph_indices_status", "graph_indices", ["status"])
    op.create_index(
        "ix_graph_indices_book_version",
        "graph_indices",
        ["book_id", "index_version"],
        unique=True,
    )
    op.create_index("ix_graph_indices_book_status", "graph_indices", ["book_id", "status"])


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("graph_indices"):
        op.drop_table("graph_indices")
