"""Add note pinning field and notes indexes.

Revision ID: add_note_pinning_and_index
Revises: add_quality_tier
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "add_note_pinning_and_index"
down_revision: Union[str, None] = "add_quality_tier"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "lead_notes",
        sa.Column("is_pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("idx_lead_notes_is_pinned", "lead_notes", ["is_pinned"])
    op.create_index("idx_lead_notes_note_type", "lead_notes", ["note_type"])


def downgrade() -> None:
    op.drop_index("idx_lead_notes_note_type", table_name="lead_notes")
    op.drop_index("idx_lead_notes_is_pinned", table_name="lead_notes")
    op.drop_column("lead_notes", "is_pinned")
