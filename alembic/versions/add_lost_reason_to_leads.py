"""Add lost_reason column to leads.

Revision ID: add_lost_reason_to_leads
Revises: add_note_pinning_and_index
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = "add_lost_reason_to_leads"
down_revision: Union[str, None] = "add_note_pinning_and_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("leads", sa.Column("lost_reason", sa.String(length=32), nullable=True))
    op.create_index("idx_leads_lost_reason", "leads", ["lost_reason"])


def downgrade() -> None:
    op.drop_index("idx_leads_lost_reason", table_name="leads")
    op.drop_column("leads", "lost_reason")
