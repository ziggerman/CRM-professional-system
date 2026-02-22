"""Add last_lead_assigned_at to users for round-robin assignment.

Revision ID: add_user_last_lead_assigned_at
Revises: add_lost_reason_to_leads
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "add_user_last_lead_assigned_at"
down_revision: Union[str, None] = "add_lost_reason_to_leads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("last_lead_assigned_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("idx_users_last_lead_assigned_at", "users", ["last_lead_assigned_at"])


def downgrade() -> None:
    op.drop_index("idx_users_last_lead_assigned_at", table_name="users")
    op.drop_column("users", "last_lead_assigned_at")
