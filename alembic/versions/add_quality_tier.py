"""Add quality_tier column to leads

Revision ID: add_quality_tier
Revises: add_soft_delete_and_indexes
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = 'add_quality_tier'
down_revision: Union[str, None] = 'add_soft_delete_and_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add quality_tier enum column
    op.add_column('leads', sa.Column('quality_tier', sa.String(16), nullable=True))
    # Index for fast filtering by quality tier
    op.create_index('idx_leads_quality_tier', 'leads', ['quality_tier'])


def downgrade() -> None:
    op.drop_index('idx_leads_quality_tier', 'leads')
    op.drop_column('leads', 'quality_tier')
