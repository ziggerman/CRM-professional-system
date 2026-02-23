"""Add soft delete fields and database indexes.

Revision ID: add_soft_delete_and_indexes
Revises: e60027802bd9_phase_10_12_13_updates_v3
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = 'add_soft_delete_and_indexes'
down_revision: Union[str, None] = '565db3ab855f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ======= Soft Delete Fields =======
    # Add is_deleted boolean field
    op.add_column('leads', sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'))
    op.create_index('idx_leads_is_deleted', 'leads', ['is_deleted'])
    
    # Add deleted_at timestamp
    op.add_column('leads', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))
    
    # Add deleted_by string
    op.add_column('leads', sa.Column('deleted_by', sa.String(128), nullable=True))
    
    # ======= Performance Indexes =======
    # Index for AI score filtering (partial index - only non-null scores)
    op.create_index('idx_leads_ai_score', 'leads', ['ai_score'], postgresql_where=sa.text('ai_score IS NOT NULL'))
    
    # Composite index for stage + created_at (common query pattern)
    op.create_index('idx_leads_stage_created', 'leads', ['stage', sa.text('created_at DESC')])
    
    # Composite index for assignment + stage filtering
    op.create_index('idx_leads_assigned_stage', 'leads', ['assigned_to_id', 'stage'])
    
    # Composite index for source + business_domain filtering
    op.create_index('idx_leads_source_domain', 'leads', ['source', 'business_domain'])
    
    # Sales table indexes
    op.create_index('idx_sales_stage_lead', 'sales', ['stage', 'lead_id'])
    
    # History table indexes
    op.create_index('idx_history_lead_created', 'lead_history', ['lead_id', sa.text('created_at DESC')])
    
    # Notes table indexes  
    op.create_index('idx_notes_lead_created', 'lead_notes', ['lead_id', sa.text('created_at DESC')])


def downgrade() -> None:
    # Remove indexes
    op.drop_index('idx_notes_lead_created', 'lead_notes')
    op.drop_index('idx_history_lead_created', 'lead_history')
    op.drop_index('idx_sales_stage_lead', 'sales')
    op.drop_index('idx_leads_source_domain', 'leads')
    op.drop_index('idx_leads_assigned_stage', 'leads')
    op.drop_index('idx_leads_stage_created', 'leads')
    op.drop_index('idx_leads_ai_score', 'leads')
    op.drop_index('idx_leads_is_deleted', 'leads')
    
    # Remove columns
    op.drop_column('leads', 'deleted_by')
    op.drop_column('leads', 'deleted_at')
    op.drop_column('leads', 'is_deleted')
