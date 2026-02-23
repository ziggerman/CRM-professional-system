"""Initial migration - create leads and sales tables

Revision ID: 001_initial
Revises: 
Create Date: 2026-02-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create leads table
    op.create_table(
        'leads',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('telegram_id', sa.String(length=64), nullable=True),
        sa.Column('source', sa.Enum('SCANNER', 'PARTNER', 'MANUAL', name='leadsource'), nullable=False),
        sa.Column('stage', sa.Enum('NEW', 'CONTACTED', 'QUALIFIED', 'TRANSFERRED', 'LOST', name='coldstage'), nullable=False),
        sa.Column('business_domain', sa.Enum('FIRST', 'SECOND', 'THIRD', name='businessdomain'), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False, default=0),
        sa.Column('ai_score', sa.Float(), nullable=True),
        sa.Column('ai_recommendation', sa.String(length=64), nullable=True),
        sa.Column('ai_reason', sa.String(length=512), nullable=True),
        sa.Column('ai_analyzed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_leads_telegram_id'), 'leads', ['telegram_id'], unique=False)
    
    # Create sales table
    op.create_table(
        'sales',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('lead_id', sa.Integer(), nullable=False),
        sa.Column('stage', sa.Enum('NEW', 'KYC', 'AGREEMENT', 'PAID', 'LOST', name='salestage'), nullable=False),
        sa.Column('amount', sa.Integer(), nullable=True),
        sa.Column('notes', sa.String(length=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['lead_id'], ['leads.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('lead_id')
    )


def downgrade() -> None:
    op.drop_table('sales')
    op.drop_index(op.f('ix_leads_telegram_id'), table_name='leads')
    op.drop_table('leads')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS leadsource')
    op.execute('DROP TYPE IF EXISTS coldstage')
    op.execute('DROP TYPE IF EXISTS businessdomain')
    op.execute('DROP TYPE IF EXISTS salestage')
