"""Add response time and SLA tracking fields.

This migration adds fields for tracking:
- first_response_at: When lead was first contacted
- sla_deadline_at: When SLA expires for the lead
- is_overdue: Whether lead is past SLA
- days_in_stage: How many days lead has been in current stage
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_response_time_and_sla'
down_revision = 'e60027802bd9_phase_10_12_13_updates_v3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add response time tracking fields to leads table
    op.add_column('leads', sa.Column('first_response_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('leads', sa.Column('sla_deadline_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('leads', sa.Column('is_overdue', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('leads', sa.Column('days_in_stage', sa.Integer(), server_default='0', nullable=False))
    
    # Add indexes for performance
    op.create_index('ix_leads_first_response_at', 'leads', ['first_response_at'])
    op.create_index('ix_leads_sla_deadline_at', 'leads', ['sla_deadline_at'])
    op.create_index('ix_leads_is_overdue', 'leads', ['is_overdue'])
    
    # Add activity log table for tracking lead response times
    op.create_table(
        'lead_activities',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('activity_type', sa.String(50), nullable=False),  # 'created', 'contacted', 'qualified', 'note_added', etc.
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_lead_activities_created_at', 'lead_activities', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_lead_activities_created_at', table_name='lead_activities')
    op.drop_table('lead_activities')
    op.drop_index('ix_leads_is_overdue', table_name='leads')
    op.drop_index('ix_leads_sla_deadline_at', table_name='leads')
    op.drop_index('ix_leads_first_response_at', table_name='leads')
    op.drop_column('leads', 'days_in_stage')
    op.drop_column('leads', 'is_overdue')
    op.drop_column('leads', 'sla_deadline_at')
    op.drop_column('leads', 'first_response_at')
