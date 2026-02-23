"""
Lead Activity model for tracking lead lifecycle events and response times.
"""
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base


class LeadActivity(Base):
    """Activity log for tracking lead response times and lifecycle events."""
    
    __tablename__ = "lead_activities"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    lead_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("leads.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    
    # Activity type: 'created', 'contacted', 'qualified', 'note_added', 'stage_changed', 'assigned', etc.
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Optional description of the activity
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Who performed the activity (user ID or 'system')
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Metadata JSON for additional data (e.g., old_stage, new_stage)
    metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(UTC)
    )
    
    # Relationship to lead
    lead = relationship("Lead", backref="activities")
    
    # Index for efficient queries
    __table_args__ = (
        Index('ix_lead_activities_lead_created', 'lead_id', 'created_at'),
        Index('ix_lead_activities_type_created', 'activity_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<LeadActivity id={self.id} lead_id={self.lead_id} type={self.activity_type}>"
