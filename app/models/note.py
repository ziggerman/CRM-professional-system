"""
Lead Note model - comments and activity history for leads.
"""
from datetime import datetime, UTC
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Text, ForeignKey, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base

if TYPE_CHECKING:
    from app.models.lead import Lead


class LeadNote(Base):
    """Note/comment attached to a lead."""
    
    __tablename__ = "lead_notes"
    __allow_unmapped__ = True
    
    id: int = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: int = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id: Optional[str] = mapped_column(String(64), nullable=True)
    author_name: Optional[str] = mapped_column(String(128), nullable=True)
    content: str = mapped_column(Text, nullable=False)
    note_type: str = mapped_column(String(32), default="comment")
    created_at: datetime = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: datetime = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC)
    )
    
    # Relationship
    lead: "Lead" = relationship("Lead", back_populates="notes")
    
    def __repr__(self):
        return f"<LeadNote id={self.id} lead_id={self.lead_id} type={self.note_type}>"
