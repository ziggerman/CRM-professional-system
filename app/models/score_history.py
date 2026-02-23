"""
Lead Score History Model - tracks all AI analysis results over time.
"""
from datetime import datetime, UTC

from sqlalchemy import Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base


class LeadScoreHistory(Base):
    """
    Stores history of AI score analysis for each lead.
    Enables trend analysis and audit trail.
    """
    __tablename__ = "lead_score_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("leads.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # AI Analysis Results
    score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Analysis Metadata
    analyzed_by: Mapped[str] = mapped_column(String(64), nullable=False, default="openai")  # "openai" or "fallback"
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True
    )
    
    # Relationship
    lead: Mapped["Lead"] = relationship("Lead", back_populates="score_history")


# Add import for TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.lead import Lead
