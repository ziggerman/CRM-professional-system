from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base


class LeadHistory(Base):
    """Audit log for lead transitions and important events."""
    
    __tablename__ = "lead_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    lead_id: Mapped[int] = mapped_column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    
    old_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    new_stage: Mapped[str] = mapped_column(String(50))
    
    changed_by: Mapped[str] = mapped_column(String(100), default="System")
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    
    # Relationship
    lead = relationship("app.models.lead.Lead", back_populates="history")


class SaleHistory(Base):
    """Audit log for sale transitions."""
    
    __tablename__ = "sale_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sale_id: Mapped[int] = mapped_column(Integer, ForeignKey("sales.id", ondelete="CASCADE"), index=True)
    
    old_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    new_stage: Mapped[str] = mapped_column(String(50))
    
    changed_by: Mapped[str] = mapped_column(String(100), default="System")
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    
    # Relationship
    sale = relationship("app.models.sale.Sale", back_populates="history")
