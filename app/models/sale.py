"""
Sale model for sales pipeline.
"""
import enum
from datetime import datetime, UTC

from sqlalchemy import String, Integer, DateTime, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base


class SaleStage(str, enum.Enum):
    """Sales pipeline stages."""
    NEW = "new"
    KYC = "kyc"
    AGREEMENT = "agreement"
    PAID = "paid"
    LOST = "lost"


# Ordered sequence for validation — cannot skip steps
SALE_STAGE_ORDER = [
    SaleStage.NEW,
    SaleStage.KYC,
    SaleStage.AGREEMENT,
    SaleStage.PAID,
    SaleStage.LOST,
]

# Stages that are terminal — cannot be changed once set
TERMINAL_SALE_STAGES = {SaleStage.PAID, SaleStage.LOST}


class Sale(Base):
    """Sale model - represents a lead that has been transferred to sales."""
    __tablename__ = "sales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lead_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("leads.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True
    )
    
    stage: Mapped[SaleStage] = mapped_column(
        SAEnum(SaleStage), 
        nullable=False, 
        default=SaleStage.NEW
    )
    
    # Optional: amount in cents
    amount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Notes for sales team
    notes: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    
    # Relationship to lead
    lead: Mapped["Lead"] = relationship("Lead", back_populates="sale")
