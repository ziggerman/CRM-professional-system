"""
Lead model for CRM pipeline.
"""
import enum
from datetime import datetime, UTC

from sqlalchemy import String, Float, Integer, DateTime, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base
from app.models.note import LeadNote
from app.models.user import User


class LeadSource(str, enum.Enum):
    """Lead source types."""
    SCANNER = "scanner"
    PARTNER = "partner"
    MANUAL = "manual"


class BusinessDomain(str, enum.Enum):
    """Business domain categories."""
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"


class ColdStage(str, enum.Enum):
    """Cold lead stages."""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    TRANSFERRED = "transferred"
    LOST = "lost"


# Ordered sequence for validation — cannot skip steps
COLD_STAGE_ORDER = [
    ColdStage.NEW,
    ColdStage.CONTACTED,
    ColdStage.QUALIFIED,
    ColdStage.TRANSFERRED,
    ColdStage.LOST,
]

# Stages that are terminal — cannot be changed once set
TERMINAL_COLD_STAGES = {ColdStage.TRANSFERRED, ColdStage.LOST}


class Lead(Base):
    """Lead model - represents a potential customer."""
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    source: Mapped[LeadSource] = mapped_column(SAEnum(LeadSource), nullable=False)
    stage: Mapped[ColdStage] = mapped_column(
        SAEnum(ColdStage), nullable=False, default=ColdStage.NEW
    )
    business_domain: Mapped[BusinessDomain | None] = mapped_column(
        SAEnum(BusinessDomain), nullable=True
    )

    # Assignment
    assigned_to_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Activity metrics
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # AI fields — stored for auditability
    ai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_recommendation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ai_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ai_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    sale: Mapped["Sale | None"] = relationship("Sale", back_populates="lead", uselist=False)
    assigned_to: Mapped["User | None"] = relationship("User", back_populates="assigned_leads")
    notes: Mapped[list["LeadNote"]] = relationship(
        "LeadNote", 
        back_populates="lead", 
        cascade="all, delete-orphan",
        order_by="LeadNote.created_at.desc()"
    )
