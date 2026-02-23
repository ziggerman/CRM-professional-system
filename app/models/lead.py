"""
Lead model for CRM pipeline - ТЗ version.
Only contains fields required by the specification.
"""
import enum
from typing import TYPE_CHECKING
from datetime import datetime, UTC

from sqlalchemy import String, Float, Integer, DateTime, Enum as SAEnum, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base

if TYPE_CHECKING:
    from app.models.sale import Sale
    from app.models.user import User
    from app.models.note import LeadNote
    from app.models.attachment import LeadAttachment
    from app.models.history import LeadHistory
    from app.models.activity import LeadActivity


class LeadSource(str, enum.Enum):
    """Lead source types - ТЗ: scanner / partner / manual"""
    SCANNER = "SCANNER"
    PARTNER = "PARTNER"
    MANUAL = "MANUAL"


class BusinessDomain(str, enum.Enum):
    """Business domain categories - ТЗ: first / second / third"""
    FIRST = "FIRST"
    SECOND = "SECOND"
    THIRD = "THIRD"


class ColdStage(str, enum.Enum):
    """Cold lead stages - ТЗ: new/contacted/qualified/transferred/lost"""
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    QUALIFIED = "QUALIFIED"
    TRANSFERRED = "TRANSFERRED"
    LOST = "LOST"


class LostReason(str, enum.Enum):
    """Standardized reasons for lost leads (required for analytics)."""
    NO_BUDGET = "NO_BUDGET"
    NO_RESPONSE = "NO_RESPONSE"
    COMPETITOR = "COMPETITOR"
    NOT_INTERESTED = "NOT_INTERESTED"
    INVALID_CONTACT = "INVALID_CONTACT"
    OTHER = "OTHER"


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

# Reversible stage transitions for rollback
REVERSIBLE_STAGE_TRANSITIONS = {
    ColdStage.CONTACTED: ColdStage.NEW,
    ColdStage.QUALIFIED: ColdStage.CONTACTED,
}


# Quality tier enum for quick filtering
class QualityTier(str, enum.Enum):
    """Lead quality tiers based on AI score."""
    HOT = "HOT"      # score >= 0.8
    WARM = "WARM"    # 0.6 <= score < 0.8
    COLD = "COLD"    # 0.3 <= score < 0.6
    DEAD = "DEAD"    # score < 0.3


def calculate_quality_tier(score: float | None) -> QualityTier | None:
    """Calculate quality tier from AI score."""
    if score is None:
        return None
    if score >= 0.8:
        return QualityTier.HOT
    if score >= 0.6:
        return QualityTier.WARM
    if score >= 0.3:
        return QualityTier.COLD
    return QualityTier.DEAD


class Lead(Base):
    """Lead model - ТЗ version.
    
    Fields according to specification:
    - source: scanner / partner / manual
    - stage: new / contacted / qualified / transferred / lost  
    - business_domain: first / second / third
    - message_count: activity (number of communications)
    - ai_score: AI probability of successful deal
    """
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Required fields according to ТЗ
    source: Mapped[LeadSource] = mapped_column(SAEnum(LeadSource), nullable=False)
    stage: Mapped[ColdStage] = mapped_column(
        SAEnum(ColdStage), nullable=False, default=ColdStage.NEW
    )
    business_domain: Mapped[BusinessDomain | None] = mapped_column(
        SAEnum(BusinessDomain), nullable=True
    )
    lost_reason: Mapped[LostReason | None] = mapped_column(
        SAEnum(LostReason), nullable=True
    )
    
    # Activity - number of communications
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # AI evaluation fields - ТЗ: "вероятность успешной продажи (оценка AI)"
    ai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_recommendation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ai_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ai_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    quality_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Assignment
    assigned_to_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Soft delete fields
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    
    # SLA and response time tracking (Step 7 - KPI Dashboard)
    first_response_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sla_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_overdue: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    days_in_stage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Relationships
    assigned_to: Mapped["User | None"] = relationship("User", back_populates="leads", foreign_keys=[assigned_to_id])
    
    # Relationship to Sale
    sale: Mapped["Sale | None"] = relationship(
        "Sale", 
        back_populates="lead", 
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    # Relationship to Notes
    notes: Mapped[list["LeadNote"]] = relationship(
        "LeadNote",
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadNote.created_at.desc()"
    )
    
    # Relationship to Attachments
    attachments: Mapped[list["LeadAttachment"]] = relationship(
        "LeadAttachment",
        back_populates="lead",
        cascade="all, delete-orphan"
    )
    
    # Relationship to History
    history: Mapped[list["LeadHistory"]] = relationship(
        "LeadHistory",
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadHistory.created_at.desc()"
    )
    
    # Relationship to Activities (for KPI tracking)
    activities: Mapped[list["LeadActivity"]] = relationship(
        "LeadActivity",
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadActivity.created_at.desc()"
    )
