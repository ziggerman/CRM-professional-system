from datetime import datetime, UTC
from sqlalchemy import Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.core.base import Base

class AIAnalysisLog(Base):
    """
    Audit log for AI decisions — Step 4.3.
    Stores the full context of what the AI saw and what it decided.
    """
    __tablename__ = "ai_analysis_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lead_id: Mapped[int] = mapped_column(Integer, ForeignKey("leads.id"), index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendation: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Usage metrics
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(64))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
