"""
Pydantic schemas for Lead API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.lead import LeadSource, BusinessDomain, ColdStage


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class LeadCreate(BaseModel):
    """Schema for creating a new lead."""
    source: LeadSource
    business_domain: Optional[BusinessDomain] = None
    telegram_id: Optional[str] = Field(None, max_length=64)


class LeadStageUpdate(BaseModel):
    """Schema for updating lead stage."""
    stage: ColdStage


class LeadMessageUpdate(BaseModel):
    """Schema for incrementing message count."""
    increment: int = Field(default=1, ge=1, le=100)


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────

class AIAnalysisResult(BaseModel):
    """Schema for AI analysis results."""
    score: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    reason: str


class LeadResponse(BaseModel):
    """Schema for lead response."""
    id: int
    telegram_id: Optional[str]
    source: LeadSource
    stage: ColdStage
    business_domain: Optional[BusinessDomain]
    message_count: int
    ai_score: Optional[float]
    ai_recommendation: Optional[str]
    ai_reason: Optional[str]
    ai_analyzed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LeadListResponse(BaseModel):
    """Schema for paginated lead list."""
    items: list[LeadResponse]
    total: int
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_prev: bool = Field(default=False, description="Whether there are previous pages")
