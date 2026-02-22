"""
Pydantic schemas for Lead API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.lead import LeadSource, BusinessDomain, ColdStage
from app.core.sanitization import sanitize_short, sanitize_long


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class LeadCreate(BaseModel):
    """Schema for creating a new lead."""
    source: LeadSource
    business_domain: Optional[BusinessDomain] = None
    telegram_id: Optional[str] = Field(None, max_length=64)
    
    # New fields
    full_name: Optional[str] = Field(None, max_length=128)
    email: Optional[str] = Field(None, max_length=128)
    phone: Optional[str] = Field(None, max_length=32)
    external_username: Optional[str] = Field(None, max_length=128)
    intent: Optional[str] = Field(None, max_length=256)
    company: Optional[str] = Field(None, max_length=128)
    position: Optional[str] = Field(None, max_length=128)
    budget: Optional[str] = Field(None, max_length=64)
    pain_points: Optional[str] = Field(None, max_length=1024)

    @field_validator("full_name", "email", "company", "position", "external_username", mode="before")
    @classmethod
    def sanitize_short_fields(cls, v: Optional[str]) -> Optional[str]:
        return sanitize_short(v)

    @field_validator("intent", "budget", "pain_points", mode="before")
    @classmethod
    def sanitize_long_fields(cls, v: Optional[str]) -> Optional[str]:
        return sanitize_long(v)


class LeadUpdate(BaseModel):
    """Schema for updating lead details."""
    source: Optional[LeadSource] = None
    business_domain: Optional[BusinessDomain] = None
    full_name: Optional[str] = Field(None, max_length=128)
    email: Optional[str] = Field(None, max_length=128)
    phone: Optional[str] = Field(None, max_length=32)
    external_username: Optional[str] = Field(None, max_length=128)
    intent: Optional[str] = Field(None, max_length=256)
    company: Optional[str] = Field(None, max_length=128)
    position: Optional[str] = Field(None, max_length=128)
    budget: Optional[str] = Field(None, max_length=64)
    pain_points: Optional[str] = Field(None, max_length=1024)


class LeadStageUpdate(BaseModel):
    """Schema for updating lead stage."""
    stage: ColdStage

    @field_validator("stage", mode="before")
    @classmethod
    def coerce_to_upper(cls, v: str) -> str:
        if isinstance(v, str):
            return v.upper()
        return v


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
    
    # New fields
    full_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    external_username: Optional[str]
    intent: Optional[str]
    company: Optional[str]
    position: Optional[str]
    budget: Optional[str]
    pain_points: Optional[str]

    ai_score: Optional[float]
    ai_recommendation: Optional[str]
    ai_reason: Optional[str]
    ai_analyzed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class LeadAttachmentResponse(BaseModel):
    """Schema for lead attachments."""
    id: int
    lead_id: int
    file_name: str
    file_type: str
    file_path: str
    file_size_bytes: Optional[int]
    uploaded_by: Optional[str]
    created_at: datetime
    
    model_config = {"from_attributes": True}

class LeadHistoryResponse(BaseModel):
    """Schema for lead history records."""
    id: int
    lead_id: int
    old_stage: Optional[str]
    new_stage: str
    changed_by: str
    reason: Optional[str]
    created_at: datetime
    
    model_config = {"from_attributes": True}


class LeadListResponse(BaseModel):
    """Schema for paginated lead list."""
    items: list[LeadResponse]
    total: int
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_prev: bool = Field(default=False, description="Whether there are previous pages")
