"""
Pydantic schemas for Lead API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.lead import LeadSource, BusinessDomain, ColdStage, LostReason
from app.core.sanitization import sanitize_short, sanitize_long


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class LeadCreate(BaseModel):
    """Schema for creating a new lead."""
    full_name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=64)
    email: Optional[str] = Field(None, max_length=255)
    source: LeadSource
    business_domain: Optional[BusinessDomain] = None
    telegram_id: Optional[str] = Field(None, max_length=64)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("business_domain", mode="before")
    @classmethod
    def normalize_domain(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v


class LeadUpdate(BaseModel):
    """Schema for updating lead details."""
    full_name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=64)
    email: Optional[str] = Field(None, max_length=255)
    source: Optional[LeadSource] = None
    business_domain: Optional[BusinessDomain] = None

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("business_domain", mode="before")
    @classmethod
    def normalize_domain(cls, v):
        if isinstance(v, str):
            return v.strip().upper()
        return v


class LeadStageUpdate(BaseModel):
    """Schema for updating lead stage."""
    stage: ColdStage
    lost_reason: Optional[LostReason] = None

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
    full_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    source: LeadSource
    stage: ColdStage
    business_domain: Optional[BusinessDomain]
    lost_reason: Optional[LostReason]
    message_count: int

    ai_score: Optional[float]
    ai_recommendation: Optional[str]
    ai_reason: Optional[str]
    ai_analyzed_at: Optional[datetime]
    quality_tier: Optional[str]
    
    # Assignment
    assigned_to_id: Optional[int]
    
    # SLA tracking
    first_response_at: Optional[datetime] = None
    sla_deadline_at: Optional[datetime] = None
    is_overdue: bool = False
    days_in_stage: int = 0
    
    # Soft delete
    is_deleted: bool
    deleted_at: Optional[datetime]
    deleted_by: Optional[str]
    
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


class CursorPageResponse(BaseModel):
    """Schema for cursor-based pagination response."""
    items: list[LeadResponse]
    total: int
    next_cursor: Optional[str] = Field(default=None, description="Cursor for next page (base64)")
    has_next: bool = Field(default=False, description="Whether there are more items")
    # Legacy support
    page: int = Field(default=1, description="Current page number (deprecated, use cursor)")
    page_size: int = Field(default=50, description="Items per page (deprecated)")
    has_prev: bool = Field(default=False, description="Whether there are previous pages (deprecated)")


class LeadListResponse(BaseModel):
    """Schema for paginated lead list."""
    items: list[LeadResponse]
    total: int
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")
    has_next: bool = Field(default=False, description="Whether there are more Pages")
    has_prev: bool = Field(default=False, description="Whether there are previous pages")
    # Cursor support
    next_cursor: Optional[str] = Field(default=None, description="Cursor for next page")
