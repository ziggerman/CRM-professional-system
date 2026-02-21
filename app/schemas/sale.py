"""
Pydantic schemas for Sale API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.sale import SaleStage


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class SaleCreate(BaseModel):
    """Schema for creating a new sale (usually done via transfer)."""
    lead_id: int
    stage: SaleStage = SaleStage.NEW
    amount: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=1024)


class SaleStageUpdate(BaseModel):
    """Schema for updating sale stage."""
    stage: SaleStage


class SaleUpdate(BaseModel):
    """Schema for updating sale details."""
    amount: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=1024)


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────

class SaleResponse(BaseModel):
    """Schema for sale response."""
    id: int
    lead_id: int
    stage: SaleStage
    amount: Optional[int]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SaleListResponse(BaseModel):
    """Schema for paginated sale list."""
    items: list[SaleResponse]
    total: int
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Items per page")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_prev: bool = Field(default=False, description="Whether there are previous pages")
