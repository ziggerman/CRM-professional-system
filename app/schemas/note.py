"""
Pydantic schemas for Lead Notes API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NoteCreate(BaseModel):
    """Schema for creating a new note."""
    content: str = Field(..., min_length=1, max_length=5000)
    note_type: str = Field(default="comment", pattern="^(comment|system|ai)$")
    author_id: Optional[str] = Field(None, max_length=64)
    author_name: Optional[str] = Field(None, max_length=128)


class NoteUpdate(BaseModel):
    """Schema for updating a note."""
    content: str = Field(..., min_length=1, max_length=5000)


class NoteResponse(BaseModel):
    """Schema for note response."""
    id: int
    lead_id: int
    author_id: Optional[str]
    author_name: Optional[str]
    content: str
    note_type: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class NoteListResponse(BaseModel):
    """Schema for paginated note list."""
    items: list[NoteResponse]
    total: int
    page: int = Field(default=1)
    page_size: int = Field(default=50)
    has_next: bool = False
    has_prev: bool = False
