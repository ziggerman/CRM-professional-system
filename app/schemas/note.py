"""
Pydantic schemas for Lead Notes API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.sanitization import sanitize_long


class NoteCreate(BaseModel):
    """Schema for creating a new note."""
    content: str = Field(..., min_length=1, max_length=5000)
    note_type: str = Field(
        default="general",
        pattern="^(comment|system|ai|general|contact|email|meeting|problem|success|task|objection)$"
    )
    category: Optional[str] = Field(None, max_length=32)
    author_id: Optional[str] = Field(None, max_length=64)
    author_name: Optional[str] = Field(None, max_length=128)

    @field_validator("content", mode="before")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return sanitize_long(v) or ""

    @field_validator("note_type", mode="before")
    @classmethod
    def normalize_note_type(cls, v: str) -> str:
        if not v:
            return "general"
        value = str(v).strip().lower()
        alias_map = {
            "note": "general",
            "notes": "general",
            "category": "general",
            "call": "contact",
            "meeting_note": "meeting",
            "issue": "problem",
        }
        return alias_map.get(value, value)

    @model_validator(mode="after")
    def merge_category_into_note_type(self):
        """Backward compatibility: accept 'category' from bot payloads."""
        if self.category and (not self.note_type or self.note_type == "general"):
            normalized = self.normalize_note_type(self.category)
            allowed = {"comment", "system", "ai", "general", "contact", "email", "meeting", "problem", "success", "task", "objection"}
            self.note_type = normalized if normalized in allowed else "general"
        return self


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
    is_pinned: bool = False
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
