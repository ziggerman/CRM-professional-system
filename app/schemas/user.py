from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from app.models.user import UserRole


class UserBase(BaseModel):
    """Base user schema."""
    username: Optional[str] = Field(None, max_length=128)
    full_name: str = Field(..., max_length=256)
    email: Optional[str] = None
    role: UserRole = UserRole.MANAGER
    is_active: bool = True
    max_leads: int = 50
    current_leads: int = 0


class UserCreate(UserBase):
    """Schema for creating a new user with password."""
    email: str
    password: str = Field(..., min_length=8, max_length=128)


class UserUpdate(BaseModel):
    """Schema for updating user details."""
    username: Optional[str] = Field(None, max_length=128)
    full_name: Optional[str] = Field(None, max_length=256)
    email: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    telegram_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
