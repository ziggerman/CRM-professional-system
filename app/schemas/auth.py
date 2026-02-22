from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: str
    password: str


class PasswordChangeRequest(BaseModel):
    """Schema for password change/setup."""
    password: str = Field(..., min_length=8, max_length=128)
