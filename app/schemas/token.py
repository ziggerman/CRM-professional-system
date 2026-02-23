from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str


class TokenData(BaseModel):
    """Token payload data schema."""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None
