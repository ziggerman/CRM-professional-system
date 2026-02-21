"""
Manager/User model for lead assignment.
"""
from datetime import datetime, UTC
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Boolean, Integer, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.base import Base

if TYPE_CHECKING:
    from app.models.lead import Lead


class UserRole(str, enum.Enum):
    """User roles."""
    ADMIN = "admin"
    MANAGER = "manager"
    VIEWER = "viewer"


class User(Base):
    """User/Manager model for lead assignment."""
    
    __tablename__ = "users"
    __allow_unmapped__ = True
    
    id: int = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Optional[str] = mapped_column(String(64), unique=True, nullable=True, index=True)
    username: Optional[str] = mapped_column(String(128), nullable=True)
    full_name: str = mapped_column(String(256), nullable=False)
    role: str = mapped_column(SAEnum(UserRole), default=UserRole.MANAGER)
    is_active: bool = mapped_column(Boolean, default=True)
    
    # Assignment settings
    max_leads: int = mapped_column(Integer, default=50)
    current_leads: int = mapped_column(Integer, default=0)
    domains: Optional[str] = mapped_column(String(512), nullable=True)
    
    # Stats
    leads_handled: int = mapped_column(Integer, default=0)
    sales_converted: int = mapped_column(Integer, default=0)
    
    created_at: datetime = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: datetime = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC)
    )
    
    # Relationships
    assigned_leads: list["Lead"] = relationship("Lead", back_populates="assigned_to")
    
    def __repr__(self):
        return f"<User id={self.id} name={self.full_name} role={self.role.value}>"
