"""User Repository - database operations for users."""
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import nullsfirst

from app.models.user import User, UserRole


class UserRepository:
    """Repository for User model operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_all(self) -> list[User]:
        """Get all users."""
        result = await self.session.execute(select(User))
        return list(result.scalars().all())

    async def get_round_robin_manager(self, business_domain: str | None = None) -> Optional[User]:
        """
        Return next available manager/admin for auto-assignment.

        Round-robin strategy:
        1) active MANAGER/ADMIN only
        2) with capacity (current_leads < max_leads)
        3) optional domain preference
        4) oldest last_lead_assigned_at first (NULLs first), then lowest load
        """
        stmt = (
            select(User)
            .where(User.is_active == True)
            .where(User.role.in_([UserRole.MANAGER, UserRole.ADMIN]))
            .where(User.current_leads < User.max_leads)
        )

        if business_domain:
            # domains stored as CSV string like "FIRST,SECOND"
            stmt = stmt.where(User.domains.ilike(f"%{business_domain}%"))

        stmt = stmt.order_by(
            nullsfirst(User.last_lead_assigned_at.asc()),
            User.current_leads.asc(),
            User.id.asc(),
        )

        result = await self.session.execute(stmt.limit(1))
        return result.scalar_one_or_none()
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_telegram_id(self, telegram_id: str) -> Optional[User]:
        """Get user by Telegram ID."""
        result = await self.session.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by Email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def create(self, user: User) -> User:
        """Create a new user."""
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def save(self, user: User) -> User:
        """Save/update user."""
        if user.last_lead_assigned_at is None and user.current_leads > 0:
            user.last_lead_assigned_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def delete(self, user: User) -> None:
        """Delete user."""
        await self.session.delete(user)
        await self.session.flush()
