"""
User Repository - database operations for users.
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:
    """Repository for User model operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_all(self) -> list[User]:
        """Get all users."""
        result = await self.session.execute(select(User))
        return list(result.scalars().all())
    
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
        await self.session.flush()
        await self.session.refresh(user)
        return user
    
    async def delete(self, user: User) -> None:
        """Delete user."""
        await self.session.delete(user)
        await self.session.flush()
