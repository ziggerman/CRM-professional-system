"""
Database configuration and session management.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.base import Base
from app.core.config import settings

# Import models to register them with Base.metadata
from app.models.lead import Lead
from app.models.sale import Sale
from app.models.note import LeadNote

# Create async engine
# SQLite doesn't support pool_size/max_overflow, so we conditionally add them
engine_args = {
    "echo": settings.DEBUG,
}

# Only add pool settings for PostgreSQL (not SQLite)
if "sqlite" not in settings.DATABASE_URL:
    engine_args.update({
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
    })

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    **engine_args,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.
    Usage: async def endpoint(db: AsyncSession = Depends(get_db)):
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
