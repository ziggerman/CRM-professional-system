import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from unittest.mock import MagicMock, AsyncMock

# Mock Redis BEFORE importing main.app to ensure middleware uses the mock
import redis.asyncio as redis
mock_redis = AsyncMock()
redis.from_url = MagicMock(return_value=mock_redis)
mock_redis.get.return_value = None
mock_redis.setex.return_value = True
mock_redis.incr = MagicMock(return_value=None) # Queues in pipeline
mock_redis.expire = MagicMock(return_value=None) # Queues in pipeline
mock_redis.pipeline = MagicMock(return_value=mock_redis) # Returns self (the mock redis)
mock_redis.__aenter__ = AsyncMock(return_value=mock_redis)
mock_redis.__aexit__ = AsyncMock(return_value=None)
mock_redis.execute = AsyncMock(return_value=[1, 1]) # For rate limit window

from main import app
from app.core.database import get_db, Base
from app.core.config import settings

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_token(client: AsyncClient) -> str:
    # Use the static dev token for simple testing of protected endpoints
    # Or implement a real login if needed.
    return settings.API_SECRET_TOKEN
