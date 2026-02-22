"""
Health check endpoints for monitoring and container orchestration.
"""
import logging
import time
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(lambda: AsyncSessionLocal())):
    """Comprehensive health check for all components."""
    start_time = time.perf_counter()
    
    # 1. DB Check
    db_status = "healthy"
    db_latency = 0
    try:
        db_start = time.perf_counter()
        await session.execute(text("SELECT 1"))
        db_latency = round((time.perf_counter() - db_start) * 1000, 2)
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        db_status = "unhealthy"

    # 2. Redis Check
    redis_status = "healthy"
    redis_latency = 0
    try:
        import redis.asyncio as redis
        db_redis = redis.from_url(settings.REDIS_URL)
        redis_start = time.perf_counter()
        await db_redis.ping()
        redis_latency = round((time.perf_counter() - redis_start) * 1000, 2)
        await db_redis.close()
    except Exception as e:
        logger.warning(f"Health check Redis error: {e}")
        redis_status = "unhealthy"

    # 3. OpenAI Config Check (no real request to save cost)
    openai_status = "healthy" if settings.OPENAI_API_KEY and "sk-" in settings.OPENAI_API_KEY else "unconfigured"

    # 4. Overall status
    components = {
        "database": {"status": db_status, "latency_ms": db_latency},
        "redis": {"status": redis_status, "latency_ms": redis_latency},
        "openai": {"status": openai_status},
    }
    
    is_healthy = all(c["status"] in ["healthy", "unconfigured"] for c in components.values())
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "components": components,
        "total_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
    }


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(lambda: AsyncSessionLocal())):
    """
    Readiness check - verifies database connection.
    Used by Kubernetes for pod readiness.
    """
    try:
        # Try to execute a simple query
        await db.execute("SELECT 1")
        return {
            "status": "ready",
            "database": "connected",
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - simple check that app is running.
    Used by Kubernetes for pod liveness.
    """
    return {"status": "alive"}
