"""
Health check endpoints for monitoring and container orchestration.
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import settings
from app.api.errors import build_error_payload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db)):
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
    redis_detail = None
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
        redis_detail = str(e)

    # 3. Celery Check
    celery_status = "healthy"
    celery_detail = None
    try:
        from app.celery.config import celery_app

        insp = celery_app.control.inspect(timeout=1.0)
        ping = insp.ping() or {}
        if not ping:
            celery_status = "degraded"
            celery_detail = "No celery worker responded to ping"
    except Exception as e:
        celery_status = "unhealthy"
        celery_detail = str(e)

    # 4. OpenAI Config Check (no real request to save cost)
    openai_status = "healthy" if settings.OPENAI_API_KEY and "sk-" in settings.OPENAI_API_KEY else "unconfigured"

    # Overall status
    components = {
        "database": {"status": db_status, "latency_ms": db_latency},
        "redis": {"status": redis_status, "latency_ms": redis_latency, "detail": redis_detail},
        "celery": {"status": celery_status, "detail": celery_detail},
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
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness check - verifies database connection.
    Used by Kubernetes for pod readiness.
    """
    try:
        # Try to execute a simple query
        await db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected",
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=build_error_payload(
                code="readiness_failed",
                message="Database unavailable",
                detail=str(e),
            ),
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - simple check that app is running.
    Used by Kubernetes for pod liveness.
    """
    return {"status": "alive"}
