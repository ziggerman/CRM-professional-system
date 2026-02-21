"""
Health check endpoints for monitoring and container orchestration.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health_check():
    """Basic health check - always returns 200 if app is running."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
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
