from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import (
    leads, sales, automation, dashboard, users, export, ws, auth, notifications
)
from app.api.health import router as health_router
from app.bot.webhook import router as webhook_router
from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware
from app.api.rate_limit import RateLimitMiddleware

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging
    setup_logging()
    
    # Startup - create tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Warm up AI model (Step 5.3) logic
        from app.ai.ai_service import AIService
        import asyncio
        asyncio.create_task(AIService().warm_up())
    except Exception as e:
        print(f"Warning: Could not create tables: {e}")
    
    yield
    
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title="CRM Lead Management",
    description="AI-assisted lead pipeline management",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument Prometheus
if HAS_PROMETHEUS:
    Instrumentator().instrument(app).expose(app)
else:
    print("Warning: prometheus_fastapi_instrumentator not found, metrics disabled.")

# Add middleware (order matters - last added = first executed)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(webhook_router)
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(notifications.router, prefix="/api/v1", tags=["notifications"])
app.include_router(leads.router, prefix="/api/v1/leads", tags=["leads"])
app.include_router(sales.router, prefix="/api/v1/sales", tags=["sales"])
app.include_router(automation.router, prefix="/api/v1/automation", tags=["automation"])
app.include_router(dashboard.router, tags=["dashboard"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(export.router, prefix="/api/v1/export", tags=["export"])
app.add_api_websocket_route("/api/v1/ws/dashboard", ws.websocket_dashboard)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
