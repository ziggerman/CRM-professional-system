from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import leads, sales, automation, dashboard
from app.api.health import router as health_router
from app.bot.webhook import router as webhook_router
from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware
from app.api.rate_limit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging
    setup_logging()
    
    # Startup - create tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
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
app.include_router(leads.router, prefix="/api/v1/leads", tags=["leads"])
app.include_router(sales.router, prefix="/api/v1/sales", tags=["sales"])
app.include_router(automation.router, prefix="/api/v1/automation", tags=["automation"])
app.include_router(dashboard.router, tags=["dashboard"])
