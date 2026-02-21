"""
Application configuration using pydantic-settings.
"""
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields in .env
    )

    # Application
    APP_NAME: str = "CRM Lead Management"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/crm_db"
    )

    # OpenAI
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Business Rules
    MIN_TRANSFER_SCORE: float = 0.6

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # AI Cache TTL (in seconds)
    AI_CACHE_TTL: int = 3600  # 1 hour
    
    # AI Analysis Stale Threshold (in days)
    AI_ANALYSIS_STALE_DAYS: int = 7  # Re-analyze if older than 7 days


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
