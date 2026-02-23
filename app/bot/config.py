"""
Telegram Bot configuration.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Telegram Bot settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields in .env
    )

    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = Field(default="")
    TELEGRAM_ADMIN_IDS: list[int] = Field(default_factory=list)
    API_SECRET_TOKEN: str = Field(default="dev_secret_token_123")
    API_BASE_URL: str = Field(default="http://localhost:8000")
    
    # Webhook settings
    TELEGRAM_WEBHOOK_URL: Optional[str] = None
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None


@lru_cache
def get_bot_settings() -> BotSettings:
    """Get cached bot settings instance."""
    return BotSettings()


# Global bot settings instance
bot_settings = get_bot_settings()
