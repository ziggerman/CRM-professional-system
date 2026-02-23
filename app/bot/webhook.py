"""
Telegram Webhook handlers.
"""
import hashlib
import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header
from aiogram import Bot, Dispatcher
from aiogram.types import Update

from app.bot.config import bot_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global bot instance
bot: Optional[Bot] = None
dp: Optional[Dispatcher] = None


def init_webhook_bot() -> Bot:
    """Initialize bot for webhook mode."""
    global bot
    if bot is None:
        bot = Bot(token=bot_settings.TELEGRAM_BOT_TOKEN)
    return bot


def get_dispatcher() -> Dispatcher:
    """Get dispatcher with registered handlers."""
    global dp
    if dp is None:
        from app.bot.handlers import router as handlers_router
        dp = Dispatcher()
        dp.include_router(handlers_router)
    return dp


def _verify_secret(secret: str) -> bool:
    """Verify webhook secret token."""
    if not bot_settings.TELEGRAM_WEBHOOK_SECRET:
        return True  # No secret configured
    
    expected = bot_settings.TELEGRAM_WEBHOOK_SECRET
    return secret == expected


@router.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None, alias="X-Telegram-Bot-Api-Secret-Token"),
):
    """
    Telegram webhook endpoint.
    
    Set webhook URL in Telegram:
    https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://your-domain.com/webhook/telegram
    """
    # Verify secret token if configured
    if x_telegram_bot_api_secret_token:
        if not _verify_secret(x_telegram_bot_api_secret_token):
            logger.warning("Invalid webhook secret token")
            raise HTTPException(status_code=403, detail="Invalid secret token")
    
    try:
        update = Update.model_validate(await request.json())
    except Exception as e:
        logger.error(f"Failed to parse update: {e}")
        raise HTTPException(status_code=400, detail="Invalid update")
    
    # Initialize bot and dispatcher
    webhook_bot = init_webhook_bot()
    dispatcher = get_dispatcher()
    
    # Process update
    try:
        await dispatcher.feed_update(webhook_bot, update)
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        raise HTTPException(status_code=500, detail="Processing error")
    
    return {"ok": True}


@router.get("/webhook/telegram/info")
async def webhook_info():
    """Get webhook status."""
    webhook_bot = init_webhook_bot()
    
    try:
        webhook_info = await webhook_bot.get_webhook_info()
        me = await webhook_bot.get_me()
        
        return {
            "ok": True,
            "bot": {
                "id": me.id,
                "name": me.first_name,
                "username": me.username,
            },
            "webhook": {
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_updates_count": webhook_info.pending_updates_count,
            },
        }
    except Exception as e:
        logger.error(f"Error getting webhook info: {e}")
        return {"ok": False, "error": str(e)}


async def setup_webhook(webhook_url: str) -> bool:
    """
    Set up webhook automatically.
    
    Args:
        webhook_url: Full URL for webhook (e.g., https://domain.com/webhook/telegram)
    """
    webhook_bot = init_webhook_bot()
    
    try:
        # Delete any existing webhook
        await webhook_bot.delete_webhook()
        
        # Set new webhook with secret
        secret = bot_settings.TELEGRAM_WEBHOOK_SECRET or None
        
        await webhook_bot.set_webhook(
            url=webhook_url,
            secret_token=secret,
            drop_pending_updates=True,
        )
        
        logger.info(f"Webhook set to: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        return False
