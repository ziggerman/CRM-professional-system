import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from app.bot.config import bot_settings
from app.bot.handlers import router as handlers_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Start the bot in polling mode."""
    if not bot_settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment")
        return

    # Initialize bot and dispatcher
    bot = Bot(
        token=bot_settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML")
    )
    dp = Dispatcher()
    dp.include_router(handlers_router)

    logger.info("Starting Bot in POLLING mode...")
    
    # Setup native Telegram menu commands
    from aiogram.types import BotCommand
    commands = [
        BotCommand(command="start", description="Restart the bot"),
        BotCommand(command="menu", description="Open Main Menu"),
        BotCommand(command="settings", description="Manage Settings"),
        BotCommand(command="help", description="Show Help Information"),
    ]
    await bot.set_my_commands(commands)
    
    # Delete webhook if it was set
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
