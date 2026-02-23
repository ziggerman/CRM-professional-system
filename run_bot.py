import asyncio
import logging
import sys
import os
from pathlib import Path
import fcntl
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from app.bot.config import bot_settings
from app.bot.handlers import router as handlers_router
from app.bot.middleware import setup_middleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

_lock_file = None


def acquire_single_instance_lock() -> bool:
    """Prevent multiple polling instances (fixes TelegramConflictError)."""
    global _lock_file
    lock_path = Path(__file__).parent / ".run_bot.lock"

    _lock_file = open(lock_path, "w")
    try:
        fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
        return True
    except BlockingIOError:
        return False

async def main():
    """Start the bot in polling mode."""
    if not acquire_single_instance_lock():
        logger.error("Another run_bot.py instance is already running. Exiting to avoid TelegramConflictError.")
        return

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

    # Setup middleware (FSM timeout, user activity tracking)
    setup_middleware(dp)

    logger.info("Starting Bot in POLLING mode...")
    
    # Setup native Telegram menu commands
    from aiogram.types import BotCommand
    commands = [
        BotCommand(command="start", description="Restart the bot"),
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
