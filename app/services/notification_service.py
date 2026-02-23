"""
Notification Service - Centralized Telegram notifications for the backend.
"""
import logging
from aiogram import Bot
from app.bot.config import bot_settings

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for sending Telegram notifications from background tasks."""
    
    def __init__(self, token: str = None):
        self.token = token or bot_settings.TELEGRAM_BOT_TOKEN
        self._bot = None

    @property
    def bot(self) -> Bot:
        if self._bot is None:
            if not self.token:
                logger.error("TELEGRAM_BOT_TOKEN not found in settings")
                return None
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_direct(self, telegram_id: str | int, text: str):
        """Send a message to a specific Telegram user."""
        if not self.bot:
            return False
            
        try:
            await self.bot.send_message(telegram_id, text, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification to {telegram_id}: {e}")
            return False

    async def notify_admins(self, text: str):
        """Broadcast a message to all configured admins."""
        if not bot_settings.TELEGRAM_ADMIN_IDS:
            logger.warning("No admin IDs configured for notifications")
            return 0
            
        success_count = 0
        for admin_id in bot_settings.TELEGRAM_ADMIN_IDS:
            if await self.send_direct(admin_id, text):
                success_count += 1
        return success_count

    async def notify_all_managers(self, text: str, db_session) -> int:
        """Broadcast a message to all active managers/admins in the DB (Step 8.3)."""
        from app.repositories.user_repo import UserRepository
        repo = UserRepository(db_session)
        users = await repo.get_all()
        
        targets = [u.telegram_id for u in users if u.is_active and u.telegram_id]
        
        success_count = 0
        for tid in targets:
            if await self.send_direct(tid, text):
                success_count += 1
        return success_count

    async def close(self):
        """Close the bot session."""
        if self._bot:
            await self._bot.session.close()
            self._bot = None
