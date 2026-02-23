"""
Telegram Bot Middleware - FSM Timeout and User Activity Tracking
Automatically clears states after user inactivity to prevent stuck conversations.
"""
import logging
import time
from typing import Any, Dict, TYPE_CHECKING

from aiogram import Dispatcher, Bot, types
from aiogram.types import Update

logger = logging.getLogger(__name__)

# Timeout in seconds (5 minutes)
FSM_TIMEOUT_SECONDS = 300  # 5 minutes

# States that should NOT be auto-cleared on timeout (they have their own flow)
PROTECTED_STATES = frozenset([
    "VoiceChatState:active",
])


class FSMTimeoutMiddleware:
    """
    Middleware that automatically clears FSM state after user inactivity.
    This prevents users from getting stuck in old conversation flows.
    """
    
    def __init__(self, timeout_seconds: int = FSM_TIMEOUT_SECONDS):
        self.timeout = timeout_seconds
        self._last_activity: Dict[int, float] = {}
    
    async def __call__(self, handler, event: types.Update, data: dict):
        # Get user ID from event
        user_id = self._get_user_id(event)
        if not user_id:
            return await handler(event, data)
        
        # Check if user has an active state
        from aiogram.fsm.context import FSMContext
        state: FSMContext = data.get("state")
        if state:
            current_state = await state.get_state()
            
            if current_state and current_state not in PROTECTED_STATES:
                # Get last activity time
                last_activity = self._last_activity.get(user_id, 0)
                current_time = time.time()
                
                # Check if timeout has passed
                if current_time - last_activity > self.timeout:
                    # State has timed out - clear it and notify user
                    logger.info(f"FSM timeout for user {user_id}, state: {current_state}")
                    await state.clear()
                    
                    # Send timeout message (only for messages, not callbacks)
                    # Main menu is disabled - just clear state without showing menu
                    if isinstance(event, types.Message):
                        await event.answer(
                            "⏰ <b>Час очікування вичерпано</b>\n\n"
                            "Вашу сесію було очищено через неактивність. "
                            "Напишіть /start щоб почати знову.",
                            parse_mode="HTML"
                        )
                    # For callbacks, we'll just clear state silently
        
        # Update last activity time
        self._last_activity[user_id] = time.time()
        
        # Continue with handler
        return await handler(event, data)
    
    def _get_user_id(self, event: types.Update) -> int | None:
        """Extract user ID from update event."""
        if event.message:
            return event.message.from_user.id
        elif event.callback_query:
            return event.callback_query.from_user.id
        elif event.inline_query:
            return event.inline_query.from_user.id
        elif event.chosen_inline_result:
            return event.chosen_inline_result.from_user.id
        return None


class UserActivityMiddleware:
    """
    Middleware that tracks user activity for analytics and debugging.
    """
    
    def __init__(self):
        self._user_stats: Dict[int, dict] = {}
    
    async def __call__(self, handler, event: types.Update, data: dict):
        user_id = self._get_user_id(event)
        if not user_id:
            return await handler(event, data)
        
        # Update user stats
        if user_id not in self._user_stats:
            self._user_stats[user_id] = {
                "message_count": 0,
                "callback_count": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
        
        stats = self._user_stats[user_id]
        stats["last_seen"] = time.time()
        
        if isinstance(event, types.Message):
            stats["message_count"] += 1
        elif isinstance(event, types.CallbackQuery):
            stats["callback_count"] += 1
        
        # Add stats to data for handlers to access if needed
        data["user_stats"] = stats
        
        return await handler(event, data)
    
    def _get_user_id(self, event: types.Update) -> int | None:
        """Extract user ID from update event."""
        if event.message:
            return event.message.from_user.id
        elif event.callback_query:
            return event.callback_query.from_user.id
        elif event.inline_query:
            return event.inline_query.from_user.id
        elif event.chosen_inline_result:
            return event.chosen_inline_result.from_user.id
        return None
    
    def get_user_stats(self, user_id: int) -> dict | None:
        """Get stats for a specific user."""
        return self._user_stats.get(user_id)


def setup_middleware(dp: Dispatcher, timeout_seconds: int = FSM_TIMEOUT_SECONDS):
    """
    Setup all bot middleware.
    Call this from your bot initialization code.
    """
    # Register FSM timeout middleware
    fsm_timeout = FSMTimeoutMiddleware(timeout_seconds)
    dp.update.middleware.register(fsm_timeout)
    
    # Register user activity middleware
    activity = UserActivityMiddleware()
    dp.update.middleware.register(activity)
    
    logger.info(f"Bot middleware configured with {timeout_seconds}s FSM timeout")
    
    return {
        "fsm_timeout": fsm_timeout,
        "activity": activity,
    }
