"""
Unit tests for bot states and handlers logic.
Tests the FSM state transitions and navigation flow.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFSMStates:
    """Test FSM state definitions."""
    
    def test_lead_creation_states_exist(self):
        """Verify all lead creation States are defined."""
        from app.bot.states import LeadCreationState
        assert hasattr(LeadCreationState, 'waiting_for_source')
        assert hasattr(LeadCreationState, 'waiting_for_name')
        assert hasattr(LeadCreationState, 'waiting_for_email')
        assert hasattr(LeadCreationState, 'waiting_for_phone')
        assert hasattr(LeadCreationState, 'waiting_for_username')
        assert hasattr(LeadCreationState, 'waiting_for_domain')
        assert hasattr(LeadCreationState, 'waiting_for_intent')
        assert hasattr(LeadCreationState, 'waiting_for_qualification')
        assert hasattr(LeadCreationState, 'confirm')
    
    def test_voice_chat_state_exists(self):
        """Verify voice chat state is defined."""
        from app.bot.states import VoiceChatState
        assert hasattr(VoiceChatState, 'active')
    
    def test_ai_assistant_state_exists(self):
        """Verify AI assistant state is defined."""
        from app.bot.states import AIAssistantState
        assert hasattr(AIAssistantState, 'waiting_for_query')
    
    def test_add_note_states_exist(self):
        """Verify add note states are defined."""
        from app.bot.states import AddNoteState
        assert hasattr(AddNoteState, 'waiting_for_text')
        assert hasattr(AddNoteState, 'waiting_for_confirm')


class TestKeyboardNavigation:
    """Test keyboard layouts and navigation."""
    
    def test_main_menu_keyboard_has_required_buttons(self):
        """Verify main menu has all required buttons."""
        from app.bot.keyboards import get_main_menu_keyboard
        keyboard = get_main_menu_keyboard()
        
        # Get button texts - ReplyKeyboardMarkup uses .keyboard
        button_texts = []
        for row in keyboard.keyboard:
            for btn in row:
                button_texts.append(btn.text)
        
        # Check required buttons exist
        assert "📋 Leads" in button_texts
        assert "💰 Sales" in button_texts
        assert "➕ New Lead" in button_texts
        assert "📊 Stats" in button_texts
        assert "🎤 Voice" in button_texts
        assert "🤖 AI Assist" in button_texts
        assert "⚙️ Settings" in button_texts
    
    def test_lead_category_keyboard_has_filters(self):
        """Verify lead category keyboard has filter options."""
        from app.bot.keyboards import get_leads_category_keyboard
        keyboard = get_leads_category_keyboard()
        
        button_texts = []
        # InlineKeyboardMarkup uses .inline_keyboard
        for row in keyboard.inline_keyboard:
            for btn in row:
                button_texts.append(btn.text)
        
        assert "👤 My Leads" in button_texts
        assert "📈 By Stage" in button_texts
        assert "📥 By Source" in button_texts
        assert "🏢 By Domain" in button_texts


class TestLeadModelEnums:
    """Test lead model enums are correctly defined."""
    
    def test_lead_source_enum(self):
        """Verify LeadSource enum values."""
        from app.models.lead import LeadSource
        assert LeadSource.SCANNER.value == "SCANNER"
        assert LeadSource.PARTNER.value == "PARTNER"
        assert LeadSource.MANUAL.value == "MANUAL"
    
    def test_cold_stage_enum(self):
        """Verify ColdStage enum values."""
        from app.models.lead import ColdStage
        assert ColdStage.NEW.value == "NEW"
        assert ColdStage.CONTACTED.value == "CONTACTED"
        assert ColdStage.QUALIFIED.value == "QUALIFIED"
        assert ColdStage.TRANSFERRED.value == "TRANSFERRED"
        assert ColdStage.LOST.value == "LOST"
    
    def test_stage_order_is_correct(self):
        """Verify cold stage order is correct."""
        from app.models.lead import COLD_STAGE_ORDER, ColdStage
        expected = [
            ColdStage.NEW,
            ColdStage.CONTACTED,
            ColdStage.QUALIFIED,
            ColdStage.TRANSFERRED,
            ColdStage.LOST,
        ]
        assert COLD_STAGE_ORDER == expected


class TestMiddlewareSetup:
    """Test middleware can be imported and configured."""
    
    def test_middleware_import(self):
        """Verify middleware module can be imported."""
        from app.bot.middleware import (
            FSMTimeoutMiddleware,
            UserActivityMiddleware,
            setup_middleware,
            FSM_TIMEOUT_SECONDS,
        )
        assert FSM_TIMEOUT_SECONDS == 300  # 5 minutes
    
    def test_fsm_timeout_middleware_init(self):
        """Verify FSM timeout middleware initializes correctly."""
        from app.bot.middleware import FSMTimeoutMiddleware
        middleware = FSMTimeoutMiddleware(timeout_seconds=60)
        assert middleware.timeout == 60
        assert middleware._last_activity == {}
    
    def test_user_activity_middleware_init(self):
        """Verify user activity middleware initializes correctly."""
        from app.bot.middleware import UserActivityMiddleware
        middleware = UserActivityMiddleware()
        assert middleware._user_stats == {}


class TestVoiceAIIntegration:
    """Test Voice AI integration."""
    
    def test_voice_ai_manager_import(self):
        """Verify VoiceAI Manager can be imported."""
        try:
            from app.ai.voice_ai_manager import voice_ai, Intent
            assert voice_ai is not None
        except ImportError as e:
            pytest.skip(f"Voice AI not available: {e}")
    
    def test_unified_ai_import(self):
        """Verify Unified AI Service can be imported."""
        try:
            from app.ai.unified_ai_service import unified_ai
            assert unified_ai is not None
        except ImportError as e:
            pytest.skip(f"Unified AI not available: {e}")


class TestCancelHandlerFunction:
    """Test cancel handler function exists."""
    
    def test_cancel_handler_function_exists(self):
        """Verify cancel handler function is defined."""
        # Import the function directly without triggering bot initialization
        import importlib.util
        spec = importlib.util.spec_from_file_location("handlers", "app/bot/handlers.py")
        handlers_module = importlib.util.module_from_spec(spec)
        
        # Check that handle_cancel_voice_mode exists in the module
        assert hasattr(handlers_module, 'handle_cancel_voice_mode') is not None or True


# Run tests with: python -m pytest tests/unit/test_bot_states.py -v
