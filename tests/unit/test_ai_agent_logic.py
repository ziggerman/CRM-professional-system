import pytest

from app.ai.voice_ai_manager import VoiceAIManager, Intent
from app.ai.unified_ai_service import UnifiedAIService


class TestVoiceAIManagerContext:
    def test_resolve_pronoun_from_context(self):
        manager = VoiceAIManager()
        user_id = 101

        manager.update_context_lead(user_id, 42, "Nikolas")
        text, lead_id, lead_name = manager.resolve_pronoun("Додай нотатку до нього", user_id)

        assert text == "Додай нотатку до нього"
        assert lead_id == 42
        assert lead_name == "Nikolas"

    @pytest.mark.asyncio
    async def test_process_text_add_note_uses_last_lead_context(self):
        manager = VoiceAIManager()
        user_id = 102
        manager.update_context_lead(user_id, 7, "Lead Seven")

        result = await manager.process_text("додай нотатку важливий контакт", user_id)

        assert result["success"] is True
        assert result["action"].intent == Intent.ADD_NOTE
        assert "#7" in result["response"]

    @pytest.mark.asyncio
    async def test_process_text_edit_requires_confirmation(self):
        manager = VoiceAIManager()
        user_id = 103

        result = await manager.process_text("редагуй лід #12", user_id)

        assert result["needs_confirmation"] is True
        assert "ПІДТВЕРДЖЕННЯ" in result["response"]


class TestUnifiedAIServiceParsing:
    def test_parse_command_maps_intent_and_entities(self):
        service = UnifiedAIService()

        parsed = service.parse_command("видали лід #15", user_id=201)

        assert parsed["action"] == "delete"
        assert parsed["lead_data"]["lead_id"] == 15
        assert parsed["confidence"] >= 0.8

    def test_parse_command_uses_context_for_followup_note(self):
        service = UnifiedAIService()
        user_id = 202
        service.update_context(user_id=user_id, lead_id=88, lead_name="Lead 88", action="view")

        parsed = service.parse_command("додай нотатку передзвонити завтра", user_id=user_id)

        assert parsed["action"] == "note"
        assert parsed["lead_data"]["lead_id"] == 88


class TestVoiceQualityAssessment:
    def test_quality_empty_text_is_low(self):
        manager = VoiceAIManager()
        quality = manager.assess_transcription_quality("")

        assert quality["label"] == "LOW"
        assert quality["needs_clarification"] is True

    def test_quality_normal_text_is_high_or_medium(self):
        manager = VoiceAIManager()
        quality = manager.assess_transcription_quality("додай нотатку до ліда #12 передзвонити завтра")

        assert quality["label"] in {"HIGH", "MEDIUM"}
        assert quality["score"] > 0.5

    def test_unified_service_exposes_quality_assessment(self):
        service = UnifiedAIService()
        quality = service.assess_transcription_quality("покажи ліди")

        assert "score" in quality
        assert "label" in quality
