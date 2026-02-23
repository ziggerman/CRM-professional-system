"""
Unified AI Copilot service (text + voice).

Thin orchestration layer over UnifiedAIService to keep one entry point
for bot handlers and future expansion (confirm/clarify workflows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.unified_ai_service import unified_ai


@dataclass
class CopilotResult:
    action: str
    lead_data: dict[str, Any]
    query: str
    confidence: float
    raw_text: str


class AICopilotService:
    """Single AI copilot facade used by both text and voice flows."""

    async def transcribe_voice(self, voice_content: bytes) -> str | None:
        return await unified_ai.transcribe_voice(voice_content)

    def parse(self, text: str, user_id: int | None = None) -> CopilotResult:
        parsed = unified_ai.parse_command(text, user_id=user_id)
        return CopilotResult(
            action=parsed.get("action", "ai_query"),
            lead_data=parsed.get("lead_data", {}) or {},
            query=parsed.get("query", text),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            raw_text=parsed.get("raw_text", text),
        )

    async def answer_query(self, query: str, leads: list[dict[str, Any]]) -> str:
        return await unified_ai.process_query(query, leads)

    async def categorize_note(self, note_content: str) -> str:
        return await unified_ai.categorize_note(note_content)

    def assess_transcription_quality(self, text: str) -> dict[str, Any]:
        return unified_ai.assess_transcription_quality(text)

    def resolve_pronoun(self, text: str, user_id: int):
        return unified_ai.resolve_pronoun(text, user_id)

    def update_context(self, user_id: int, lead_id: int | None = None, lead_name: str | None = None, action: str | None = None):
        return unified_ai.update_context(user_id, lead_id=lead_id, lead_name=lead_name, action=action)


copilot_ai = AICopilotService()
