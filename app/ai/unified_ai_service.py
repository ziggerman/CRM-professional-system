"""
Unified AI Service - Combines Voice Processing and AI Assistant.
Provides free voice-to-text (local & API) and improved natural language understanding.
"""
import os
import re
import json
import logging
from io import BytesIO
from typing import Optional

import httpx
from app.ai.voice_ai_manager import voice_ai, Intent, IntentDetector

logger = logging.getLogger(__name__)

# Try to import faster-whisper for local transcription
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install for local offline transcription.")


class UnifiedAIService:
    """
    Unified AI Service that handles:
    - Voice transcription (local or API - FREE & paid options)
    - Text command understanding with CONTEXT AWARENESS
    - Lead queries
    - Note categorization
    
    Transcription priority:
    1. Local faster-whisper (FREE, offline, fastest)
    2. HuggingFace API (FREE, online)
    3. OpenAI Whisper (paid, reliable)
    
    Context features:
    - Remembers last mentioned lead ID
    - Tracks conversation state
    - Supports pronouns ("того", "його", "that lead")
    - Allows follow-up without repeating ID
    """
    
    # Class-level model cache
    _whisper_model = None
    _model_loaded = False
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.local_whisper_model = os.getenv("LOCAL_WHISPER_MODEL", "base")  # tiny, base, small
        self.api_base_url = "http://localhost:8000"
        self._openai_disabled_reason: Optional[str] = None
        
        # Context tracking per user (in production, use Redis or DB)
        self._user_context: dict[int, dict] = {}
    
    def get_user_context(self, user_id: int) -> dict:
        """Get context for user from unified VoiceAIManager."""
        ctx = voice_ai.get_context(user_id)
        return {
            "last_lead_id": ctx.last_lead_id,
            "last_lead_name": ctx.last_lead_name,
            "last_action": ctx.last_action,
            "conversation_history": ctx.conversation_history,
        }
    
    def update_context(self, user_id: int, lead_id: int = None, lead_name: str = None, action: str = None):
        """Update user context in VoiceAIManager."""
        if lead_id:
            voice_ai.update_context_lead(user_id, lead_id, lead_name)
        if action:
            ctx = voice_ai.get_context(user_id)
            ctx.last_action = action
    
    def clear_context(self, user_id: int):
        """Clear user context."""
        voice_ai._user_contexts.pop(user_id, None)
    
    def resolve_pronoun(self, text: str, user_id: int):
        """Resolve pronouns using VoiceAIManager context resolver."""
        return voice_ai.resolve_pronoun(text, user_id)

    def assess_transcription_quality(self, text: str) -> dict:
        """Expose voice transcription quality assessment from VoiceAIManager."""
        return voice_ai.assess_transcription_quality(text)
    
    # ==================== VOICE PROCESSING ====================
    
    async def transcribe_voice(self, voice_content: bytes) -> str | None:
        """
        Transcribe voice to text.
        Priority: Local (faster-whisper) > HuggingFace API > OpenAI API
        """
        # Normalize incoming Telegram payload (bytes or BytesIO)
        voice_bytes = self._ensure_bytes(voice_content)
        if not voice_bytes:
            logger.error("Empty voice payload received")
            return None

        # 1. Try local faster-whisper (FREE, offline, fastest)
        if FASTER_WHISPER_AVAILABLE:
            result = await self._transcribe_local(voice_bytes)
            if result:
                logger.info("Used local faster-whisper transcription")
                return result
        
        # 2. Try HuggingFace API (FREE, online)
        if self.huggingface_token:
            result = await self._transcribe_huggingface(voice_bytes)
            if result:
                logger.info("Used HuggingFace API transcription")
                return result
        
        # 3. Try OpenAI Whisper (paid, reliable)
        if self.openai_api_key:
            result = await self._transcribe_openai(voice_bytes)
            if result:
                logger.info("Used OpenAI Whisper transcription")
                return result
        
        logger.error("No voice transcription service available")
        return None

    def _ensure_bytes(self, voice_content) -> bytes:
        """Normalize voice payload from aiogram download_file (BytesIO) or raw bytes."""
        if voice_content is None:
            return b""
        if isinstance(voice_content, (bytes, bytearray)):
            return bytes(voice_content)
        if isinstance(voice_content, BytesIO):
            return voice_content.getvalue()
        # aiogram may return file-like objects
        if hasattr(voice_content, "getvalue"):
            try:
                return voice_content.getvalue()
            except Exception:
                pass
        if hasattr(voice_content, "read"):
            try:
                pos = voice_content.tell() if hasattr(voice_content, "tell") else None
                if hasattr(voice_content, "seek"):
                    voice_content.seek(0)
                data = voice_content.read()
                if pos is not None and hasattr(voice_content, "seek"):
                    voice_content.seek(pos)
                return data if isinstance(data, (bytes, bytearray)) else b""
            except Exception:
                return b""
        return b""
    
    async def _transcribe_local(self, voice_content: bytes) -> str | None:
        """Transcribe using local faster-whisper (FREE, offline)."""
        try:
            # Load model once and cache it
            if not UnifiedAIService._whisper_model:
                logger.info(f"Loading local Whisper model: {self.local_whisper_model}")
                UnifiedAIService._whisper_model = WhisperModel(
                    self.local_whisper_model, 
                    device="cpu", 
                    compute_type="int8"
                )
                UnifiedAIService._model_loaded = True
            
            # Save to temp file (faster-whisper needs file path)
            import tempfile
            import asyncio
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def transcribe():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                    tmp.write(voice_content)
                    tmp_path = tmp.name
                
                try:
                    segments, info = UnifiedAIService._whisper_model.transcribe(
                        tmp_path,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    full_text = ""
                    for segment in segments:
                        full_text += segment.text.strip() + " "
                    
                    return full_text.strip()
                finally:
                    os.unlink(tmp_path)
            
            result = await loop.run_in_executor(None, transcribe)
            return result if result else None
            
        except Exception as e:
            logger.warning(f"Local transcription failed: {e}")
            return None
    
    async def _transcribe_huggingface(self, voice_content: bytes) -> str | None:
        """Transcribe using HuggingFace Inference API (FREE)."""
        try:
            async with httpx.AsyncClient() as client:
                files = {"file": ("voice.ogg", voice_content, "audio/ogg")}
                data = {"model": "openai/whisper-base"}
                headers = {"Authorization": f"Bearer {self.huggingface_token}"}
                
                response = await client.post(
                    "https://api-inference.huggingface.co/models/openai/whisper-base",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "").strip()
            else:
                logger.warning(f"HuggingFace Whisper error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"HuggingFace transcription failed: {e}")
            return None
    
    async def _transcribe_openai(self, voice_content: bytes) -> str | None:
        """Transcribe using OpenAI Whisper API (paid)."""
        if not self.openai_api_key:
            return None
            
        try:
            files = {"file": ("voice.ogg", voice_content, "audio/ogg")}
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data={"model": "whisper-1"},
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "").strip()
            else:
                logger.error(f"OpenAI Whisper error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI transcription error: {e}")
            return None
    
    # ==================== COMMAND PARSING ====================
    
    def parse_command(self, text: str, user_id: int = None) -> dict:
        """Parse command through VoiceAIManager to keep one NLU source of truth."""
        ctx = voice_ai.get_context(user_id) if user_id else None
        action = IntentDetector.detect(text, ctx)

        action_map = {
            Intent.CREATE_LEAD: "create",
            Intent.LIST_LEADS: "list",
            Intent.SHOW_NOTES: "notes",
            Intent.ADD_NOTE: "note",
            Intent.STATS: "stats",
            Intent.SEARCH: "search",
            Intent.SALES: "sales",
            Intent.ANALYZE_LEAD: "analyze",
            Intent.EDIT_LEAD: "edit",
            Intent.DELETE_LEAD: "delete",
            Intent.UNKNOWN: "ai_query",
        }

        lead_data = {
            "lead_id": action.entities.lead_id,
            "name": action.entities.lead_name,
            "phone": action.entities.phone,
            "email": action.entities.email,
            "source": action.entities.source,
            "domain": action.entities.domain,
            "content": action.entities.note_content,
        }
        lead_data = {k: v for k, v in lead_data.items() if v is not None}

        mapped_action = action_map.get(action.intent, "ai_query")
        if action.intent == Intent.UNKNOWN:
            mapped_action = None
        ui_hint = self._build_ui_hint(mapped_action, lead_data, action.confidence)
        missing_fields = self._detect_missing_fields(mapped_action, lead_data)

        return {
            "action": mapped_action,            "query": action.entities.search_query or text,
            "lead_data": lead_data,
            "confidence": action.confidence,
            "raw_text": text,
            "ui_hint": ui_hint,
            "missing_fields": missing_fields,
        }

    def _detect_missing_fields(self, action: Optional[str], lead_data: dict) -> list[str]:
        if action == "create":
            if not (lead_data.get("name") or lead_data.get("phone") or lead_data.get("email")):
                return ["name_or_phone_or_email"]
            return []
        if action == "analyze":
            return [] if lead_data.get("lead_id") else ["lead_id"]
        if action == "note":
            missing = []
            if not lead_data.get("lead_id"):
                missing.append("lead_id")
            if not lead_data.get("content"):
                missing.append("content")
            return missing
        return []

    def _build_ui_hint(self, action: str, lead_data: dict, confidence: float) -> dict:
        """Decide if interactive buttons should be rendered for AI response."""
        risk_actions = {"delete", "edit"}

        if confidence < 0.45:
            return {
                "kind": "clarify",
                "show_buttons": False,
                "reason": "low_confidence",
            }

        if action == "create":
            has_min_data = bool(lead_data.get("name") or lead_data.get("phone") or lead_data.get("email"))
            if has_min_data and confidence >= 0.5:
                return {
                    "kind": "lead_draft_actions",
                    "show_buttons": True,
                    "reason": "create_with_min_data",
                }
            return {
                "kind": "clarify",
                "show_buttons": False,
                "reason": "create_incomplete_or_low_confidence",
            }

        if action == "analyze":
            if lead_data.get("lead_id"):
                return {
                    "kind": "analysis_next_steps",
                    "show_buttons": True,
                    "reason": "analysis_with_target",
                }
            return {
                "kind": "clarify",
                "show_buttons": False,
                "reason": "analysis_without_target",
            }

        if action in risk_actions:
            return {
                "kind": "confirm_required",
                "show_buttons": True,
                "reason": "risk_action",
            }

        if action in {"list", "stats", "search", "notes", "sales"}:
            return {
                "kind": "none",
                "show_buttons": False,
                "reason": "informational",
            }

        return {
            "kind": "none",
            "show_buttons": False,
            "reason": "fallback",
        }
    
    def _extract_lead_id(self, text: str) -> int | None:
        """Extract lead ID from text."""
        patterns = [
            r'лід\s*#?(\d+)',
            r'lead\s*#?(\d+)',
            r'до\s*лід[ау]\s*#?(\d+)',
            r'для\s*лід[ау]\s*#?(\d+)',
            r'#(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        return None
    
    def _parse_lead_data(self, text: str) -> dict:
        """Parse lead data from text/voice input."""
        text_lower = text.lower()
        
        result = {
            "name": None,
            "phone": None,
            "email": None,
            "source": "MANUAL",
            "domain": None
        }
        
        # Extract name
        name_patterns = [
            r'лід[ау]?[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)?)',
            r'додай\s+(?:нового\s+)?ліда[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)?)',
            r'([А-Я][а-яёЇїІіЄє]+(?:\s+[А-Я][а-яёЇїІіЄє]+)(?:\s+[А-Я][а-яёЇїІіЄє]+)?)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and not any(kw in name.lower() for kw in ['додай', 'ліда', 'номер', 'тел', 'email']):
                    result["name"] = name
                    break
        
        # Extract phone
        phone_patterns = [r'\+?380\d{9}', r'\+?\d{10,12}', r'\d{3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}']
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                result["phone"] = match.group()
                break
        
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            result["email"] = email_match.group()
        
        # Extract source
        if "сканер" in text_lower or "scanner" in text_lower:
            result["source"] = "SCANNER"
        elif "партнер" in text_lower or "partner" in text_lower:
            result["source"] = "PARTNER"
        
        # Extract domain
        if "перший" in text_lower or "first" in text_lower:
            result["domain"] = "FIRST"
        elif "другий" in text_lower or "second" in text_lower:
            result["domain"] = "SECOND"
        elif "третій" in text_lower or "third" in text_lower:
            result["domain"] = "THIRD"
        
        return result
    
    # ==================== AI QUERY PROCESSING ====================
    
    async def process_query(self, query: str, leads: list) -> str:
        """Process natural language query about leads using AI."""
        if not self.openai_api_key:
            return self._format_fallback_response(self._simple_query_response(query, leads))
        
        try:
            context = self._prepare_context(leads)
            system_prompt = self._build_prompt()
            user_prompt = f"Query: {query}\n\nLeads Data:\n{context}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.3
                    },
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                # If API key is invalid, disable OpenAI for current process
                # and immediately switch to local rule-based fallback.
                if response.status_code == 401:
                    logger.warning("OpenAI key is invalid (401). Switching to local fallback responses.")
                    self.openai_api_key = None
                    self._openai_disabled_reason = "invalid_api_key"
                return self._format_fallback_response(self._simple_query_response(query, leads))
                
        except Exception as e:
            logger.error(f"AI query error: {e}")
            return self._format_fallback_response(self._simple_query_response(query, leads))

    def _format_fallback_response(self, content: str) -> str:
        """Attach a small notice when OpenAI is unavailable and fallback rules are used."""
        if self._openai_disabled_reason == "invalid_api_key":
            notice = "⚠️ <i>OpenAI тимчасово недоступний (невірний API ключ). Працюю в локальному режимі.</i>\n\n"
            return f"{notice}{content}"
        return content
    
    def _simple_query_response(self, query: str, leads: list) -> str:
        """Simple rule-based response when AI is not available."""
        query_lower = query.lower()
        
        # Hot leads
        if "гаряч" in query_lower or "hot" in query_lower or "best" in query_lower:
            hot_leads = [l for l in leads if l.get("ai_score", 0) >= 0.6]
            if hot_leads:
                response = "🔥 <b>Гарячі ліди:</b>\n\n"
                for lead in hot_leads[:5]:
                    response += f"• #{lead.get('id')}: {lead.get('full_name')} (score: {lead.get('ai_score', 0):.0%})\n"
                return response
            return "Гарячі ліди не знайдені."
        
        # Count by source
        if "сканер" in query_lower or "scanner" in query_lower:
            count = len([l for l in leads if l.get("source") == "SCANNER"])
            return f"Лідів зі сканера: <b>{count}</b>"
        
        if "партнер" in query_lower or "partner" in query_lower:
            count = len([l for l in leads if l.get("source") == "PARTNER"])
            return f"Лідів від партнерів: <b>{count}</b>"
        
        # Stage counts
        for stage in ["new", "contacted", "qualified", "transferred", "lost"]:
            if stage in query_lower:
                count = len([l for l in leads if l.get("stage", "").lower() == stage.upper()])
                return f"Лідів в стадії {stage}: <b>{count}</b>"
        
        # Default
        return f"Знайдено <b>{len(leads)}</b> лідів. Уточніть ваш запит."
    
    def _prepare_context(self, leads: list) -> str:
        if not leads:
            return "No leads in database."
        
        sample = leads[:30]
        summaries = []
        for lead in sample:
            s = f"ID:{lead.get('id')} | {lead.get('full_name', 'N/A')} | {lead.get('source')} | {lead.get('stage')} | {lead.get('business_domain', '-')}"
            if lead.get('ai_score'):
                s += f" | Score:{lead.get('ai_score', 0):.0%}"
            summaries.append(s)
        return "\n".join(summaries)
    
    def _build_prompt(self) -> str:
        return """Ти — CRM-асистент для системи управління лідрами.

Доступні дані: id, full_name, source (SCANNER/PARTNER/MANUAL), stage (NEW/CONTACTED/QUALIFIED/TRANSFERRED/LOST), business_domain (FIRST/SECOND/THIRD), ai_score (0.0-1.0).

ВІДПОВІДАЙ УКРАЇНСЬКОЮ!
Будь коротким і корисним.
Використовуй форматування HTML (<b>, <i>, •)."""
    
    # ==================== NOTE CATEGORIZATION ====================
    
    async def categorize_note(self, note_content: str) -> str:
        """Categorize a note."""
        if not self.openai_api_key:
            return self._simple_categorize(note_content)
        
        try:
            prompt = f"""Визнач категорію нотатки одним словом: contact, email, meeting, general, problem або success.

Текст: {note_content[:200]}"""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20},
                    timeout=10.0
                )
            
            if response.status_code == 200:
                cat = response.json()["choices"][0]["message"]["content"].strip().lower()
                valid = ["contact", "email", "meeting", "general", "problem", "success"]
                return cat if cat in valid else "general"
        except:
            pass
        
        return self._simple_categorize(note_content)
    
    def _simple_categorize(self, text: str) -> str:
        text_lower = text.lower()
        keywords = {
            "problem": ["проблем", "біль", "скарг", "погано", "issue"],
            "success": ["успіх", "відмінно", "добре", "виграш", "угод"],
            "contact": ["дзвін", "телефон", "розмов", "call"],
            "email": ["email", "лист", "пошта"],
            "meeting": ["зустріч", "мітинг", "нарада"]
        }
        for cat, kws in keywords.items():
            if any(k in text_lower for k in kws):
                return cat
        return "general"


# Singleton
unified_ai = UnifiedAIService()
