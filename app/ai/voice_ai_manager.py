"""
🎯 VoiceAI Manager - Unified Module for Voice and AI Assistance
==============================================================

This module combines:
- Voice transcription (free Whisper)
- Natural Language Understanding
- Context-aware conversation
- Universal action execution

Single source of truth for all voice/text AI interactions.
"""
import os
import re
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install for local offline transcription.")


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

class Intent(Enum):
    """All possible user intents."""
    CREATE_LEAD = "create_lead"
    LIST_LEADS = "list_leads"
    SHOW_LEAD = "show_lead"
    EDIT_LEAD = "edit_lead"
    DELETE_LEAD = "delete_lead"
    ADD_NOTE = "add_note"
    SHOW_NOTES = "show_notes"
    ANALYZE_LEAD = "analyze_lead"
    SEARCH = "search"
    STATS = "stats"
    SALES = "sales"
    UNKNOWN = "unknown"
    CONFIRM = "confirm"
    CANCEL = "cancel"


@dataclass
class Entities:
    """Extracted entities from user input."""
    lead_id: Optional[int] = None
    lead_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    stage: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    note_content: Optional[str] = None
    search_query: Optional[str] = None


@dataclass
class Action:
    """Represents a user action to execute."""
    intent: Intent
    entities: Entities
    confidence: float = 0.5
    requires_confirmation: bool = False
    original_text: str = ""


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    timestamp: datetime
    user_input: str
    action: Action
    bot_response: str


@dataclass
class UserContext:
    """User conversation context."""
    user_id: int
    last_lead_id: Optional[int] = None
    last_lead_name: Optional[str] = None
    last_action: Optional[str] = None
    confirmation_pending: Optional[Action] = None
    conversation_history: list = field(default_factory=list)
    state: str = "idle"  # idle, awaiting_confirmation, editing
    last_seen_at: datetime = field(default_factory=datetime.now)
    
    def add_turn(self, turn: ConversationTurn):
        """Add turn to history, keep last 10."""
        self.conversation_history.append(turn)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        self.last_seen_at = datetime.now()


# ─────────────────────────────────────────────────────────────
# Intent Detection
# ─────────────────────────────────────────────────────────────

class IntentDetector:
    """Detects user intent from text."""
    
    # Intent patterns - flexible matching
    PATTERNS = {
        Intent.CREATE_LEAD: {
            "keywords": ["лід", "ліда", "лідів"],
            "verbs": ["додай", "додати", "потрібно", "створи", "створити", "новий", "new", "add", "create"],
            "phrases": ["додай ліда", "додати ліда", "потрібно додати", "новий ліда", "new lead"]
        },
        Intent.LIST_LEADS: {
            "keywords": ["лід", "ліди", "лідів", "lead", "leads"],
            "verbs": ["покажи", "показати", "список", "show", "list", "мої", "всі"],
            "phrases": ["покажи ліди", "show leads", "мої ліди", "список лідів"]
        },
        Intent.SHOW_NOTES: {
            "keywords": ["нотатк", "нотаток", "нотатки", "заміт", "note", "notes"],
            "verbs": ["покажи", "показати", "show", "мої"],
            "phrases": ["покажи нотатки", "show notes"]
        },
        Intent.ADD_NOTE: {
            "keywords": ["нотатк", "нотатку", "заміт", "note"],
            "verbs": ["додай", "додати", "запиши", "записати", "add", "create"],
            "phrases": ["додай нотатку", "add note"]
        },
        Intent.STATS: {
            "keywords": ["статистик", "звіт", "stats", "дашборд", "dashboard"],
            "verbs": ["покажи", "show"],
            "phrases": ["статистика", "show stats"]
        },
        Intent.SEARCH: {
            "keywords": ["знайди", "пошук", "search", "find", "шукай"],
            "verbs": ["знайди", "шукай", "search", "find"],
            "phrases": ["знайди", "search"]
        },
        Intent.SALES: {
            "keywords": ["продаж", "sale", "sales", "pipeline", "воронк"],
            "verbs": ["покажи", "show"],
            "phrases": ["продажі", "sales", "pipeline"]
        },
        Intent.ANALYZE_LEAD: {
            "keywords": ["гаряч", "найкращ", "best", "hot", "top", "оцін", "score", "аналіз", "analyze"],
            "verbs": ["оціни", "проаналізуй", "analyze"],
            "phrases": ["гарячі ліди", "hot leads", "оціни ліда"]
        },
        Intent.EDIT_LEAD: {
            "keywords": ["лід", "ліда"],
            "verbs": ["редагуй", "редагувати", "зміни", "змінити", "edit", "change", "онов"],
            "phrases": ["редагуй ліда", "edit lead"]
        },
        Intent.DELETE_LEAD: {
            "keywords": ["лід", "ліда", "лідів"],
            "verbs": ["видали", "видалити", "delete", "remove"],
            "phrases": ["видали ліда", "delete lead"]
        }
    }
    
    @classmethod
    def detect(cls, text: str, current_context: Optional[UserContext] = None) -> Action:
        """Detect intent from text with context awareness."""
        text_lower = text.lower()
        
        # Try exact phrase matching first (highest confidence)
        for intent, pattern_data in cls.PATTERNS.items():
            for phrase in pattern_data.get("phrases", []):
                if phrase in text_lower:
                    entities = cls._extract_entities(text)
                    # Use context to fill missing lead_id
                    if current_context and not entities.lead_id:
                        entities.lead_id = current_context.last_lead_id
                    return Action(
                        intent=intent,
                        entities=entities,
                        confidence=0.9,
                        original_text=text
                    )
        
        # Try keyword + verb combination
        for intent, pattern_data in cls.PATTERNS.items():
            keywords = pattern_data.get("keywords", [])
            verbs = pattern_data.get("verbs", [])
            
            has_keyword = any(k in text_lower for k in keywords)
            has_verb = any(v in text_lower for v in verbs)
            
            if has_keyword and has_verb:
                entities = cls._extract_entities(text)
                if current_context and not entities.lead_id:
                    entities.lead_id = current_context.last_lead_id
                return Action(
                    intent=intent,
                    entities=entities,
                    confidence=0.8,
                    original_text=text
                )
        
        # Handle confirmation/cancel keywords
        if any(w in text_lower for w in ["так", "так", "yes", "підтверджую", "confirm", "ок", "окей"]):
            return Action(Intent.CONFIRM, Entities(), 0.95, original_text=text)
        if any(w in text_lower for w in ["ні", "no", "скасуй", "cancel", "відміна"]):
            return Action(Intent.CANCEL, Entities(), 0.95, original_text=text)
        
        # Default to unknown - will use AI fallback
        entities = cls._extract_entities(text)
        return Action(Intent.UNKNOWN, entities, 0.3, original_text=text)
    
    @staticmethod
    def _extract_entities(text: str) -> Entities:
        """Extract entities from text."""
        entities = Entities()
        text_lower = text.lower()
        
        # Extract lead ID
        lead_patterns = [
            r'лід\s*#?(\d+)',
            r'lead\s*#?(\d+)',
            r'до\s*лід[ау]\s*#?(\d+)',
            r'для\s*лід[ау]\s*#?(\d+)',
            r'#(\d+)',
        ]
        for pattern in lead_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    entities.lead_id = int(match.group(1))
                    break
                except:
                    pass
        
        # Extract phone
        phone_patterns = [r'\+?380\d{9}', r'\+?\d{10,12}']
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                entities.phone = match.group()
                break
        
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            entities.email = email_match.group()
        
        # Extract name (simple pattern)
        name_patterns = [
            r'лід[ау]?[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)',
            r'додай\s+(?:нового\s+)?ліда[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and not any(kw in name.lower() for kw in ['додай', 'ліда', 'номер']):
                    entities.lead_name = name
                    break
        
        # Extract source
        if "сканер" in text_lower or "scanner" in text_lower:
            entities.source = "SCANNER"
        elif "партнер" in text_lower or "partner" in text_lower:
            entities.source = "PARTNER"
        
        # Extract domain
        if "перший" in text_lower or "first" in text_lower:
            entities.domain = "FIRST"
        elif "другий" in text_lower or "second" in text_lower:
            entities.domain = "SECOND"
        elif "третій" in text_lower or "third" in text_lower:
            entities.domain = "THIRD"
        
        # Extract search query
        search_verbs = ["знайди", "шукай", "search", "find"]
        for verb in search_verbs:
            if verb in text_lower:
                query = text_lower.replace(verb, "", 1).strip()
                if query:
                    entities.search_query = query
                    break
        
        # Note content - everything after action verbs
        note_verbs = ["додай нотатку", "запиши", "нотатка"]
        for verb in note_verbs:
            if verb in text_lower:
                idx = text_lower.find(verb) + len(verb)
                content = text[idx:].strip()
                if content:
                    entities.note_content = content
                break
        
        return entities


# ─────────────────────────────────────────────────────────────
# VoiceAI Manager
# ─────────────────────────────────────────────────────────────

class VoiceAIManager:
    """
    🎯 Unified VoiceAI Manager
    
    Single module for handling both voice and text input.
    Provides context-aware conversation and action execution.
    
    Usage:
        manager = VoiceAIManager()
        
        # Process voice
        result = await manager.process_voice(audio_bytes, user_id)
        
        # Process text
        result = await manager.process_text("додай ліда", user_id)
    """
    
    _whisper_model = None
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.local_whisper_model = os.getenv("LOCAL_WHISPER_MODEL", "base")
        self.context_ttl_minutes = int(os.getenv("VOICE_AI_CONTEXT_TTL_MINUTES", "120"))
        
        # User contexts - in production, use Redis/DB
        self._user_contexts: dict[int, UserContext] = {}

    def assess_transcription_quality(self, text: str) -> dict:
        """Heuristic quality estimation for transcribed voice text."""
        raw = (text or "").strip()
        if not raw:
            return {
                "score": 0.0,
                "label": "LOW",
                "needs_clarification": True,
                "hints": ["Спробуйте говорити чіткіше", "Запишіть голосове в тихому місці"],
            }

        score = 1.0
        hints: list[str] = []

        words = re.findall(r"\w+", raw, flags=re.UNICODE)
        word_count = len(words)
        if len(raw) < 6:
            score -= 0.4
            hints.append("Коротке повідомлення — додайте більше контексту")
        if word_count < 2:
            score -= 0.25
            hints.append("Сформулюйте команду повним реченням")

        weird_chars = re.findall(r"[^\w\s\#\+\-\.@,:;!?іїєґІЇЄҐа-яА-Яa-zA-Z0-9]", raw)
        weird_ratio = (len(weird_chars) / max(len(raw), 1))
        if weird_ratio > 0.2:
            score -= 0.2
            hints.append("Багато шуму в тексті — перевірте мікрофон")

        if re.search(r"(.)\1{4,}", raw):
            score -= 0.15
            hints.append("Ймовірна помилка розпізнавання — повторіть команду")

        score = max(0.0, min(score, 1.0))
        if score >= 0.75:
            label = "HIGH"
        elif score >= 0.5:
            label = "MEDIUM"
        else:
            label = "LOW"

        return {
            "score": score,
            "label": label,
            "needs_clarification": score < 0.5,
            "hints": hints[:3],
        }
    
    # ─────────────────────────────────────────────────────────
    # Context Management
    # ─────────────────────────────────────────────────────────
    
    def get_context(self, user_id: int) -> UserContext:
        """Get or create user context."""
        self._cleanup_contexts()
        if user_id not in self._user_contexts:
            self._user_contexts[user_id] = UserContext(user_id=user_id)
        self._user_contexts[user_id].last_seen_at = datetime.now()
        return self._user_contexts[user_id]

    def _cleanup_contexts(self):
        """Drop stale user contexts to avoid memory leaks and stale follow-ups."""
        if not self._user_contexts:
            return
        cutoff = datetime.now() - timedelta(minutes=self.context_ttl_minutes)
        stale_ids = [uid for uid, ctx in self._user_contexts.items() if ctx.last_seen_at < cutoff]
        for uid in stale_ids:
            self._user_contexts.pop(uid, None)
    
    def update_context_lead(self, user_id: int, lead_id: int, lead_name: str = None):
        """Update last lead in context."""
        ctx = self.get_context(user_id)
        ctx.last_lead_id = lead_id
        if lead_name:
            ctx.last_lead_name = lead_name
    
    def set_confirmation(self, user_id: int, action: Action):
        """Set pending confirmation."""
        ctx = self.get_context(user_id)
        ctx.confirmation_pending = action
        ctx.state = "awaiting_confirmation"
    
    def clear_confirmation(self, user_id: int):
        """Clear pending confirmation."""
        ctx = self.get_context(user_id)
        ctx.confirmation_pending = None
        ctx.state = "idle"

    def resolve_pronoun(self, text: str, user_id: int) -> tuple[str, Optional[int], Optional[str]]:
        """Resolve references like 'його' / 'того ліда' using user context."""
        ctx = self.get_context(user_id)
        text_lower = text.lower()

        pronoun_patterns = [
            "того ліда", "того", "його", "йому", "нього", "неї", "останнього", "останньому",
            "that lead", "that one", "the previous", "him",
        ]
        if any(p in text_lower for p in pronoun_patterns) and ctx.last_lead_id:
            return text, ctx.last_lead_id, ctx.last_lead_name

        return text, None, None
    
    # ─────────────────────────────────────────────────────────
    # Voice Processing
    # ─────────────────────────────────────────────────────────
    
    async def process_voice(self, voice_content: bytes, user_id: int) -> dict:
        """
        Process voice input: transcription → understanding → action.
        
        Returns:
            dict with keys: success, text, action, response
        """
        # 1. Transcribe
        text = await self._transcribe(voice_content)
        if not text:
            return {
                "success": False,
                "text": None,
                "action": None,
                "response": "⚠️ Не вдалося розпізнати голос. Спробуйте ще раз."
            }
        
        # 2. Process as text
        return await self.process_text(text, user_id)
    
    async def _transcribe(self, voice_content: bytes) -> Optional[str]:
        """Transcribe voice to text using available services."""
        # Try local faster-whisper first
        if FASTER_WHISPER_AVAILABLE:
            result = await self._transcribe_local(voice_content)
            if result:
                logger.info("Used local Whisper transcription")
                return result
        
        # Try HuggingFace API
        if self.huggingface_token:
            result = await self._transcribe_huggingface(voice_content)
            if result:
                logger.info("Used HuggingFace transcription")
                return result
        
        # Try OpenAI Whisper
        if self.openai_api_key:
            result = await self._transcribe_openai(voice_content)
            if result:
                logger.info("Used OpenAI transcription")
                return result
        
        logger.error("No transcription service available")
        return None
    
    async def _transcribe_local(self, voice_content: bytes) -> Optional[str]:
        """Local faster-whisper transcription."""
        try:
            if not VoiceAIManager._whisper_model:
                VoiceAIManager._whisper_model = WhisperModel(
                    self.local_whisper_model, device="cpu", compute_type="int8"
                )
            
            import tempfile
            loop = asyncio.get_event_loop()
            
            def transcribe():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                    tmp.write(voice_content)
                    tmp_path = tmp.name
                
                try:
                    segments, _ = VoiceAIManager._whisper_model.transcribe(
                        tmp_path, beam_size=5, vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    return " ".join(s.text.strip() for s in segments)
                finally:
                    os.unlink(tmp_path)
            
            return await loop.run_in_executor(None, transcribe)
        except Exception as e:
            logger.warning(f"Local transcription failed: {e}")
            return None
    
    async def _transcribe_huggingface(self, voice_content: bytes) -> Optional[str]:
        """HuggingFace API transcription."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api-inference.huggingface.co/models/openai/whisper-base",
                    headers={"Authorization": f"Bearer {self.huggingface_token}"},
                    files={"file": ("voice.ogg", voice_content, "audio/ogg")},
                    data={"model": "openai/whisper-base"},
                    timeout=30.0
                )
            
            if response.status_code == 200:
                return response.json().get("text", "").strip()
        except Exception as e:
            logger.warning(f"HuggingFace transcription failed: {e}")
        return None
    
    async def _transcribe_openai(self, voice_content: bytes) -> Optional[str]:
        """OpenAI Whisper API transcription."""
        if not self.openai_api_key:
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    files={"file": ("voice.ogg", voice_content, "audio/ogg")},
                    data={"model": "whisper-1"},
                    timeout=30.0
                )
            
            if response.status_code == 200:
                return response.json().get("text", "").strip()
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
        return None
    
    # ─────────────────────────────────────────────────────────
    # Text Processing
    # ─────────────────────────────────────────────────────────
    
    async def process_text(self, text: str, user_id: int) -> dict:
        """
        Process text input: understanding → action → response.
        
        Returns:
            dict with keys: success, text, action, response, needs_confirmation
        """
        context = self.get_context(user_id)
        resolved_text, resolved_lead_id, _ = self.resolve_pronoun(text, user_id)
        
        # Handle confirmation state
        if context.state == "awaiting_confirmation" and context.confirmation_pending:
            text_lower = text.lower()
            if any(w in text_lower for w in ["так", "yes", "підтверджую", "confirm", "ок"]):
                # Execute confirmed action
                return await self._execute_action(
                    context.confirmation_pending, user_id, confirmed=True
                )
            elif any(w in text_lower for w in ["ні", "no", "скасуй", "cancel"]):
                self.clear_confirmation(user_id)
                return {
                    "success": True,
                    "text": text,
                    "action": Intent.CANCEL,
                    "response": "❌ Скасовано.",
                    "needs_confirmation": False
                }
        
        # Detect intent
        action = IntentDetector.detect(resolved_text, context)

        if resolved_lead_id and not action.entities.lead_id:
            action.entities.lead_id = resolved_lead_id
        
        # Update context
        if action.entities.lead_id:
            self.update_context_lead(user_id, action.entities.lead_id)
        
        # Determine if confirmation needed
        needs_confirmation = self._needs_confirmation(action)
        if needs_confirmation:
            self.set_confirmation(user_id, action)
            return await self._build_confirmation_message(action, user_id)
        
        # Execute action directly
        return await self._execute_action(action, user_id, confirmed=False)
    
    def _needs_confirmation(self, action: Action) -> bool:
        """Check if action requires confirmation."""
        create_actions = [Intent.CREATE_LEAD, Intent.DELETE_LEAD, Intent.EDIT_LEAD]
        
        # Confirm creation/deletion OR when data is incomplete
        if action.intent in create_actions:
            return True
        
        return False
    
    async def _build_confirmation_message(self, action: Action, user_id: int) -> dict:
        """Build confirmation message for action."""
        entities = action.entities
        
        if action.intent == Intent.CREATE_LEAD:
            response = (
                "📋 <b>ПІДТВЕРДЖЕННЯ</b>\n\n"
                f"Створити ліда?\n\n"
                f"👤 <b>Ім'я:</b> {entities.lead_name or '—'}\n"
                f"📞 <b>Телефон:</b> {entities.phone or '—'}\n"
                f"📧 <b>Email:</b> {entities.email or '—'}\n"
                f"📡 <b>Джерело:</b> {entities.source or 'MANUAL'}\n\n"
                "<i>Напишіть 'так' для підтвердження або 'ні' для скасування.</i>"
            )
        elif action.intent == Intent.DELETE_LEAD:
            response = (
                f"⚠️ <b>ВИДАЛЕННЯ ЛІДА #{entities.lead_id}</b>\n\n"
                "Цю дію неможливо відновити!\n\n"
                "<i>Напишіть 'так' для підтвердження або 'ні' для скасування.</i>"
            )
        elif action.intent == Intent.EDIT_LEAD:
            response = (
                f"✏️ <b>ПІДТВЕРДЖЕННЯ ОНОВЛЕННЯ</b>\n\n"
                f"Оновити ліда #{entities.lead_id or '—'}?\n\n"
                "<i>Напишіть 'так' для підтвердження або 'ні' для скасування.</i>"
            )
        else:
            response = "Підтвердіть вашу дію: так/ні"
        
        return {
            "success": True,
            "text": action.original_text,
            "action": action,
            "response": response,
            "needs_confirmation": True
        }
    
    async def _execute_action(self, action: Action, user_id: int, confirmed: bool) -> dict:
        """Execute the action and build response."""
        context = self.get_context(user_id)
        
        # Clear confirmation state after execution
        if confirmed:
            self.clear_confirmation(user_id)
        
        # For unknown intent, use AI fallback
        if action.intent == Intent.UNKNOWN:
            return await self._ai_fallback(action.original_text, user_id)
        
        # Build response based on intent
        response = await self._build_action_response(action, user_id, confirmed)
        
        # Update context
        if action.entities.lead_id:
            self.update_context_lead(user_id, action.entities.lead_id)
        
        # Add to history
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=action.original_text,
            action=action,
            bot_response=response.get("text", "")
        )
        context.add_turn(turn)
        
        return {
            "success": True,
            "text": action.original_text,
            "action": action,
            "response": response.get("text", ""),
            "response_type": response.get("type", "text"),
            "keyboard": response.get("keyboard"),
            "followup_hint": response.get("followup_hint"),
            "suggestions": response.get("suggestions", []),
            "needs_confirmation": False
        }
    
    async def _build_action_response(self, action: Action, user_id: int, confirmed: bool) -> dict:
        """Build response message for specific action."""
        context = self.get_context(user_id)
        
        if action.intent == Intent.CREATE_LEAD:
            if confirmed:
                # Would call API to create lead
                return {
                    "type": "lead_created",
                    "text": f"✅ <b>Лід створений!</b>\n\n"
                            f"Ім'я: {action.entities.lead_name or '—'}\n"
                            f"Телефон: {action.entities.phone or '—'}"
                }
            else:
                return {
                    "type": "confirmation_needed",
                    "text": "Підтвердіть створення ліда: так/ні"
                }
        
        elif action.intent == Intent.LIST_LEADS:
            return {
                "type": "leads_list",
                "text": "📋 <b>Ваші ліди:</b>\n\nПоказую список...",
                "suggestions": ["покажи нотатки", "знайди гарячі", "статистика"]
            }
        
        elif action.intent == Intent.STATS:
            return {
                "type": "stats",
                "text": "📊 <b>Статистика:</b>\n\nЗавантажую...",
                "suggestions": ["гарячі ліди", "продажі", "знайди кваліфіковані"]
            }
        
        elif action.intent == Intent.ANALYZE_LEAD:
            lead_id = action.entities.lead_id or context.last_lead_id
            if lead_id:
                return {
                    "type": "analysis",
                    "text": f"🤖 <b>Аналіз ліда #{lead_id}</b>\n\nЗавантажую...",
                    "followup_hint": "Після аналізу можу запропонувати наступний крок: nurture або transfer."
                }
            return {"type": "error", "text": "Вкажіть ID ліда для аналізу."}
        
        elif action.intent == Intent.SEARCH:
            query = action.entities.search_query
            return {
                "type": "search_results",
                "text": f"🔍 <b>Результати пошуку:</b> {query}\n\nШукаю..."
            }
        
        elif action.intent == Intent.SHOW_NOTES:
            return {
                "type": "notes_list",
                "text": "📝 <b>Нотатки:</b>\n\nПоказую..."
            }
        
        elif action.intent == Intent.ADD_NOTE:
            lead_id = action.entities.lead_id or context.last_lead_id
            content = action.entities.note_content
            if lead_id and content:
                return {
                    "type": "note_added",
                    "text": f"📝 Нотатка для ліда #{lead_id}:\n{content}"
                }
            return {"type": "error", "text": "Вкажіть ліда та текст нотатки."}
        
        elif action.intent == Intent.SALES:
            return {
                "type": "sales_pipeline",
                "text": "💰 <b>Продажі:</b>\n\nПоказую воронку..."
            }
        
        # Default
        return {
            "type": "text",
            "text": f"Дію '{action.intent.value}' виконано."
        }
    
    async def _ai_fallback(self, text: str, user_id: int) -> dict:
        """Use AI for complex queries."""
        if not self.openai_api_key:
            return {
                "success": True,
                "text": text,
                "action": Intent.UNKNOWN,
                "response": "Не зовсім зрозумів запит. Уточніть: що саме зробити з лідом?",
                "followup_hint": "Наприклад: 'покажи ліди', 'додай ліда', 'додай нотатку до ліда #12'.",
                "needs_confirmation": False
            }
        
        try:
            # Fetch leads for context
            leads_data = await self._fetch_leads(user_id)
            
            system_prompt = """Ти — CRM-асистент. Відповідай українською мовою.
Доступні дані: id, full_name, source, stage, business_domain, ai_score."""
            
            user_prompt = f"Запит: {text}\n\nДані лідів:\n{leads_data}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 300,
                        "temperature": 0.3
                    },
                    timeout=30.0
                )
            
            if response.status_code == 200:
                ai_response = response.json()["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "text": text,
                    "action": Intent.UNKNOWN,
                    "response": ai_response,
                    "needs_confirmation": False
                }
        except Exception as e:
            logger.error(f"AI fallback error: {e}")
        
        return {
            "success": True,
            "text": text,
            "action": Intent.UNKNOWN,
            "response": "Не вдалося обробити запит. Спробуйте ще раз.",
            "needs_confirmation": False
        }
    
    async def _fetch_leads(self, user_id: int) -> str:
        """Fetch leads as text context for AI."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "http://localhost:8000/api/v1/leads",
                    timeout=10.0
                )
                if resp.status_code == 200:
                    leads = resp.json().get("items", [])[:20]
                    return "\n".join(
                        f"ID:{l.get('id')} | {l.get('full_name')} | {l.get('stage')}"
                        for l in leads
                    )
        except:
            pass
        return "Дані недоступні."


# Import asyncio for executor
import asyncio

# Singleton instance
voice_ai = VoiceAIManager()
