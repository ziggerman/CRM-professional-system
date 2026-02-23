"""
Voice processing service using OpenAI Whisper API.
Handles voice message recognition and command parsing.
"""
import os
import re
import logging
import httpx

logger = logging.getLogger(__name__)


class VoiceService:
    """Service for processing voice messages."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    async def transcribe_voice(self, voice_content: bytes) -> str | None:
        """Transcribe voice audio to text using Whisper API."""
        if not self.openai_api_key:
            logger.error("OpenAI API key not configured")
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
                logger.error(f"Whisper API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            return None
    
    def parse_command(self, text: str) -> dict:
        """Parse voice text into structured command."""
        text_lower = text.lower()
        
        result = {
            "action": None,
            "query": None,
            "lead_data": {}
        }
        
        # Command patterns
        create_patterns = [
            "додай ліда", "додай новий ліда", "створи ліда", "новий ліда",
            "add lead", "create lead", "new lead"
        ]
        
        search_patterns = [
            "знайди", "пошук", "покажи", "search", "show", "find"
        ]
        
        stats_patterns = [
            "статистика", "статисти", "stats", "звіти"
        ]
        
        # Notes patterns
        notes_patterns = [
            "нотатк", "замітк", "записк", "note", "notes", "записа",
            "додай нотатк", "додай замітк", "додай note"
        ]
        
        show_notes_patterns = [
            "покажи нотатк", "покажи замітк", "show notes", "мої нотатк"
        ]
        
        # Check for notes command
        if any(p in text_lower for p in show_notes_patterns):
            result["action"] = "notes"
        
        # Check for add note command
        elif any(p in text_lower for p in notes_patterns):
            # Try to extract lead ID
            lead_id = self._extract_lead_id(text)
            result["action"] = "note"
            result["lead_data"] = {
                "lead_id": lead_id,
                "content": text
            }
        
        # Check for create command
        elif any(p in text_lower for p in create_patterns):
            result["action"] = "create"
            result["lead_data"] = self._parse_lead_data(text)
        
        # Check for search command
        elif any(p in text_lower for p in search_patterns):
            result["action"] = "search"
            # Extract search query
            query = text
            for word in ["знайди", "пошук", "покажи", "search", "show", "find"]:
                query = query.replace(word, "", 1).strip()
            result["query"] = query
        
        # Check for stats command
        elif any(p in text_lower for p in stats_patterns):
            result["action"] = "stats"
        
        return result
    
    def _extract_lead_id(self, text: str) -> int | None:
        """Extract lead ID from text if mentioned."""
        # Look for patterns like "лід 5", "lead 5", "до ліда 5"
        import re
        patterns = [
            r'лід\s*#?(\d+)',
            r'lead\s*#?(\d+)',
            r'до\s*лід[ау]\s*#?(\d+)',
            r'для\s*лід[ау]\s*#?(\d+)',
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
        """Parse lead data from voice text."""
        text_lower = text.lower()
        
        result = {
            "name": None,
            "phone": None,
            "email": None,
            "source": "MANUAL",
            "domain": None
        }
        
        # Extract name - look for patterns like "ліда [Name]" or "[Name], номер"
        # Common patterns in Ukrainian:
        # "додай ліда Іван" -> name: Іван
        # "додай ліда Петренко Олександр" -> name: Петренко Олександр  
        # "Стоцький Микола Володимирович" - full name in voice
        
        name_patterns = [
            # After "лід" or "ліда" command (with optional period or comma)
            r'лід[ау]?[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)?)',
            # After "додай" command - various forms
            r'додай\s+(?:нового\s+)?ліда[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)?)',
            # After "створи" command
            r'створи\s+лід[ау]?[.,]?\s*([А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+(?:\s+[А-Яа-яёЇїІіЄєA-Za-z]+)?)?)',
            # Full name pattern (2-3 words capitalized) anywhere in text
            r'([А-Я][а-яёЇїІіЄє]+(?:\s+[А-Я][а-яёЇїІіЄє]+)(?:\s+[А-Я][а-яёЇїІіЄє]+)?)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                # Validate it's a name (not too short, not a keyword)
                if len(potential_name) > 2 and not any(kw in potential_name.lower() for kw in ['додай', 'ліда', 'номер', 'тел', 'email', 'партнер', 'сканер']):
                    result["name"] = potential_name
                    break
        
        # Extract phone
        phone_patterns = [
            r'\+?380\d{9}',
            r'\+?\d{10,12}',
            r'\d{3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}',
        ]
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


# Singleton instance
voice_service = VoiceService()
