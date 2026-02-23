"""
AI Assistant service for natural language queries about leads.
Can search, analyze, and provide insights about the lead database.
"""
import os
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class AIAssistant:
    """AI Assistant for lead database queries."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.api_base_url = "http://localhost:8000"
    
    async def process_query(self, query: str, leads: list) -> str:
        """Process natural language query about leads."""
        if not self.openai_api_key:
            return "⚠️ AI Assistant requires OpenAI API key configuration."
        
        try:
            # Prepare context from leads
            context = self._prepare_context(leads)
            
            # Build prompt
            system_prompt = self._build_prompt()
            user_prompt = f"Query: {query}\n\nLeads Data:\n{context}"
            
            # Call OpenAI API
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
                return f"⚠️ Error processing query: {response.status_code}"
                
        except Exception as e:
            logger.error(f"AI Assistant error: {e}")
            return f"⚠️ Error: {str(e)[:100]}"
    
    def _prepare_context(self, leads: list) -> str:
        """Prepare leads data as context for AI."""
        if not leads:
            return "No leads in database."
        
        # Limit to last 50 leads for context
        sample_leads = leads[:50]
        
        lead_summaries = []
        for lead in sample_leads:
            summary = f"ID: {lead.get('id')}, "
            summary += f"Name: {lead.get('full_name', 'N/A')}, "
            summary += f"Source: {lead.get('source', 'N/A')}, "
            summary += f"Stage: {lead.get('stage', 'N/A')}, "
            summary += f"Domain: {lead.get('business_domain', 'N/A')}, "
            summary += f"Score: {lead.get('ai_score', 'N/A')}"
            lead_summaries.append(summary)
        
        return "\n".join(lead_summaries)
    
    def _build_prompt(self) -> str:
        """Build system prompt for AI assistant."""
        return """Ти — CRM-асистент для системи управління лідрами.

Ти маєш доступ до бази даних лідів. Користувачі можуть запитувати про лідів природною мовою.

Доступні поля лідів:
- id: ID ліда
- full_name: Ім'я контакту
- source: SCANNER, PARTNER або MANUAL
- stage: NEW, CONTACTED, QUALIFIED, TRANSFERRED або LOST
- business_domain: FIRST, SECOND або THIRD
- ai_score: 0.0-1.0 ймовірність успішної угоди
- created_at: Дата створення

Приклади запитів, на які ти можеш відповісти:
- "Покажи гарячі ліди" (ai_score >= 0.6)
- "Скільки лідів зі сканера?" (source=SCANNER)
- "Хто найкращий кандидат?" (найвищий ai_score)
- "Ліди в кваліфікованій стадії"
- "Статистика по доменах"
- "Які ліди потребують уваги?"

ВІДПОВІДАЙ УКРАЇНСЬКОЮ МОВОЮ!
Надавай короткі, корисні відповіді.
Використовуй буллети для списків.
Форматуй числа чітко.
Якщо потрібно шукати — скажи користувачеві, що саме ти шукаєш."""

    async def categorize_note(self, note_content: str) -> str:
        """Categorize a note using AI.
        
        Categories:
        - contact: phone calls, communication
        - email: email correspondence
        - meeting: meetings, discussions
        - general: general notes
        - problem: pain points, issues
        - success: successes, achievements
        """
        if not self.openai_api_key:
            # Fallback to simple keyword-based categorization
            return self._simple_categorize(note_content)
        
        try:
            prompt = f"""Визнач КАТЕГОРІЮ цієї нотатки. ВІДПОВІДАЙ ОДНИМ СЛОВОМ!

Доступні категорії:
- contact: дзвінки, телефонна комунікація
- email: email-листування, листи
- meeting: зустрічі, відеодзвінки, наради
- general: загальна інформація, нагадування
- problem: проблеми, скарги, біль, виклики
- success: успіхи, досягнення, позитивні результати

Текст нотатки: {note_content}

Відповідь українською мовою!"""

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
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 20,
                        "temperature": 0.1
                    },
                    timeout=10.0
                )
            
            if response.status_code == 200:
                result = response.json()
                category = result["choices"][0]["message"]["content"].strip().lower()
                
                # Validate category
                valid_categories = ["contact", "email", "meeting", "general", "problem", "success"]
                if category in valid_categories:
                    return category
                return "general"
            else:
                return self._simple_categorize(note_content)
                
        except Exception as e:
            logger.error(f"Note categorization error: {e}")
            return self._simple_categorize(note_content)
    
    def _simple_categorize(self, text: str) -> str:
        """Simple keyword-based note categorization."""
        text_lower = text.lower()
        
        # Problem keywords
        problem_keywords = ["проблем", "біль", "скарг", "погано", "не працює", "помилк", "issue", "problem", "complaint", "pain", "challenge"]
        if any(kw in text_lower for kw in problem_keywords):
            return "problem"
        
        # Success keywords
        success_keywords = ["успіх", "відмінно", "добре", "виграш", "закрили", "угод", "success", "won", "closed", "great", "excellent", "achievement"]
        if any(kw in text_lower for kw in success_keywords):
            return "success"
        
        # Contact keywords
        contact_keywords = ["дзвін", "телефон", "розмов", "передзвон", "call", "phone", "voicemail", "telegram", "message"]
        if any(kw in text_lower for kw in contact_keywords):
            return "contact"
        
        # Email keywords
        email_keywords = ["email", "лист", "пошта", "e-mail", "gmail", "mail"]
        if any(kw in text_lower for kw in email_keywords):
            return "email"
        
        # Meeting keywords
        meeting_keywords = ["зустріч", "мітинг", "дзвінок", "конференц", "zoom", "meeting", "call scheduled", "call scheduled"]
        if any(kw in text_lower for kw in meeting_keywords):
            return "meeting"
        
        return "general"
    
    async def split_long_note(self, note_content: str) -> list:
        """Split a long note into smaller categorized notes using AI.
        
        Returns a list of note objects with 'content' and 'category' keys.
        """
        if len(note_content) < 500:
            # No need to split
            category = await self.categorize_note(note_content)
            return [{"content": note_content, "category": category}]
        
        if not self.openai_api_key:
            # Simple fallback split
            words = note_content.split()
            half = len(words) // 2
            return [
                {"content": " ".join(words[:half]), "category": "general"},
                {"content": " ".join(words[half:]), "category": "general"}
            ]
        
        try:
            prompt = f"""Розділи цю довгу нотатку на менші логічні частини.
Для кожної частини визнач категорію та зміст.

Доступні категорії:
- contact: дзвінки, комунікація
- email: email-листування
- meeting: зустрічі, наради
- general: загальні нотатки
- problem: проблеми, питання
- success: успіхи, досягнення

Оригінальна нотатка:
{note_content}

Відповідай у форматі JSON масиву об'єктів:
[
  {{"category": "contact", "content": "перша частина нотатки"}},
  {{"category": "general", "content": "друга частина нотатки"}}
]

Кожна частина 200-400 символів. Максимум 5 частин."""

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
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.2
                    },
                    timeout=30.0
                )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON
                import json
                try:
                    notes = json.loads(content)
                    return notes
                except:
                    # Fallback
                    return [{"content": note_content, "category": "general"}]
            else:
                return [{"content": note_content, "category": "general"}]
                
        except Exception as e:
            logger.error(f"Note splitting error: {e}")
            return [{"content": note_content, "category": "general"}]


# Singleton instance
ai_assistant = AIAssistant()
