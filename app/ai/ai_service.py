"""
AI Service — advisory layer only.

Contract:
  - Receives plain dict with lead features
  - Returns AIAnalysisResult (score + recommendation + reason)
  - NEVER modifies lead state
  - NEVER makes business decisions
  - Score is a SIGNAL, not a gate — gates live in TransferService
  
Caching:
  - Results are cached in Redis with TTL from settings.AI_CACHE_TTL
  - Cache key is based on lead features hash
"""
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis
from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.lead import AIAnalysisResult
from app.ai.prompts import LEAD_ANALYSIS_SYSTEM_PROMPT, build_lead_analysis_prompt
from app.models.lead import Lead

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Raised when AI service fails."""
    pass


class AIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis connection established for AI caching")
            except Exception as e:
                logger.warning(f"Redis unavailable, caching disabled: {e}")
                self._redis = None
        return self._redis

    def _get_cache_key(self, lead: Lead) -> str:
        """Generate cache key based on lead features."""
        features = self._build_features(lead)
        features_json = json.dumps(features, sort_keys=True)
        hash_value = hashlib.sha256(features_json.encode()).hexdigest()[:16]
        return f"ai:lead:analysis:{hash_value}"

    async def _get_cached_result(self, lead: Lead) -> Optional[AIAnalysisResult]:
        """Get cached analysis result from Redis."""
        redis_client = await self._get_redis()
        if not redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(lead)
            cached = await redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for lead {lead.id}")
                data = json.loads(cached)
                return AIAnalysisResult(
                    score=float(data["score"]),
                    recommendation=str(data["recommendation"]),
                    reason=str(data["reason"]),
                )
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    async def _set_cached_result(self, lead: Lead, result: AIAnalysisResult) -> None:
        """Cache analysis result in Redis."""
        redis_client = await self._get_redis()
        if not redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(lead)
            data = {
                "score": result.score,
                "recommendation": result.recommendation,
                "reason": result.reason,
            }
            await redis_client.setex(
                cache_key,
                settings.AI_CACHE_TTL,
                json.dumps(data)
            )
            logger.info(f"Cached analysis for lead {lead.id} (TTL: {settings.AI_CACHE_TTL}s)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _build_features(self, lead: Lead) -> dict:
        """
        Extract only the features we expose to AI.
        Explicit about what AI sees — no leaking of internal IDs or raw DB state.
        """
        # Handle timezone-aware/naive datetime
        created = lead.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_since_created = (datetime.now(timezone.utc) - created).days
        
        return {
            "source": lead.source.value,
            "stage": lead.stage.value,
            "message_count": lead.message_count,
            "has_business_domain": lead.business_domain is not None,
            "business_domain": lead.business_domain.value if lead.business_domain else None,
            "days_since_created": days_since_created,
        }

    async def analyze_lead(self, lead: Lead) -> AIAnalysisResult:
        """
        Call LLM and return structured analysis.
        Uses Redis caching to avoid redundant API calls.
        Raises AIServiceError on failure so caller can handle gracefully.
        """
        # Check cache first
        cached_result = await self._get_cached_result(lead)
        if cached_result:
            return cached_result
        
        # Cache miss - call AI
        features = self._build_features(lead)
        prompt = build_lead_analysis_prompt(features)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LEAD_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temp for consistent scoring
                max_tokens=256,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"OpenAI API error for lead {lead.id}: {e}")
            raise AIServiceError(f"AI service unavailable: {e}") from e

        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            result = AIAnalysisResult(
                score=float(data["score"]),
                recommendation=str(data["recommendation"]),
                reason=str(data["reason"]),
            )
            
            # Cache the result
            await self._set_cached_result(lead, result)
            
            return result
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"AI response parse error: {raw!r} — {e}")
            raise AIServiceError(f"Invalid AI response format: {e}") from e
