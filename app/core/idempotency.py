"""Idempotency helpers for sensitive POST operations."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_memory_store: dict[str, tuple[float, dict[str, Any]]] = {}
_memory_lock = asyncio.Lock()


class IdempotencyStore:
    """Stores idempotent operation results in Redis, with in-memory fallback."""

    def __init__(self) -> None:
        self._redis = None
        self._redis_unavailable = False

    async def _get_redis(self):
        if self._redis_unavailable:
            return None
        if self._redis is None:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
                await self._redis.ping()
            except Exception as exc:
                logger.warning("Idempotency Redis unavailable, using in-memory store: %s", exc)
                self._redis_unavailable = True
                self._redis = None
        return self._redis

    async def get(self, key: str) -> dict[str, Any] | None:
        redis_conn = await self._get_redis()
        if redis_conn:
            raw = await redis_conn.get(key)
            if not raw:
                return None
            return json.loads(raw)

        async with _memory_lock:
            item = _memory_store.get(key)
            if not item:
                return None
            expires_at, payload = item
            if expires_at <= time.time():
                _memory_store.pop(key, None)
                return None
            return payload

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 600) -> None:
        redis_conn = await self._get_redis()
        if redis_conn:
            await redis_conn.setex(key, ttl_seconds, json.dumps(value))
            return

        async with _memory_lock:
            _memory_store[key] = (time.time() + ttl_seconds, value)


idempotency_store = IdempotencyStore()
