"""
Rate limiting middleware using Redis.
"""
import time
import logging
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from jose import jwt
from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Rate limit configuration."""
    # Default limits
    DEFAULT_REQUESTS = 100  # requests per window
    DEFAULT_WINDOW = 60  # seconds
    
    # Endpoints-specific limits
    ENDPOINT_LIMITS = {
        "/api/v1/leads": {"requests": 50, "window": 60},
        "/api/v1/leads/analyze": {"requests": 10, "window": 60},
        "/api/v1/sales": {"requests": 50, "window": 60},
    }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app, redis_url: Optional[str] = None):
        super().__init__(app)
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        self.config = RateLimitConfig()
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis.ping()
            except Exception as e:
                logger.warning(f"Redis unavailable, rate limiting disabled: {e}")
                self._redis = False  # Mark as unavailable
        return self._redis if self._redis else None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Try to get user ID from JWT token."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return str(payload.get("sub"))
        except:
            return None
    
    def _get_limit(self, path: str) -> tuple[int, int]:
        """Get rate limit for endpoint."""
        # Check for specific endpoint limit
        for endpoint, limit in self.config.ENDPOINT_LIMITS.items():
            if path.startswith(endpoint):
                return limit["requests"], limit["window"]
        
        # Default limit
        return self.config.DEFAULT_REQUESTS, self.config.DEFAULT_WINDOW
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Skip if Redis unavailable
        redis = await self._get_redis()
        if not redis:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        requests_limit, window = self._get_limit(request.url.path)
        
        # Create rate limit key: user-based preferred, fallback to IP
        if user_id:
            key = f"rate_limit:user:{user_id}:{request.url.path}"
        else:
            key = f"rate_limit:ip:{client_ip}:{request.url.path}"
        
        try:
            # Get current count
            current = await redis.get(key)
            current_count = int(current) if current else 0
            
            if current_count >= requests_limit:
                # Get TTL for response header
                ttl = await redis.ttl(key)
                if ttl < 0:
                    ttl = window
                
                logger.warning(f"Rate limit exceeded for {client_ip} on {request.url.path}")
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": ttl,
                        "limit": requests_limit,
                        "window": window,
                    },
                    headers={
                        "X-RateLimit-Limit": str(requests_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + ttl),
                        "Retry-After": str(ttl),
                    }
                )
            
            # Increment counter
            pipe = redis.pipeline()
            pipe.incr(key)
            if current_count == 0:
                pipe.expire(key, window)
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            # Allow request if rate limiting fails
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(requests_limit)
        
        try:
            current = await redis.get(key)
            remaining = max(0, requests_limit - (int(current) if current else 0))
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        except:
            pass
        
        return response
