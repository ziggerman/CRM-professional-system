"""
Request logging middleware.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                }
            )
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors and returning proper responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception(
                f"Unhandled error: {request.method} {request.url.path}",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "error": str(e),
                }
            )
            
            # Return 500 error
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if hasattr(request.app.state, "debug") and request.app.state.debug else "An error occurred",
                }
            )
