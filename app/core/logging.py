"""
Structured logging configuration.
"""
import logging
import sys
from typing import Any

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create formatters
    class StructuredFormatter(logging.Formatter):
        """JSON-like structured formatter for production."""
        
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Add extra fields
            if hasattr(record, "extra"):
                log_data.update(record.extra)
            
            # Add exception info
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            
            # Add request info if available
            if hasattr(record, "request_id"):
                log_data["request_id"] = record.request_id
            
            if hasattr(record, "user_id"):
                log_data["user_id"] = record.user_id
            
            return str(log_data)
    
    # Console formatter (human-readable)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Set third-party loggers to WARNING
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra fields to log records."""
    
    def __init__(self, **kwargs: Any):
        self.extra = kwargs
        self._old_factory = None
    
    def __enter__(self) -> "LogContext":
        # This is a simplified version - in production you'd use
        # contextvars for thread-safe context
        return self
    
    def __exit__(self, *args: Any) -> None:
        pass
