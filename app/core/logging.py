"""
Structured logging setup with fallback to standard library.
Step 2.1 — Observability
"""
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

from app.core.config import settings


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for production logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stdout_handler)

    file_handler = RotatingFileHandler(
        filename="api.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handlers.append(file_handler)
    return handlers


def setup_logging() -> None:
    """Configure logging. Uses structlog if available, otherwise standard logging."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if HAS_STRUCTLOG:
        shared_processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if settings.DEBUG:
            renderer = structlog.dev.ConsoleRenderer()
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
            wrapper_class=structlog.stdlib.BoundLogger,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        for handler in _build_handlers():
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    else:
        # Fallback to standard logging with JSON in production
        for handler in _build_handlers():
            if settings.DEBUG:
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            else:
                handler.setFormatter(JsonFormatter())
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        logging.info("Structlog not found, falling back to standard logging.")

    # Set third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str):
    """Get a logger instance."""
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)


class LogContext:
    """Compatibility context manager."""
    def __init__(self, **kwargs: Any):
        self.extra = kwargs
    def __enter__(self):
        return self
    def __exit__(self, *args: Any):
        pass
