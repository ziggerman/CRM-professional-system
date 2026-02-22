"""
Structured logging setup with fallback to standard library.
Step 2.1 — Observability
"""
import logging
import sys
from typing import Any

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

from app.core.config import settings


def setup_logging() -> None:
    """Configure logging. Uses structlog if available, otherwise standard logging."""
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

        handler = logging.StreamHandler(sys.stdout)
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    else:
        # Fallback to standard logging
        logging.basicConfig(
            level=logging.DEBUG if settings.DEBUG else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout
        )
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
