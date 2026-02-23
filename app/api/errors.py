"""Unified API error response helpers."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def build_error_payload(
    *,
    code: str,
    message: str,
    detail: Any = None,
    context: dict[str, Any] | None = None,
    request: Request | None = None,
) -> dict[str, Any]:
    payload_context = context.copy() if context else {}
    if request is not None:
        payload_context.setdefault("request_id", getattr(request.state, "request_id", None))
        payload_context.setdefault("correlation_id", getattr(request.state, "correlation_id", None))
        payload_context.setdefault("path", request.url.path)
        payload_context.setdefault("method", request.method)

    return {
        "code": code,
        "message": message,
        "detail": detail,
        "context": payload_context,
    }


def raise_api_error(
    *,
    status_code: int,
    code: str,
    message: str,
    detail: Any = None,
    context: dict[str, Any] | None = None,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "detail": detail,
            "context": context or {},
        },
    )
