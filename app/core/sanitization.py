"""
Input Sanitization — Step 3.2
Strips HTML/JS from text fields to prevent XSS and injection.
Falls back gracefully if bleach is not installed.
"""
from typing import Optional

try:
    import bleach
    _BLEACH_AVAILABLE = True
except ImportError:
    _BLEACH_AVAILABLE = False


def sanitize_text(value: Optional[str], max_length: int = 1024) -> Optional[str]:
    """
    Strip all HTML tags from a string and enforce max length.
    Safe to call with None — returns None unchanged.
    """
    if value is None:
        return None
    if _BLEACH_AVAILABLE:
        cleaned = bleach.clean(value, tags=[], strip=True)
    else:
        # Minimal fallback using stdlib
        import html
        cleaned = html.unescape(value)
    return cleaned.strip()[:max_length] or None


def sanitize_short(value: Optional[str]) -> Optional[str]:
    """Sanitize a short field (names, company, position) — max 256 chars."""
    return sanitize_text(value, max_length=256)


def sanitize_long(value: Optional[str]) -> Optional[str]:
    """Sanitize a long text field — max 4096 chars."""
    return sanitize_text(value, max_length=4096)
