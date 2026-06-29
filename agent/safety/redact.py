"""Secret redaction for logs AND reports (Phase 5).

A single :func:`redact` function scrubs obvious secret patterns (OpenAI/Anthropic
``sk-...`` keys, ``Bearer`` tokens, Google ``AIza...`` keys) plus the *actual*
configured key values from settings. It is wired into the logging handlers via
:class:`RedactionFilter` and applied to every report before it is written, so a
secret never reaches ``logs/agent.log`` or any report — even on error paths.

This module imports settings lazily (inside functions) so it can be safely
imported from ``agent.config`` while logging is being set up.
"""

import logging
import re

REDACTED = "***REDACTED***"

# Structural patterns for obvious secrets, independent of any configured value.
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),                # OpenAI / Anthropic style keys
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=/+]{6,}", re.I),  # Authorization: Bearer <token>
    re.compile(r"AIza[A-Za-z0-9_\-]{10,}"),              # Google API keys
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}"),            # Anthropic explicit prefix
]


def _configured_secret_values():
    """The actual key values from settings (longest first), or [] if unavailable."""
    try:
        from agent.config import settings
    except Exception:
        return []
    values = []
    for attr in ("anthropic_api_key", "openai_api_key", "google_api_key"):
        val = (getattr(settings, attr, "") or "").strip()
        if val:
            values.append(val)
    return sorted(values, key=len, reverse=True)


def redact(text):
    """Return ``text`` with secrets replaced by ``***REDACTED***``."""
    if text is None:
        return text
    s = text if isinstance(text, str) else str(text)
    # Exact configured values first (longest first so a prefix can't leak).
    for val in _configured_secret_values():
        if val:
            s = s.replace(val, REDACTED)
    for pat in _PATTERNS:
        s = pat.sub(REDACTED, s)
    return s


class RedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from every formatted log record."""

    def filter(self, record):
        try:
            message = record.getMessage()
            redacted = redact(message)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        except Exception:
            # Never let redaction break logging.
            pass
        return True
