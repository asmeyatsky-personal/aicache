"""Shared helpers for SDK integrations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Fields that must participate in the fingerprint for a cached response
# to be correct. Temperature/seed/top_p change output, so they vary the
# key. Streaming is excluded because a cached response satisfies both
# streaming and non-streaming calls.
_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "model",
    "messages",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
    "stop",
    "stop_sequences",
)


def build_request_fingerprint(
    provider: str,
    params: dict[str, Any],
) -> str:
    """Build a deterministic, collision-resistant cache key for a request.

    Only fields that affect output are hashed. Stream flag, timeouts,
    and SDK-specific transport options are ignored so a cached response
    satisfies both streaming and non-streaming invocations.
    """

    payload: dict[str, Any] = {"provider": provider}
    for field in _FINGERPRINT_FIELDS:
        if field in params and params[field] is not None:
            payload[field] = params[field]

    # json.dumps with sort_keys is stable across Python runs; for
    # non-JSON-serialisable content (eg. enum members) fall back to
    # repr so we still get *a* stable key.
    try:
        serialised = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        serialised = repr(sorted(payload.items()))

    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return f"{provider}:{digest}"


def extract_text_for_fingerprint(params: dict[str, Any]) -> str:
    """Flatten messages + system prompt into a single string.

    Used by the QueryNormalizer / semantic path — not for exact-match
    keying (which uses the full fingerprint above).
    """
    parts: list[str] = []
    system = params.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(str(block) for block in system)

    messages = params.get("messages") or []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))

    return "\n".join(parts)
