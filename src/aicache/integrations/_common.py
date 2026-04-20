"""Shared helpers for SDK integrations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from ..domain.models import AICallEvent
from ..infrastructure import Container

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


def record_ai_call_event(
    container: Container,
    *,
    provider: str,
    params: dict[str, Any],
    response: Any,
    fingerprint: str,
    latency_ms: float,
    cache_hit: bool,
    match_type: str,
) -> AICallEvent:
    """Build and record an :class:`AICallEvent` via the container.

    Token counts are pulled from the SDK's ``response.usage`` when the
    response is present (miss path or cache hit with payload), falling
    back to zero when the SDK-native usage field is absent. Cost is
    estimated through the token counter port so the port abstraction is
    honoured — no SDK pricing leaks into business logic (§3.3).
    """

    model = str(params.get("model", "unknown"))
    tokens_in, tokens_out = _extract_usage(response, provider, params, container, fingerprint)
    cost_usd = container.token_counter.estimate_cost(model, tokens_in, tokens_out)

    event = AICallEvent(
        timestamp=datetime.now(tz=UTC),
        provider=provider,
        model=model,
        model_version=_extract_model_version(response),
        prompt_hash=fingerprint,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        match_type=match_type,
    )

    from ..infrastructure.telemetry import record_event_blocking

    record_event_blocking(container.telemetry, event)
    return event


def _extract_usage(
    response: Any,
    provider: str,
    params: dict[str, Any],
    container: Container,
    fingerprint: str,
) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a provider response.

    Anthropic and OpenAI both expose ``usage`` on Pydantic responses
    (input_tokens/output_tokens and prompt_tokens/completion_tokens
    respectively). On cache hits without usage we approximate via the
    token counter so downstream reports still have a number.
    """

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is not None:
        if provider == "anthropic":
            tokens_in = _attr(usage, "input_tokens")
            tokens_out = _attr(usage, "output_tokens")
            return int(tokens_in or 0), int(tokens_out or 0)
        if provider == "openai":
            tokens_in = _attr(usage, "prompt_tokens")
            tokens_out = _attr(usage, "completion_tokens")
            return int(tokens_in or 0), int(tokens_out or 0)

    # Fallback: estimate from the flattened prompt. Good enough for
    # telemetry; not a substitute for usage fields when present.
    model = str(params.get("model", "unknown"))
    prompt = extract_text_for_fingerprint(params)
    return container.token_counter.count_prompt_tokens(prompt, model), 0


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_model_version(response: Any) -> str | None:
    if response is None:
        return None
    return _attr(response, "model")


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
