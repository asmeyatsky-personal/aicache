"""Telemetry adapters + OpenTelemetry shim (§6).

Two concrete :class:`TelemetryPort` implementations:

- :class:`InMemoryTelemetryAdapter` — keeps events in a deque, useful
  for tests and short-lived processes.
- :class:`JSONLTelemetryAdapter` — append-only newline-delimited JSON
  on disk. Rolling-savings reports read it back; offline tools can
  pipe the file into DuckDB / Polars for ad-hoc analysis (§6).

Plus :func:`traced` — a no-op decorator unless the optional
``opentelemetry`` extra is installed, in which case spans are emitted
around the wrapped coroutine. Keeps the core path dependency-free.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from ..domain.models import AICallEvent
from ..domain.ports import TelemetryPort

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------
# Telemetry adapters
# ---------------------------------------------------------------------


class InMemoryTelemetryAdapter(TelemetryPort):
    """Keeps the most recent events in memory. FIFO-bounded."""

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: deque[AICallEvent] = deque(maxlen=max_events)

    async def record_ai_call(self, event: AICallEvent) -> None:
        self._events.append(event)

    async def read_recent(self, days: int = 30) -> list[AICallEvent]:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        return [e for e in self._events if _aware(e.timestamp) >= cutoff]


class JSONLTelemetryAdapter(TelemetryPort):
    """Append-only newline-delimited JSON log.

    One write per call, flushed immediately. We accept the small
    overhead for the big wins: crash-safe, tail-friendly, trivially
    ingestible by DuckDB / ``jq`` / Polars.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        default = Path.home() / ".cache" / "aicache" / "events.jsonl"
        self._path = Path(path) if path else default
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def record_ai_call(self, event: AICallEvent) -> None:
        line = json.dumps(_event_to_dict(event), separators=(",", ":"))
        # Synchronous append; cheap (single fsync-less write) and keeps
        # ordering deterministic. Move to aiofiles only if the event
        # log becomes a hot path.
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    async def read_recent(self, days: int = 30) -> list[AICallEvent]:
        if not self._path.exists():
            return []
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        events: list[AICallEvent] = []
        with self._path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = _dict_to_event(data)
                except (json.JSONDecodeError, ValueError, KeyError) as err:
                    logger.warning("skipping malformed event: %s", err)
                    continue
                if _aware(event.timestamp) >= cutoff:
                    events.append(event)
        return events


# ---------------------------------------------------------------------
# Aggregation helpers — powers `aicache stats`
# ---------------------------------------------------------------------


def summarise(events: list[AICallEvent]) -> dict[str, Any]:
    """Roll per-call events into totals for a stats report."""
    total = len(events)
    hits = sum(1 for e in events if e.cache_hit)
    misses = total - hits
    cost_saved = sum(e.cost_usd for e in events if e.cache_hit)
    cost_spent = sum(e.cost_usd for e in events if not e.cache_hit)
    tokens_saved = sum(e.tokens_in + e.tokens_out for e in events if e.cache_hit)
    per_model: dict[str, dict[str, Any]] = {}
    for event in events:
        bucket = per_model.setdefault(
            event.model,
            {"hits": 0, "misses": 0, "cost_saved_usd": 0.0, "cost_spent_usd": 0.0},
        )
        if event.cache_hit:
            bucket["hits"] += 1
            bucket["cost_saved_usd"] += event.cost_usd
        else:
            bucket["misses"] += 1
            bucket["cost_spent_usd"] += event.cost_usd

    return {
        "total_calls": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total) if total else 0.0,
        "cost_saved_usd": round(cost_saved, 6),
        "cost_spent_usd": round(cost_spent, 6),
        "tokens_saved": tokens_saved,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------
# OpenTelemetry shim
# ---------------------------------------------------------------------


def traced(span_name: str) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function with an OTel span when available.

    Importing opentelemetry is optional (``aicache[observability]``).
    When absent, this is a free no-op — the wrapped coroutine is
    awaited directly, so the cache-path cost stays at zero.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        tracer = _get_tracer()
        if tracer is None:
            return fn

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            with tracer.start_as_current_span(span_name):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


def _get_tracer() -> Any | None:
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError:
        return None
    return trace.get_tracer("aicache")


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _event_to_dict(event: AICallEvent) -> dict[str, Any]:
    return {
        "timestamp": _aware(event.timestamp).isoformat(),
        "provider": event.provider,
        "model": event.model,
        "model_version": event.model_version,
        "prompt_hash": event.prompt_hash,
        "tokens_in": event.tokens_in,
        "tokens_out": event.tokens_out,
        "latency_ms": event.latency_ms,
        "cost_usd": event.cost_usd,
        "cache_hit": event.cache_hit,
        "match_type": event.match_type,
    }


def _dict_to_event(data: dict[str, Any]) -> AICallEvent:
    return AICallEvent(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        provider=data["provider"],
        model=data["model"],
        model_version=data.get("model_version"),
        prompt_hash=data["prompt_hash"],
        tokens_in=int(data.get("tokens_in", 0)),
        tokens_out=int(data.get("tokens_out", 0)),
        latency_ms=float(data.get("latency_ms", 0.0)),
        cost_usd=float(data.get("cost_usd", 0.0)),
        cache_hit=bool(data.get("cache_hit", False)),
        match_type=data.get("match_type", "miss"),
    )


# ---------------------------------------------------------------------
# JSON log formatter — structured stdout for §6 "structured JSON logs"
# ---------------------------------------------------------------------


class JSONLogFormatter(logging.Formatter):
    """stdlib logging formatter that writes newline-delimited JSON.

    Applied by :func:`configure_structured_logging` only when the
    ``AICACHE_JSON_LOGS=1`` env var is set, so the CLI's Rich output
    isn't disturbed by default.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("correlation_id",):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_structured_logging() -> None:  # pragma: no cover — side-effect only
    if os.environ.get("AICACHE_JSON_LOGS") != "1":
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    root = logging.getLogger("aicache")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# Fire-and-forget recording helper used by synchronous integration
# paths that can't easily await. Schedules on the running loop if
# present, else runs inline.
def record_event_blocking(telemetry: TelemetryPort, event: AICallEvent) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(telemetry.record_ai_call(event))
        return
    # Inside an event loop — use a thread to avoid nested-loop trouble
    # (same strategy as the SDK integrations).
    import concurrent.futures

    async def _runner() -> None:
        await telemetry.record_ai_call(event)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(asyncio.run, _runner()).result()
