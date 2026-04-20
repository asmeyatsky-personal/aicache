"""Cached wrapper for the OpenAI SDK.

Usage::

    from openai import OpenAI
    from aicache.integrations.openai import cached_client

    client = cached_client(OpenAI())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import asyncio
import json
import time as _time
from typing import TYPE_CHECKING, Any, Protocol

from ..infrastructure import Container, get_container
from ._common import (
    build_request_fingerprint,
    extract_text_for_fingerprint,
    record_ai_call_event,
)

if TYPE_CHECKING:  # pragma: no cover
    pass


_PROVIDER = "openai"


class _ChatCompletionResult(Protocol):
    def model_dump_json(self) -> str: ...


class _CachedCompletions:
    def __init__(self, upstream: Any, container: Container) -> None:
        self._upstream = upstream
        self._container = container

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)

    def create(self, **params: Any) -> Any:
        fingerprint = build_request_fingerprint(_PROVIDER, params)
        storage_key = _storage_key(fingerprint)
        prompt = extract_text_for_fingerprint(params)

        start = _time.perf_counter()
        cached = _run_sync(self._container.query_cache.execute(fingerprint, query_text=prompt))
        if cached.hit and cached.value is not None:
            response = _deserialise_chat_completion(cached.value)
            _log_call(self._container, params, response, fingerprint, start, True, cached)
            return response

        response = self._upstream.create(**params)
        serialised = _serialise_chat_completion(response)
        _run_sync(
            self._container.store_cache.execute(
                key=storage_key, value=serialised, query_text=prompt
            )
        )
        _log_call(self._container, params, response, fingerprint, start, False, cached)
        return response

    async def acreate(self, **params: Any) -> Any:
        fingerprint = build_request_fingerprint(_PROVIDER, params)
        storage_key = _storage_key(fingerprint)
        prompt = extract_text_for_fingerprint(params)

        start = _time.perf_counter()
        cached = await self._container.query_cache.execute(fingerprint, query_text=prompt)
        if cached.hit and cached.value is not None:
            response = _deserialise_chat_completion(cached.value)
            _log_call(self._container, params, response, fingerprint, start, True, cached)
            return response

        response = await self._upstream.create(**params)
        serialised = _serialise_chat_completion(response)
        await self._container.store_cache.execute(
            key=storage_key, value=serialised, query_text=prompt
        )
        _log_call(self._container, params, response, fingerprint, start, False, cached)
        return response


class _CachedChat:
    def __init__(self, upstream: Any, container: Container) -> None:
        self._upstream = upstream
        self.completions = _CachedCompletions(upstream.completions, container)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


class CachedOpenAI:
    def __init__(self, client: Any, container: Container | None = None) -> None:
        self._client = client
        self._container = container or get_container()
        self.chat = _CachedChat(client.chat, self._container)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def cached_client(client: Any, *, container: Container | None = None) -> CachedOpenAI:
    """Wrap an OpenAI SDK client with caching."""
    return CachedOpenAI(client, container=container)


# ---------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------


def _serialise_chat_completion(completion: Any) -> bytes:
    dumper = getattr(completion, "model_dump_json", None)
    if callable(dumper):
        return dumper().encode("utf-8")
    if isinstance(completion, dict):
        return json.dumps(completion).encode("utf-8")
    raise TypeError(
        f"Don't know how to serialise OpenAI response of type {type(completion).__name__}"
    )


def _deserialise_chat_completion(payload: bytes) -> Any:
    try:  # pragma: no cover — only when openai SDK is installed
        from openai.types.chat import ChatCompletion  # type: ignore[import-not-found]

        return ChatCompletion.model_validate_json(payload)
    except Exception:
        return json.loads(payload)


def _storage_key(fingerprint: str) -> str:
    """Match the hashing used by ``QueryCacheUseCase`` so writes and
    reads land on the same slot."""
    from ..application.use_cases import QueryCacheUseCase

    return QueryCacheUseCase._generate_cache_key(fingerprint, None)


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _materialise(coro)).result()


async def _materialise(coro: Any) -> Any:
    return await coro


def _log_call(
    container: Any,
    params: dict[str, Any],
    response: Any,
    fingerprint: str,
    start: float,
    cache_hit: bool,
    cached: Any,
) -> None:
    record_ai_call_event(
        container,
        provider=_PROVIDER,
        params=params,
        response=response,
        fingerprint=fingerprint,
        latency_ms=(_time.perf_counter() - start) * 1000.0,
        cache_hit=cache_hit,
        match_type=_match_type(cache_hit, cached),
    )


def _match_type(cache_hit: bool, cached: Any) -> str:
    if not cache_hit:
        return "miss"
    if getattr(cached, "similarity_score", None) is not None:
        return "semantic"
    return "exact"
