"""Cached wrapper for the Anthropic SDK.

Usage::

    from anthropic import Anthropic
    from aicache.integrations.anthropic import cached_client

    client = cached_client(Anthropic())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

The wrapper preserves the SDK's type signatures — the returned object
is the SDK's native ``Message`` — and intercepts ``messages.create``
through the caching use cases wired in :class:`aicache.infrastructure.Container`.

No Anthropic SDK import happens unless the user actually constructs a
cached client, so ``aicache`` stays installable without the anthropic
optional dependency.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Protocol

from ..infrastructure import Container, get_container
from ._common import build_request_fingerprint

if TYPE_CHECKING:  # pragma: no cover
    pass


_PROVIDER = "anthropic"


class _MessagesCreateResult(Protocol):
    """Structural type for the object returned by ``messages.create``.

    The SDK's concrete class is ``anthropic.types.Message``; duck-typing
    keeps the wrapper decoupled from any specific SDK version.
    """

    def model_dump_json(self) -> str: ...


class _CachedMessages:
    """Proxy for ``client.messages`` that adds caching to ``create``."""

    def __init__(self, upstream: Any, container: Container) -> None:
        self._upstream = upstream
        self._container = container

    def __getattr__(self, name: str) -> Any:
        # Everything we don't explicitly intercept (count_tokens, stream
        # helpers, etc.) passes straight through to the real SDK.
        return getattr(self._upstream, name)

    def create(self, **params: Any) -> Any:
        fingerprint = build_request_fingerprint(_PROVIDER, params)
        storage_key = _storage_key(fingerprint)

        cached = _run_sync(self._container.query_cache.execute(fingerprint))
        if cached.hit and cached.value is not None:
            return _deserialise_message(cached.value, params)

        response = self._upstream.create(**params)
        serialised = _serialise_message(response)
        _run_sync(self._container.store_cache.execute(key=storage_key, value=serialised))
        return response

    async def acreate(self, **params: Any) -> Any:
        """Async variant — used when the upstream client is ``AsyncAnthropic``.

        Anthropic's async client exposes ``messages.create`` as a
        coroutine, so we need a mirrored async path to avoid running a
        new event loop inside an already-running one.
        """
        fingerprint = build_request_fingerprint(_PROVIDER, params)
        storage_key = _storage_key(fingerprint)

        cached = await self._container.query_cache.execute(fingerprint)
        if cached.hit and cached.value is not None:
            return _deserialise_message(cached.value, params)

        response = await self._upstream.create(**params)
        serialised = _serialise_message(response)
        await self._container.store_cache.execute(key=storage_key, value=serialised)
        return response


class CachedAnthropic:
    """Drop-in replacement for :class:`anthropic.Anthropic`."""

    def __init__(self, client: Any, container: Container | None = None) -> None:
        self._client = client
        self._container = container or get_container()
        self.messages = _CachedMessages(client.messages, self._container)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def cached_client(client: Any, *, container: Container | None = None) -> CachedAnthropic:
    """Wrap an Anthropic SDK client with caching.

    Accepts either a sync ``Anthropic`` or async ``AsyncAnthropic``
    instance; the ``.messages.create`` and ``.messages.acreate`` methods
    route through the shared :class:`Container`.
    """
    return CachedAnthropic(client, container=container)


# ---------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------


def _serialise_message(message: Any) -> bytes:
    """Serialise a Message-like object to JSON bytes.

    Tries the SDK's Pydantic-native ``model_dump_json`` first, then falls
    back to ``json.dumps`` on plain dicts (useful in tests with stub
    responses).
    """
    dumper = getattr(message, "model_dump_json", None)
    if callable(dumper):
        return dumper().encode("utf-8")

    if isinstance(message, dict):
        return json.dumps(message).encode("utf-8")

    raise TypeError(
        f"Don't know how to serialise Anthropic response of type {type(message).__name__}"
    )


def _deserialise_message(payload: bytes, params: dict[str, Any]) -> Any:
    """Deserialise a cached JSON payload back to the SDK's Message.

    If the anthropic SDK is importable, reconstruct its native type so
    the caller sees the same object they would from a real call. Fall
    back to a plain dict when the SDK is absent (tests, minimal installs).
    """
    try:  # pragma: no cover — only when anthropic SDK is installed
        from anthropic.types import Message  # type: ignore[import-not-found]

        return Message.model_validate_json(payload)
    except Exception:
        return json.loads(payload)


def _storage_key(fingerprint: str) -> str:
    """Mirror ``QueryCacheUseCase._generate_cache_key`` so writes and
    reads land on the same storage slot.

    The use case hashes ``(normalize(query), context)``. Our
    ``QueryNormalizer.normalize`` is ``str.lower().strip()``; the
    fingerprint is already lowercase-hex, so passing it verbatim is
    equivalent to the normalised form.
    """
    from ..application.use_cases import QueryCacheUseCase

    return QueryCacheUseCase._generate_cache_key(fingerprint, None)


def _run_sync(coro: Any) -> Any:
    """Run an awaitable from sync code without deadlocking.

    The SDK integration is called from user code that may or may not
    already be inside an event loop. Detect and dispatch accordingly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # We're inside a loop — schedule on a throwaway loop in a thread so
    # we don't block or nest event loops. Rare path; kept for safety.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _materialise(coro)).result()


async def _materialise(coro: Any) -> Any:
    return await coro
