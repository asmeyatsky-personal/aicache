"""
Composition root (infrastructure layer).

Wires domain ports to infrastructure adapters and builds application
use cases. This is the one place in the runtime that knows which
adapters exist — everything else depends on ports.

Per Architectural Rules 2026 §3.6: dependencies are explicit; per §2:
presentation imports the container, the container imports application
and infrastructure — domain never sees this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..application.use_cases import (
    CacheMetricsUseCase,
    InvalidateCacheUseCase,
    QueryCacheUseCase,
    StoreCacheUseCase,
)
from ..core.cache import CoreCache
from ..domain.models import CachePolicy, EvictionPolicy
from ..domain.ports import (
    CacheMetricsPort,
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
)
from .adapters import (
    FileSystemStorageAdapter,
    InMemoryCacheMetricsAdapter,
    InMemoryEventPublisherAdapter,
    InMemoryStorageAdapter,
    OpenAITokenCounterAdapter,
    SimpleEmbeddingGeneratorAdapter,
    SimpleQueryNormalizerAdapter,
    SimpleSemanticIndexAdapter,
)


@dataclass
class Container:
    """Wired application.

    `admin_cache` exposes the legacy CoreCache for administrative CLI
    operations (stats, list, inspect, prune) that pre-date the hex
    layers. It is kept on the CoreCache file format for backward
    compatibility with existing user caches. New business operations go
    through the use cases.
    """

    # Ports (what application depends on)
    storage: StoragePort
    semantic_index: SemanticIndexPort
    token_counter: TokenCounterPort
    query_normalizer: QueryNormalizerPort
    embedding_generator: EmbeddingGeneratorPort
    metrics: CacheMetricsPort
    event_publisher: EventPublisherPort
    policy: CachePolicy

    # Application use cases
    query_cache: QueryCacheUseCase
    store_cache: StoreCacheUseCase
    invalidate_cache: InvalidateCacheUseCase
    cache_metrics: CacheMetricsUseCase

    # Legacy admin-only handle — kept until Phase 3 retires it
    admin_cache: CoreCache


def _default_policy(enable_semantic: bool = False) -> CachePolicy:
    return CachePolicy(
        max_size_bytes=1024 * 1024 * 1024,  # 1 GiB
        default_ttl_seconds=None,
        eviction_policy=EvictionPolicy.LRU,
        semantic_match_threshold=0.85,
        enable_compression=False,
        enable_semantic_caching=enable_semantic,
    )


def _build_semantic_adapters(
    persist_directory: str | None,
) -> tuple[SemanticIndexPort, EmbeddingGeneratorPort]:
    """Construct the real Chroma + sentence-transformers pair.

    Kept isolated so the ImportError from a missing extra surfaces
    clearly at container build time rather than at first query.
    """
    from .semantic_adapters import (
        ChromaSemanticIndexAdapter,
        SentenceTransformerEmbeddingAdapter,
    )

    index = ChromaSemanticIndexAdapter(persist_directory=persist_directory)
    embedder = SentenceTransformerEmbeddingAdapter()
    return index, embedder


def build_container(
    *,
    cache_dir: str | Path | None = None,
    policy: CachePolicy | None = None,
    in_memory: bool = False,
    semantic: bool = False,
    semantic_persist_directory: str | None = None,
    storage: StoragePort | None = None,
    semantic_index: SemanticIndexPort | None = None,
    token_counter: TokenCounterPort | None = None,
    query_normalizer: QueryNormalizerPort | None = None,
    embedding_generator: EmbeddingGeneratorPort | None = None,
    metrics: CacheMetricsPort | None = None,
    event_publisher: EventPublisherPort | None = None,
) -> Container:
    """Build a wired :class:`Container`.

    All ports accept injection so tests can pass in-memory fakes.
    ``in_memory=True`` swaps the default filesystem storage for an
    in-memory adapter — the standard seam for application-layer tests.
    """

    policy = policy or _default_policy(enable_semantic=semantic)
    resolved_cache_dir = (
        str(Path(cache_dir).expanduser()) if cache_dir else str(Path.home() / ".cache" / "aicache")
    )

    storage = storage or (
        InMemoryStorageAdapter() if in_memory else FileSystemStorageAdapter(resolved_cache_dir)
    )

    if semantic_index is None or embedding_generator is None:
        if semantic:
            default_index, default_embedder = _build_semantic_adapters(semantic_persist_directory)
        else:
            default_index, default_embedder = (
                SimpleSemanticIndexAdapter(),
                SimpleEmbeddingGeneratorAdapter(),
            )
        semantic_index = semantic_index or default_index
        embedding_generator = embedding_generator or default_embedder

    token_counter = token_counter or OpenAITokenCounterAdapter()
    query_normalizer = query_normalizer or SimpleQueryNormalizerAdapter()
    metrics = metrics or InMemoryCacheMetricsAdapter()
    event_publisher = event_publisher or InMemoryEventPublisherAdapter()

    query_cache = QueryCacheUseCase(
        storage=storage,
        semantic_index=semantic_index,
        token_counter=token_counter,
        query_normalizer=query_normalizer,
        embedding_generator=embedding_generator,
        metrics=metrics,
        cache_policy=policy,
    )
    store_cache = StoreCacheUseCase(
        storage=storage,
        semantic_index=semantic_index,
        embedding_generator=embedding_generator,
        metrics=metrics,
        cache_policy=policy,
    )
    invalidate_cache = InvalidateCacheUseCase(
        storage=storage,
        semantic_index=semantic_index,
        event_publisher=event_publisher,
        metrics=metrics,
    )
    cache_metrics = CacheMetricsUseCase(metrics=metrics)

    # Admin handle — CoreCache for legacy CLI ops. Only presentation
    # touches this, and only for list/stats/inspect/prune.
    admin_cache = CoreCache(cache_dir=resolved_cache_dir) if not in_memory else CoreCache()

    return Container(
        storage=storage,
        semantic_index=semantic_index,
        token_counter=token_counter,
        query_normalizer=query_normalizer,
        embedding_generator=embedding_generator,
        metrics=metrics,
        event_publisher=event_publisher,
        policy=policy,
        query_cache=query_cache,
        store_cache=store_cache,
        invalidate_cache=invalidate_cache,
        cache_metrics=cache_metrics,
        admin_cache=admin_cache,
    )


_default_container: Container | None = None


def get_container(**kwargs: Any) -> Container:
    """Return the process-wide default container, building it on first use.

    Passing any keyword arg rebuilds the default. Tests typically skip
    this helper and call :func:`build_container` directly.
    """
    global _default_container
    if _default_container is None or kwargs:
        _default_container = build_container(**kwargs)
    return _default_container


def reset_container() -> None:
    """Drop the cached default container. Useful between tests."""
    global _default_container
    _default_container = None
