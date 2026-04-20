"""
Infrastructure Layer: MCP Server (2026 Edition)

This MCP server exposes cache capabilities following skill2026.md principles:
- Uses application layer use cases (not domain directly)
- Tools = write operations
- Resources = read operations
- Proper dependency injection
"""

import logging
from typing import Any

from ..application.use_cases import (
    CacheMetricsUseCase,
    InvalidateCacheUseCase,
    QueryCacheUseCase,
    StoreCacheUseCase,
)
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
from ..infrastructure.adapters import (
    InMemoryCacheMetricsAdapter,
    InMemoryEventPublisherAdapter,
    InMemoryStorageAdapter,
    OpenAITokenCounterAdapter,
    SimpleEmbeddingGeneratorAdapter,
    SimpleQueryNormalizerAdapter,
    SimpleSemanticIndexAdapter,
)

logger = logging.getLogger(__name__)


class MCPCacheServer:
    """
    MCP Server for AI Cache - Bounded Context as MCP Server.

    2026 Pattern: Following skill2026.md - each bounded context
    has exactly one MCP server with Tools for writes and Resources for reads.
    """

    def __init__(
        self,
        storage: StoragePort | None = None,
        semantic_index: SemanticIndexPort | None = None,
        token_counter: TokenCounterPort | None = None,
        query_normalizer: QueryNormalizerPort | None = None,
        embedding_generator: EmbeddingGeneratorPort | None = None,
        event_publisher: EventPublisherPort | None = None,
        metrics: CacheMetricsPort | None = None,
    ):
        # Dependency injection with defaults
        self.storage = storage or InMemoryStorageAdapter()
        self.semantic_index = semantic_index or SimpleSemanticIndexAdapter()
        self.token_counter = token_counter or OpenAITokenCounterAdapter()
        self.query_normalizer = query_normalizer or SimpleQueryNormalizerAdapter()
        self.embedding_generator = embedding_generator or SimpleEmbeddingGeneratorAdapter()
        self.event_publisher = event_publisher or InMemoryEventPublisherAdapter()
        self.metrics = metrics or InMemoryCacheMetricsAdapter()

        # Create cache policy
        self.policy = CachePolicy(
            max_size_bytes=100_000_000,  # 100MB
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
            semantic_match_threshold=0.85,
            enable_compression=True,
            enable_semantic_caching=True,
        )

        # Initialize use cases (application layer)
        self._query_use_case = QueryCacheUseCase(
            storage=self.storage,
            semantic_index=self.semantic_index,
            token_counter=self.token_counter,
            query_normalizer=self.query_normalizer,
            embedding_generator=self.embedding_generator,
            metrics=self.metrics,
            cache_policy=self.policy,
        )

        self._store_use_case = StoreCacheUseCase(
            storage=self.storage,
            semantic_index=self.semantic_index,
            embedding_generator=self.embedding_generator,
            metrics=self.metrics,
            cache_policy=self.policy,
        )

        self._invalidate_use_case = InvalidateCacheUseCase(
            storage=self.storage,
            semantic_index=self.semantic_index,
            event_publisher=self.event_publisher,
            metrics=self.metrics,
        )

        self._metrics_use_case = CacheMetricsUseCase(metrics=self.metrics)

    # ========== MCP TOOLS (Write Operations) ==========

    async def cache_query(
        self,
        query: str,
        response: str,
        context: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Cache a query-response pair.

        This is a TOOL (write operation) per skill2026.md.
        """
        import hashlib

        cache_key = hashlib.sha256(query.encode()).hexdigest()

        await self._store_use_case.execute(
            key=cache_key,
            value=response.encode("utf-8"),
            ttl_seconds=ttl_seconds,
            context=context,
        )

        return {
            "success": True,
            "cache_key": cache_key,
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
        }

    async def invalidate_cache(
        self, cache_key: str, reason: str = "user_request"
    ) -> dict[str, Any]:
        """
        Invalidate a specific cache entry.

        TOOL (write operation).
        """
        await self._invalidate_use_case.invalidate_key(cache_key, reason)

        return {"success": True, "cache_key": cache_key, "reason": reason}

    async def purge_expired(self) -> dict[str, Any]:
        """
        Purge all expired cache entries.

        TOOL (write operation).
        """
        count = await self._invalidate_use_case.purge_expired()

        return {"success": True, "purged_count": count}

    async def clear_cache(self, confirm: bool = False) -> dict[str, Any]:
        """
        Clear all cache entries. Requires confirm=true.

        TOOL (write operation).
        """
        if not confirm:
            return {"success": False, "message": "Must set confirm=true to clear cache"}

        await self.storage.clear()

        return {"success": True, "message": "Cache cleared"}

    # ========== MCP RESOURCES (Read Operations) ==========

    async def get_cache_entry(self, cache_key: str) -> dict[str, Any] | None:
        """
        Get a specific cache entry by key.

        RESOURCE (read operation).
        """
        entry = await self.storage.get(cache_key)

        if entry is None:
            return None

        return {
            "cache_key": entry.key,
            "value": entry.value.decode("utf-8") if entry.value else "",
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "is_expired": entry.is_expired(),
            "size_bytes": entry.get_size_bytes(),
        }

    async def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        RESOURCE (read operation).
        """
        metrics = await self._metrics_use_case.get_metrics()
        size_bytes = await self.storage.get_size_bytes()

        return {
            "total_entries": metrics.get("total_hits", 0) + metrics.get("total_misses", 0),
            "hit_rate": metrics.get("hit_rate", 0.0),
            "total_hits": metrics.get("total_hits", 0),
            "total_misses": metrics.get("total_misses", 0),
            "total_evictions": metrics.get("total_evictions", 0),
            "cache_size_bytes": size_bytes,
            "estimated_cost_saved": metrics.get("total_cost_saved", 0.0),
        }

    async def list_cache_entries(
        self, limit: int = 10, include_expired: bool = False
    ) -> dict[str, Any]:
        """
        List cache entries.

        RESOURCE (read operation).
        """
        keys = await self.storage.get_all_keys()
        entries = []

        for key in keys[:limit]:
            entry = await self.storage.get(key)
            if entry and (include_expired or not entry.is_expired()):
                entries.append(
                    {
                        "cache_key": entry.key,
                        "created_at": entry.created_at.isoformat() if entry.created_at else None,
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        "is_expired": entry.is_expired(),
                        "size_bytes": entry.get_size_bytes(),
                    }
                )

        return {"entries": entries, "total": len(entries), "limit": limit}

    # ========== MCP PROMPTS (Reusable Patterns) ==========

    async def cache_summary_prompt(self) -> str:
        """
        Generate a cache summary prompt for agent consumption.

        PROMPT (reusable pattern).
        """
        stats = await self.get_cache_stats()

        return f"""Current Cache Status:
- Hit Rate: {stats.get("hit_rate", 0):.1%}
- Total Hits: {stats.get("total_hits", 0)}
- Total Misses: {stats.get("total_misses", 0)}
- Cache Size: {stats.get("cache_size_bytes", 0):,} bytes
- Estimated Cost Saved: ${stats.get("estimated_cost_saved", 0):.2f}
"""


def create_mcp_cache_server() -> MCPCacheServer:
    """
    Factory function to create MCP cache server.

    This is the composition root where dependencies are wired.
    """
    return MCPCacheServer()


def get_mcp_server_config() -> dict[str, Any]:
    """
    Return MCP server configuration for registry.
    """
    return {
        "cache-service": {
            "command": "python",
            "args": ["-m", "aicache.infrastructure.mcp_server"],
            "env": {},
        }
    }
