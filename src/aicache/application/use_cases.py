"""
Application Layer: Use Cases (Application Services)

Use cases orchestrate domain services to implement specific features.
Each use case represents a single business operation.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from ..domain.models import CacheEntry, CacheMetadata, CachePolicy, CacheResult
from ..domain.ports import (
    CacheMetricsPort,
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
)
from ..domain.services import (
    CacheEvictionService,
    CacheInvalidationService,
    CacheTTLService,
    QueryNormalizationService,
    SemanticCachingService,
    TokenCountingService,
)

logger = logging.getLogger(__name__)


class QueryCacheUseCase:
    """
    Main use case for querying the cache.

    This use case orchestrates multiple services to:
    1. Check for exact match
    2. Check for semantic match
    3. Return result or indicate miss
    4. Track metrics
    """

    def __init__(
        self,
        storage: StoragePort,
        semantic_index: SemanticIndexPort,
        token_counter: TokenCounterPort,
        query_normalizer: QueryNormalizerPort,
        embedding_generator: EmbeddingGeneratorPort,
        metrics: CacheMetricsPort,
        cache_policy: CachePolicy,
    ):
        self.storage = storage
        self.semantic_caching = SemanticCachingService(semantic_index, embedding_generator)
        self.query_normalization = QueryNormalizationService(query_normalizer)
        self.token_counting = TokenCountingService(token_counter)
        self.metrics = metrics
        self.policy = cache_policy
        self.ttl_service = CacheTTLService()

    async def execute(self, query: str, context: dict[str, Any] | None = None) -> CacheResult:
        """Execute cache query."""
        start_time = time.time()

        try:
            # Step 1: Try exact match with normalization
            exact_result = await self._try_exact_match(query, context)
            if exact_result.hit:
                response_time_ms = (time.time() - start_time) * 1000
                await self.metrics.record_hit(
                    exact_result.entry_key or "", response_time_ms, tokens_saved=0, cost_saved=0.0
                )
                return CacheResult.create_hit(
                    exact_result.value, exact_result.entry_key, response_time_ms
                )

            # Step 2: Try semantic match if enabled
            if self.policy.enable_semantic_caching:
                semantic_result = await self._try_semantic_match(query)
                if semantic_result.hit and semantic_result.confidence > 0.85:
                    response_time_ms = (time.time() - start_time) * 1000
                    await self.metrics.record_hit(
                        semantic_result.entry_key or "",
                        response_time_ms,
                        tokens_saved=0,
                        cost_saved=0.0,
                    )
                    return semantic_result

            # Step 3: Cache miss
            response_time_ms = (time.time() - start_time) * 1000
            await self.metrics.record_miss(query, "not_found")
            return CacheResult.create_miss(response_time_ms)

        except Exception as e:
            logger.error(f"Error querying cache: {e}")
            return CacheResult.create_miss((time.time() - start_time) * 1000)

    async def _try_exact_match(self, query: str, context: dict[str, Any] | None) -> CacheResult:
        """Try to find exact cache match."""
        # Generate normalized query key
        normalized_query = self.query_normalization.normalizer.normalize(query)
        cache_key = self._generate_cache_key(normalized_query, context)

        entry = await self.storage.get(cache_key)
        if entry is None or entry.is_expired():
            return CacheResult.create_miss()

        # Refresh TTL if needed
        if self.ttl_service.should_refresh_ttl(entry):
            refreshed_entry = entry.refresh_ttl()
            await self.storage.set(refreshed_entry)

        # Touch entry for LRU tracking
        touched_entry = entry.touch()
        await self.storage.set(touched_entry)

        return CacheResult.create_hit(entry.value, cache_key)

    async def _try_semantic_match(self, query: str) -> CacheResult:
        """Try to find semantic match."""
        semantic_match = await self.semantic_caching.find_applicable_cache(
            query, self.policy.semantic_match_threshold
        )

        if not semantic_match:
            return CacheResult.create_miss()

        # Verify the match is still valid
        entry = await self.storage.get(semantic_match.matched_entry_key)
        if entry is None or entry.is_expired():
            return CacheResult.create_miss()

        # Touch entry for LRU tracking
        touched_entry = entry.touch()
        await self.storage.set(touched_entry)

        return CacheResult.create_semantic_hit(
            entry.value,
            semantic_match.matched_entry_key,
            semantic_match.similarity_score,
            semantic_match.confidence,
        )

    @staticmethod
    def _generate_cache_key(query: str, context: dict[str, Any] | None) -> str:
        """Generate deterministic cache key."""
        import hashlib
        import json

        hasher = hashlib.sha256()
        hasher.update(query.encode("utf-8"))

        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            hasher.update(sorted_context.encode("utf-8"))

        return hasher.hexdigest()


class StoreCacheUseCase:
    """
    Use case for storing new cache entries.

    Handles:
    1. Eviction if necessary
    2. Embedding generation (if semantic caching enabled)
    3. Storage
    4. Semantic indexing
    """

    def __init__(
        self,
        storage: StoragePort,
        semantic_index: SemanticIndexPort,
        embedding_generator: EmbeddingGeneratorPort,
        metrics: CacheMetricsPort,
        cache_policy: CachePolicy,
    ):
        self.storage = storage
        self.semantic_caching = SemanticCachingService(semantic_index, embedding_generator)
        self.metrics = metrics
        self.policy = cache_policy
        self.eviction_service = CacheEvictionService(cache_policy, storage)

    async def execute(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Store cache entry."""
        try:
            # Check if eviction is necessary
            cache_size = await self.storage.get_size_bytes()
            entry_size = len(value)

            evicted_keys = await self.eviction_service.evict_if_necessary(cache_size, entry_size)
            for evicted_key in evicted_keys:
                await self.metrics.record_eviction(evicted_key, self.policy.eviction_policy.value)

            # Create cache entry
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None

            metadata = CacheMetadata(
                created_at=now,
                accessed_count=0,
                normalized_query=key,
                metadata={"context": context} if context else {},
            )

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
                context=context,
            )

            # Store entry
            await self.storage.set(entry)

            # Index semantically if enabled
            if self.policy.enable_semantic_caching:
                try:
                    await self.semantic_caching.index_entry_semantically(entry)
                except Exception as e:
                    logger.warning(f"Failed to index entry semantically: {e}")

        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            raise


class InvalidateCacheUseCase:
    """Use case for cache invalidation."""

    def __init__(
        self,
        storage: StoragePort,
        semantic_index: SemanticIndexPort,
        event_publisher: EventPublisherPort,
        metrics: CacheMetricsPort,
    ):
        self.invalidation_service = CacheInvalidationService(
            storage, semantic_index, event_publisher
        )
        self.metrics = metrics

    async def invalidate_key(self, cache_key: str, reason: str = "user_request") -> None:
        """Invalidate specific cache entry."""
        await self.invalidation_service.invalidate_key(cache_key, reason, "user")

    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with prefix."""
        return await self.invalidation_service.invalidate_by_prefix(prefix, "prefix_invalidation")

    async def purge_expired(self) -> int:
        """Purge all expired entries."""
        return await self.invalidation_service.purge_expired_entries()


class CacheMetricsUseCase:
    """Use case for cache metrics and reporting."""

    def __init__(self, metrics: CacheMetricsPort):
        self.metrics = metrics

    async def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        return await self.metrics.get_metrics()

    async def calculate_roi(self) -> float:
        """Calculate return on investment."""
        metrics = await self.metrics.get_metrics()

        total_operations = metrics.get("total_hits", 0) + metrics.get("total_misses", 0)
        if total_operations == 0:
            return 0.0

        total_cost_saved = metrics.get("total_cost_saved", 0.0)
        return total_cost_saved / total_operations if total_cost_saved > 0 else 0.0

    async def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        metrics = await self.metrics.get_metrics()
        total_hits = metrics.get("total_hits", 0)
        total_misses = metrics.get("total_misses", 0)

        total = total_hits + total_misses
        if total == 0:
            return 0.0

        return total_hits / total
