"""
Application Layer: TOON-Enhanced Use Cases

These are enhanced versions of the core use cases that integrate TOON generation
for comprehensive token optimization tracking and analytics.
"""

import hashlib
import logging
import time
import uuid
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
    TOONRepositoryPort,
)
from ..domain.services import (
    CacheEvictionService,
    CacheInvalidationService,
    CacheTTLService,
    QueryNormalizationService,
    SemanticCachingService,
    TokenCountingService,
)
from ..domain.toon_service import TOONAnalyticsService, TOONGenerationService

logger = logging.getLogger(__name__)


class TOONQueryCacheUseCase:
    """
    QueryCacheUseCase enhanced with TOON generation.

    This use case orchestrates multiple services to:
    1. Check for exact match
    2. Check for semantic match
    3. Generate TOON object with optimization metadata
    4. Return result or indicate miss
    5. Track metrics and TOON data
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
        toon_repository: TOONRepositoryPort,
    ):
        self.storage = storage
        self.semantic_caching = SemanticCachingService(semantic_index, embedding_generator)
        self.query_normalization = QueryNormalizationService(query_normalizer)
        self.token_counting = TokenCountingService(token_counter)
        self.metrics = metrics
        self.policy = cache_policy
        self.ttl_service = CacheTTLService()
        self.toon_generation = TOONGenerationService(storage, token_counter, metrics)
        self.toon_repository = toon_repository

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        model: str = "claude-3-opus",
        expected_prompt_tokens: int = 0,
    ) -> CacheResult:
        """
        Execute cache query with TOON generation.

        Args:
            query: User query string
            context: Optional context dictionary
            model: AI model name for token counting
            expected_prompt_tokens: Expected prompt tokens if cache miss occurs

        Returns:
            CacheResult with cache hit/miss information
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Generate query hash and normalized form
            normalized_query = self.query_normalization.normalizer.normalize(query)
            query_hash = self._generate_cache_key(normalized_query, context)

            # Step 1: Try exact match with normalization
            exact_result = await self._try_exact_match(query, context)
            if exact_result.hit:
                response_time_ms = (time.time() - start_time) * 1000
                cache_entry = await self.storage.get(exact_result.entry_key or query_hash)

                # Generate TOON for cache hit
                toon = await self.toon_generation.generate_toon_from_cache_hit(
                    operation_id=operation_id,
                    original_query=query,
                    normalized_query=normalized_query,
                    query_hash=query_hash,
                    cache_result=exact_result,
                    cache_entry=cache_entry,
                    prompt_tokens_without_cache=expected_prompt_tokens,
                    model=model,
                    duration_ms=response_time_ms,
                    semantic_match=False,
                    context=context,
                )

                # Save TOON
                await self.toon_repository.save_toon(toon)

                # Record metrics
                await self.metrics.record_hit(
                    exact_result.entry_key or "",
                    response_time_ms,
                    tokens_saved=toon.token_delta.saved_total,
                    cost_saved=toon.token_delta.cost_saved,
                )

                logger.debug(
                    f"Cache hit (exact): {expected_prompt_tokens} tokens saved, "
                    f"${toon.token_delta.cost_saved:.6f} cost saved"
                )
                return CacheResult.create_hit(
                    exact_result.value, exact_result.entry_key, response_time_ms
                )

            # Step 2: Try semantic match if enabled
            if self.policy.enable_semantic_caching:
                semantic_result = await self._try_semantic_match(query)
                if (
                    semantic_result.hit
                    and semantic_result.confidence
                    and semantic_result.confidence > 0.85
                ):
                    response_time_ms = (time.time() - start_time) * 1000
                    cache_entry = await self.storage.get(semantic_result.entry_key or query_hash)

                    # Generate TOON for semantic hit
                    toon = await self.toon_generation.generate_toon_from_cache_hit(
                        operation_id=operation_id,
                        original_query=query,
                        normalized_query=normalized_query,
                        query_hash=query_hash,
                        cache_result=semantic_result,
                        cache_entry=cache_entry,
                        prompt_tokens_without_cache=expected_prompt_tokens,
                        model=model,
                        duration_ms=response_time_ms,
                        semantic_match=True,
                        context=context,
                    )

                    # Save TOON
                    await self.toon_repository.save_toon(toon)

                    # Record metrics
                    await self.metrics.record_hit(
                        semantic_result.entry_key or "",
                        response_time_ms,
                        tokens_saved=toon.token_delta.saved_total,
                        cost_saved=toon.token_delta.cost_saved,
                    )

                    logger.debug(
                        f"Cache hit (semantic): {semantic_result.similarity_score:.2f} similarity, "
                        f"{expected_prompt_tokens} tokens saved"
                    )
                    return semantic_result

            # Step 3: Cache miss
            response_time_ms = (time.time() - start_time) * 1000

            # Generate TOON for cache miss
            toon = await self.toon_generation.generate_toon_from_cache_miss(
                operation_id=operation_id,
                original_query=query,
                normalized_query=normalized_query,
                query_hash=query_hash,
                prompt_tokens=expected_prompt_tokens,
                completion_tokens=0,
                model=model,
                duration_ms=response_time_ms,
                semantic_attempted=self.policy.enable_semantic_caching,
                context=context,
            )

            # Save TOON
            await self.toon_repository.save_toon(toon)

            # Record metrics
            await self.metrics.record_miss(query, "not_found")

            logger.debug(f"Cache miss: {expected_prompt_tokens} tokens will be charged")
            return CacheResult.create_miss(response_time_ms)

        except Exception as e:
            logger.error(f"Error querying cache: {e}", exc_info=True)
            return CacheResult.create_miss((time.time() - start_time) * 1000)

    async def _try_exact_match(self, query: str, context: dict[str, Any] | None) -> CacheResult:
        """Try to find exact cache match."""
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
        import json

        hasher = hashlib.sha256()
        hasher.update(query.encode("utf-8"))

        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            hasher.update(sorted_context.encode("utf-8"))

        return hasher.hexdigest()


class TOONStoreCacheUseCase:
    """
    StoreCacheUseCase enhanced with TOON tracking.

    Handles:
    1. Eviction if necessary (with TOON tracking)
    2. Embedding generation
    3. Storage
    4. Semantic indexing
    5. TOON generation for store operation
    """

    def __init__(
        self,
        storage: StoragePort,
        semantic_index: SemanticIndexPort,
        embedding_generator: EmbeddingGeneratorPort,
        metrics: CacheMetricsPort,
        cache_policy: CachePolicy,
        toon_repository: TOONRepositoryPort,
    ):
        self.storage = storage
        self.semantic_caching = SemanticCachingService(semantic_index, embedding_generator)
        self.metrics = metrics
        self.policy = cache_policy
        self.eviction_service = CacheEvictionService(cache_policy, storage)
        self.toon_repository = toon_repository

    async def execute(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
        context: dict[str, Any] | None = None,
        query: str | None = None,
        model: str = "claude-3-opus",
    ) -> None:
        """
        Store cache entry with TOON tracking.

        Args:
            key: Cache key
            value: Response value to cache
            ttl_seconds: Time to live in seconds
            context: Optional context
            query: Original query string (for TOON)
            model: AI model name
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check if eviction is necessary
            cache_size = await self.storage.get_size_bytes()
            entry_size = len(value)

            evicted_keys = await self.eviction_service.evict_if_necessary(cache_size, entry_size)
            for evicted_key in evicted_keys:
                await self.metrics.record_eviction(evicted_key, self.policy.eviction_policy.value)
                logger.debug(f"Evicted cache entry: {evicted_key}")

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
                    logger.debug(f"Indexed entry semantically: {key}")
                except Exception as e:
                    logger.warning(f"Failed to index entry semantically: {e}")

            # Generate TOON for store operation
            if query:
                normalized_query = query.lower().strip()
                response_time_ms = (time.time() - start_time) * 1000

                toon = await self._create_store_toon(
                    operation_id=operation_id,
                    query=query,
                    normalized_query=normalized_query,
                    model=model,
                    response_size_bytes=entry_size,
                    ttl_seconds=ttl_seconds,
                    duration_ms=response_time_ms,
                    context=context,
                )

                await self.toon_repository.save_toon(toon)

                logger.debug(
                    f"Cached query: {len(value)} bytes stored, "
                    f"TTL: {ttl_seconds}s, "
                    f"Evicted: {len(evicted_keys)} entries"
                )

        except Exception as e:
            logger.error(f"Error storing cache entry: {e}", exc_info=True)
            raise

    async def _create_store_toon(
        self,
        operation_id: str,
        query: str,
        normalized_query: str,
        model: str,
        response_size_bytes: int,
        ttl_seconds: int | None,
        duration_ms: float,
        context: dict[str, Any] | None,
    ):
        """Create a TOON object for a store operation."""
        from ..domain.toon import (
            TOONCacheMetadata,
            TOONCacheOperation,
            TOONOperationType,
            TOONOptimizationInsight,
            TOONOptimizationLevel,
            TOONQueryMetadata,
            TOONSemanticMatchData,
            TOONStrategy,
            TOONTokenDelta,
        )

        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()

        query_metadata = TOONQueryMetadata(
            original_query=query,
            normalized_query=normalized_query,
            query_hash=query_hash,
        )

        # Token delta for store (not applicable, use zeros)
        token_delta = TOONTokenDelta(
            without_cache_prompt=0,
            without_cache_completion=0,
            without_cache_total=0,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=0,
            saved_completion=0,
            saved_total=0,
            saved_percent=0.0,
            cost_without_cache=0.0,
            cost_with_cache=0.0,
            cost_saved=0.0,
            model=model,
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.0,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key=query_hash,
            cache_age_seconds=0,
            ttl_remaining_seconds=ttl_seconds,
            access_count=0,
            last_accessed=None,
            created_at=datetime.now(),
            memory_size_bytes=response_size_bytes,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.NONE,
            roi_score=0.0,
            suggested_actions=["cache_stored"],
            eviction_risk="low",
            cache_efficiency_score=0.0,
            predictability_score=0.0,
            pattern_detected=False,
            similar_queries_found=0,
        )

        return TOONCacheOperation(
            operation_id=operation_id,
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_MISS,  # Treat store as a "miss" that gets cached
            strategy_used=TOONStrategy.NONE,
            duration_ms=duration_ms,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
            context=context,
        )


class TOONInvalidateCacheUseCase:
    """Use case for cache invalidation with TOON tracking."""

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
        logger.info(f"Invalidating cache key: {cache_key}, reason: {reason}")
        await self.invalidation_service.invalidate_key(cache_key, reason, "user")

    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with prefix."""
        logger.info(f"Invalidating cache entries with prefix: {prefix}")
        count = await self.invalidation_service.invalidate_by_prefix(prefix, "prefix_invalidation")
        logger.info(f"Invalidated {count} cache entries")
        return count

    async def purge_expired(self) -> int:
        """Purge all expired entries."""
        count = await self.invalidation_service.purge_expired_entries()
        logger.info(f"Purged {count} expired cache entries")
        return count


class TOONCacheMetricsUseCase:
    """Use case for cache metrics and TOON analytics."""

    def __init__(self, metrics: CacheMetricsPort, toon_repository: TOONRepositoryPort):
        self.metrics = metrics
        self.toon_repository = toon_repository
        self.analytics_service = TOONAnalyticsService()

    async def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        return await self.metrics.get_metrics()

    async def get_toon_analytics(self, limit: int | None = None, days: int = 1) -> dict[str, Any]:
        """
        Get TOON analytics for specified period.

        Args:
            limit: Maximum number of TOON operations to analyze
            days: Number of days of history to analyze

        Returns:
            Analytics dictionary with insights and metrics
        """
        start_time = datetime.now() - timedelta(days=days)
        end_time = datetime.now()

        toons = await self.toon_repository.get_all_toons()
        toons = [t for t in toons if start_time <= t.timestamp <= end_time]

        if limit:
            toons = toons[-limit:]

        # Aggregate analytics
        analytics = self.analytics_service.aggregate_toons(toons, start_time, end_time)
        insights = self.analytics_service.extract_insights(analytics)

        return {
            "period_days": days,
            "analytics": analytics.to_dict(),
            "insights": insights,
            "toon_count": len(toons),
        }

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
