"""
Domain Services: Core business logic for AI caching.

These services orchestrate domain objects and implement key business rules
without depending on infrastructure details.
"""

import logging
from datetime import datetime

from .models import (
    CacheEntry,
    CacheInvalidationEvent,
    CachePolicy,
    EvictionPolicy,
    InvalidationStrategy,
    SemanticMatch,
    TokenUsageMetrics,
)
from .ports import (
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
)

logger = logging.getLogger(__name__)


class QueryNormalizationService:
    """Handles query normalization and intent extraction."""

    def __init__(self, normalizer: QueryNormalizerPort):
        self.normalizer = normalizer

    def should_use_cached_response(
        self, query: str, cached_entry: CacheEntry, similarity_threshold: float = 0.85
    ) -> bool:
        """Determine if cached entry applies to new query."""
        if cached_entry.metadata is None or cached_entry.metadata.normalized_query is None:
            return False

        # Exact match after normalization
        normalized_new = self.normalizer.normalize(query)
        if normalized_new == cached_entry.metadata.normalized_query:
            return True

        # Intent-based matching
        intent_new = self.normalizer.extract_intent(query)
        if intent_new and "intent" in cached_entry.metadata.metadata:
            cached_intent = cached_entry.metadata.metadata.get("intent")
            if intent_new == cached_intent:
                return True

        # Similarity matching
        similarity = self.normalizer.similarity_score(query, cached_entry.metadata.normalized_query)
        return similarity >= similarity_threshold


class TokenCountingService:
    """Manages token accounting and cost tracking."""

    def __init__(self, counter: TokenCounterPort):
        self.counter = counter

    def calculate_tokens(self, prompt: str, completion: str, model: str) -> TokenUsageMetrics:
        """Calculate token usage for a query-response pair."""
        prompt_tokens = self.counter.count_prompt_tokens(prompt, model)
        completion_tokens = self.counter.count_completion_tokens(completion, model)
        total_tokens = prompt_tokens + completion_tokens

        cost = self.counter.estimate_cost(model, prompt_tokens, completion_tokens)

        return TokenUsageMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
        )

    def calculate_savings(self, cache_hit: bool, tokens_without_cache: int, model: str) -> float:
        """Calculate cost savings from cache hit."""
        if not cache_hit:
            return 0.0
        return self.counter.estimate_cost(model, tokens_without_cache, 0)


class SemanticCachingService:
    """Manages semantic similarity matching and caching."""

    def __init__(
        self, semantic_index: SemanticIndexPort, embedding_generator: EmbeddingGeneratorPort
    ):
        self.semantic_index = semantic_index
        self.embedding_generator = embedding_generator

    async def find_applicable_cache(
        self, query: str, min_similarity: float = 0.85
    ) -> SemanticMatch | None:
        """Find cache entry with semantic similarity."""
        embeddings = await self.embedding_generator.generate_embedding(query)
        matches = await self.semantic_index.find_similar(embeddings, min_similarity)

        if not matches:
            return None

        # Return best match
        best_match = max(matches, key=lambda m: m.similarity_score)
        return best_match

    async def index_entry_semantically(self, entry: CacheEntry) -> None:
        """Add entry to semantic index."""
        if entry.embedding is None:
            # Generate embedding if not present
            query = entry.metadata.normalized_query if entry.metadata else entry.key
            entry_embedding = await self.embedding_generator.generate_embedding(query)
        else:
            entry_embedding = entry.embedding

        metadata = {
            "key": entry.key,
            "normalized_query": entry.metadata.normalized_query if entry.metadata else None,
            "created_at": entry.created_at.isoformat(),
        }

        await self.semantic_index.index_embedding(entry.key, entry_embedding, metadata)


class CacheEvictionService:
    """Manages cache eviction policies and enforcement."""

    def __init__(self, policy: CachePolicy, storage: StoragePort):
        self.policy = policy
        self.storage = storage

    async def evict_if_necessary(self, current_size: int, new_entry_size: int) -> list[str]:
        """Evict entries if cache size exceeded."""
        if current_size + new_entry_size <= self.policy.max_size_bytes:
            return []

        evicted_keys = []
        space_needed = (current_size + new_entry_size) - self.policy.max_size_bytes

        if self.policy.eviction_policy == EvictionPolicy.LRU:
            evicted_keys = await self._evict_lru(space_needed)
        elif self.policy.eviction_policy == EvictionPolicy.LFU:
            evicted_keys = await self._evict_lfu(space_needed)
        elif self.policy.eviction_policy == EvictionPolicy.FIFO:
            evicted_keys = await self._evict_fifo(space_needed)

        return evicted_keys

    async def _evict_lru(self, space_needed: int) -> list[str]:
        """Evict least recently used entries."""
        keys = await self.storage.get_all_keys()
        entries = []
        for key in keys:
            entry = await self.storage.get(key)
            if entry:
                entries.append((key, entry))

        # Sort by last_accessed (oldest first)
        entries.sort(
            key=lambda x: (
                x[1].metadata.last_accessed_at
                if x[1].metadata and x[1].metadata.last_accessed_at
                else x[1].created_at
            )
        )

        evicted = []
        freed_space = 0
        for key, entry in entries:
            if freed_space >= space_needed:
                break
            await self.storage.delete(key)
            freed_space += entry.get_size_bytes()
            evicted.append(key)

        return evicted

    async def _evict_lfu(self, space_needed: int) -> list[str]:
        """Evict least frequently used entries."""
        keys = await self.storage.get_all_keys()
        entries = []
        for key in keys:
            entry = await self.storage.get(key)
            if entry:
                entries.append((key, entry))

        # Sort by access_count (least accessed first)
        entries.sort(key=lambda x: x[1].metadata.accessed_count if x[1].metadata else 0)

        evicted = []
        freed_space = 0
        for key, entry in entries:
            if freed_space >= space_needed:
                break
            await self.storage.delete(key)
            freed_space += entry.get_size_bytes()
            evicted.append(key)

        return evicted

    async def _evict_fifo(self, space_needed: int) -> list[str]:
        """Evict first-in-first-out entries."""
        keys = await self.storage.get_all_keys()
        entries = []
        for key in keys:
            entry = await self.storage.get(key)
            if entry:
                entries.append((key, entry))

        # Sort by created_at (oldest first)
        entries.sort(key=lambda x: x[1].created_at)

        evicted = []
        freed_space = 0
        for key, entry in entries:
            if freed_space >= space_needed:
                break
            await self.storage.delete(key)
            freed_space += entry.get_size_bytes()
            evicted.append(key)

        return evicted


class CacheInvalidationService:
    """Manages cache invalidation with traceability."""

    def __init__(
        self,
        storage: StoragePort,
        semantic_index: SemanticIndexPort,
        event_publisher: EventPublisherPort,
    ):
        self.storage = storage
        self.semantic_index = semantic_index
        self.event_publisher = event_publisher

    async def invalidate_key(
        self, cache_key: str, reason: str = "explicit", triggered_by: str = "user"
    ) -> None:
        """Invalidate specific cache entry."""
        success = await self.storage.delete(cache_key)

        if success:
            await self.semantic_index.remove_embedding(cache_key)
            event = CacheInvalidationEvent(
                cache_key=cache_key,
                reason=reason,
                triggered_by=triggered_by,
                timestamp=datetime.now(),
                strategy=InvalidationStrategy.IMMEDIATE,
                affected_entries=1,
            )
            await self.event_publisher.publish(event)

    async def invalidate_semantic_neighbors(
        self, query: str, threshold: float = 0.75, reason: str = "semantic_invalidation"
    ) -> int:
        """Invalidate semantically similar cache entries."""
        # This would require the semantic index to support reverse lookup
        # Implementation depends on index capabilities
        logger.info(f"Semantic invalidation: {query} with threshold {threshold}")
        return 0

    async def invalidate_by_prefix(self, prefix: str, reason: str = "prefix_match") -> int:
        """Invalidate all entries with key prefix."""
        keys = await self.storage.get_all_keys()
        invalidated = 0

        for key in keys:
            if key.startswith(prefix):
                await self.storage.delete(key)
                await self.semantic_index.remove_embedding(key)
                invalidated += 1

        if invalidated > 0:
            event = CacheInvalidationEvent(
                cache_key=prefix,
                reason=reason,
                triggered_by="system",
                timestamp=datetime.now(),
                strategy=InvalidationStrategy.IMMEDIATE,
                affected_entries=invalidated,
            )
            await self.event_publisher.publish(event)

        return invalidated

    async def purge_expired_entries(self) -> int:
        """Remove all expired cache entries."""
        keys = await self.storage.get_all_keys()
        purged = 0

        for key in keys:
            entry = await self.storage.get(key)
            if entry and entry.is_expired():
                await self.storage.delete(key)
                await self.semantic_index.remove_embedding(key)
                purged += 1

        if purged > 0:
            event = CacheInvalidationEvent(
                cache_key="",
                reason="ttl_expiration",
                triggered_by="system",
                timestamp=datetime.now(),
                strategy=InvalidationStrategy.IMMEDIATE,
                affected_entries=purged,
            )
            await self.event_publisher.publish(event)

        return purged


class CacheTTLService:
    """Manages TTL (time-to-live) enforcement."""

    @staticmethod
    def get_expiration_time(ttl_seconds: int | None) -> datetime | None:
        """Calculate expiration time from TTL."""
        if ttl_seconds is None:
            return None
        from datetime import timedelta

        return datetime.now() + timedelta(seconds=ttl_seconds)

    @staticmethod
    def should_refresh_ttl(entry: CacheEntry) -> bool:
        """Determine if TTL should be refreshed."""
        if entry.ttl_seconds is None:
            return False
        if entry.expires_at is None:
            return False

        # Refresh if less than 10% of TTL remaining
        age = entry.calculate_age_seconds()
        return age > (entry.ttl_seconds * 0.9)
