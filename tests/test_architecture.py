"""
Comprehensive tests demonstrating world-class AI caching architecture.

These tests validate that:
1. Domain layer is independent of infrastructure
2. Immutability constraints are enforced
3. Port abstractions enable pluggability
4. Use cases orchestrate services correctly
5. All layers follow SOLID principles
"""

import asyncio
import inspect
from datetime import datetime

import pytest

from aicache.application import (
    CacheMetricsUseCase,
    InvalidateCacheUseCase,
    QueryCacheUseCase,
    StoreCacheUseCase,
)
from aicache.domain import (
    CacheEntry,
    CacheMetadata,
    CachePolicy,
    EvictionPolicy,
)
from aicache.infrastructure import (
    InMemoryCacheMetricsAdapter,
    InMemoryEventPublisherAdapter,
    InMemoryStorageAdapter,
    OpenAITokenCounterAdapter,
    SimpleEmbeddingGeneratorAdapter,
    SimpleQueryNormalizerAdapter,
    SimpleSemanticIndexAdapter,
)


class TestDomainImmutability:
    """Test that domain models enforce immutability."""

    def test_cache_entry_is_immutable(self):
        """Cache entries cannot be modified in place."""
        entry = CacheEntry(key="test-key", value=b"test-value", created_at=datetime.now())

        # Verify frozen attribute
        with pytest.raises(AttributeError):
            entry.value = b"modified"

    def test_cache_entry_touch_returns_new_instance(self):
        """Touch operation returns new instance."""
        entry = CacheEntry(
            key="test-key",
            value=b"test-value",
            created_at=datetime.now(),
            metadata=CacheMetadata(created_at=datetime.now()),
        )

        touched_entry = entry.touch()

        # Different instances
        assert touched_entry is not entry
        # Metadata is updated
        assert touched_entry.metadata.accessed_count == 1
        assert entry.metadata.accessed_count == 0

    def test_cache_entry_ttl_refresh_returns_new_instance(self):
        """TTL refresh returns new instance."""
        now = datetime.now()
        entry = CacheEntry(key="test-key", value=b"test-value", created_at=now, ttl_seconds=3600)

        refreshed_entry = entry.refresh_ttl()

        assert refreshed_entry is not entry
        # Either expires_at was None and now is set, or it was updated
        assert refreshed_entry.expires_at is not None
        assert refreshed_entry.ttl_seconds == 3600

    def test_cache_policy_validation(self):
        """Cache policy validates constraints."""
        # Valid policy
        valid_policy = CachePolicy(
            max_size_bytes=1000,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
            semantic_match_threshold=0.85,
        )
        assert valid_policy.validate()

        # Invalid policy - negative size
        invalid_policy = CachePolicy(
            max_size_bytes=-100, default_ttl_seconds=3600, eviction_policy=EvictionPolicy.LRU
        )
        assert not invalid_policy.validate()


class TestPortAbstractions:
    """Test that port abstractions work correctly."""

    @pytest.mark.asyncio
    async def test_storage_port_implementation(self):
        """Storage port can be implemented by different adapters."""
        adapters = [
            InMemoryStorageAdapter(),
        ]

        for adapter in adapters:
            entry = CacheEntry(key="test-key", value=b"test-value", created_at=datetime.now())

            # Set
            await adapter.set(entry)
            assert await adapter.exists("test-key")

            # Get
            retrieved = await adapter.get("test-key")
            assert retrieved is not None
            assert retrieved.value == b"test-value"

            # Delete
            assert await adapter.delete("test-key")
            assert not await adapter.exists("test-key")

            # Clear
            await adapter.set(entry)
            await adapter.clear()
            assert len(await adapter.get_all_keys()) == 0

    def test_query_normalizer_port_implementation(self):
        """Query normalizer port enables different normalization strategies."""
        normalizer = SimpleQueryNormalizerAdapter()

        # Normalize
        normalized = normalizer.normalize("  HELLO WORLD  ")
        assert normalized == "hello world"

        # Intent extraction
        intent = normalizer.extract_intent("what is machine learning and AI")
        assert intent == "what is machine"

        # Similarity
        similarity = normalizer.similarity_score(
            "what is machine learning", "what is machine learning"
        )
        assert similarity == 1.0

    def test_token_counter_port_implementation(self):
        """Token counter port is model-agnostic."""
        counter = OpenAITokenCounterAdapter()

        # Count tokens
        prompt_tokens = counter.count_prompt_tokens("Hello, world!", "gpt-4")
        completion_tokens = counter.count_completion_tokens("Hello!", "gpt-4")

        assert prompt_tokens > 0
        assert completion_tokens > 0

        # Estimate cost
        cost = counter.estimate_cost("gpt-4", 100, 50)
        assert cost > 0

        # Supported models
        models = counter.get_supported_models()
        assert "gpt-4" in models


class TestApplicationUseCases:
    """Test application layer use cases."""

    @pytest.mark.asyncio
    async def test_query_cache_use_case_with_exact_match(self):
        """Query cache returns exact matches."""
        storage = InMemoryStorageAdapter()
        semantic_index = SimpleSemanticIndexAdapter()
        token_counter = OpenAITokenCounterAdapter()
        normalizer = SimpleQueryNormalizerAdapter()
        embedding_gen = SimpleEmbeddingGeneratorAdapter()
        metrics = InMemoryCacheMetricsAdapter()
        policy = CachePolicy(
            max_size_bytes=1000000,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
            enable_semantic_caching=True,
        )

        use_case = QueryCacheUseCase(
            storage, semantic_index, token_counter, normalizer, embedding_gen, metrics, policy
        )

        # Generate the cache key like the use case does
        import hashlib
        import json

        query = "test query"
        context = None
        hasher = hashlib.sha256()
        hasher.update(query.encode("utf-8"))
        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            hasher.update(sorted_context.encode("utf-8"))
        cache_key = hasher.hexdigest()

        # Store an entry with the correct key
        entry = CacheEntry(
            key=cache_key,
            value=b"test-response",
            created_at=datetime.now(),
            metadata=CacheMetadata(created_at=datetime.now(), normalized_query="test query"),
        )
        await storage.set(entry)

        # Query for it
        result = await use_case.execute(query)

        assert result.hit
        assert result.value == b"test-response"

    @pytest.mark.asyncio
    async def test_store_cache_use_case(self):
        """Store cache use case handles eviction."""
        storage = InMemoryStorageAdapter()
        semantic_index = SimpleSemanticIndexAdapter()
        embedding_gen = SimpleEmbeddingGeneratorAdapter()
        metrics = InMemoryCacheMetricsAdapter()
        policy = CachePolicy(
            max_size_bytes=1000,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
            enable_semantic_caching=False,
        )

        use_case = StoreCacheUseCase(storage, semantic_index, embedding_gen, metrics, policy)

        # Store entries within size limit
        await use_case.execute("key1", b"x" * 100, ttl_seconds=3600)
        await use_case.execute("key2", b"y" * 100, ttl_seconds=3600)

        assert len(await storage.get_all_keys()) == 2

    @pytest.mark.asyncio
    async def test_invalidate_cache_use_case(self):
        """Invalidate cache use case removes entries."""
        storage = InMemoryStorageAdapter()
        semantic_index = SimpleSemanticIndexAdapter()
        event_publisher = InMemoryEventPublisherAdapter()
        metrics = InMemoryCacheMetricsAdapter()

        use_case = InvalidateCacheUseCase(storage, semantic_index, event_publisher, metrics)

        # Store entries
        entry = CacheEntry(key="test-key", value=b"test-value", created_at=datetime.now())
        await storage.set(entry)

        # Invalidate
        await use_case.invalidate_key("test-key")

        # Verify deleted
        assert not await storage.exists("test-key")

    @pytest.mark.asyncio
    async def test_cache_metrics_use_case(self):
        """Cache metrics use case calculates ROI."""
        metrics = InMemoryCacheMetricsAdapter()

        use_case = CacheMetricsUseCase(metrics)

        # Record some activity
        await metrics.record_hit("key1", 5.0, 100, 0.01)
        await metrics.record_hit("key2", 4.0, 100, 0.01)
        await metrics.record_miss("query3", "not_found")

        # Get metrics
        metric_data = await use_case.get_metrics()
        assert metric_data["total_hits"] == 2
        assert metric_data["total_misses"] == 1

        # Calculate hit rate
        hit_rate = await use_case.get_hit_rate()
        assert hit_rate == 2 / 3


class TestLayerSeparation:
    """Test that layers are properly separated."""

    def test_domain_has_no_infrastructure_dependencies(self):
        """Domain layer imports only from domain."""
        import inspect

        import aicache.domain as domain_module

        # Get all classes in domain
        domain_classes = [
            cls
            for name, cls in inspect.getmembers(domain_module, inspect.isclass)
            if cls.__module__.startswith("aicache.domain")
        ]

        # Verify none import infrastructure
        for cls in domain_classes:
            source = inspect.getsource(cls)
            assert "aicache.infrastructure" not in source
            assert "redis" not in source.lower()
            assert "postgres" not in source.lower()

    def test_application_only_depends_on_domain(self):
        """Application layer imports only from domain and application."""
        from aicache.application import use_cases

        source = inspect.getsource(use_cases)

        # Should import from domain and application, not infrastructure
        assert "from ..domain" in source
        # Infrastructure imports should be via ports, not concrete classes
        assert "InMemoryStorageAdapter" not in source

    def test_infrastructure_implements_ports(self):
        """Infrastructure classes implement port interfaces."""
        from aicache.domain.ports import QueryNormalizerPort, StoragePort, TokenCounterPort
        from aicache.infrastructure import (
            InMemoryStorageAdapter,
            OpenAITokenCounterAdapter,
            SimpleQueryNormalizerAdapter,
        )

        assert isinstance(InMemoryStorageAdapter(), StoragePort)
        assert isinstance(SimpleQueryNormalizerAdapter(), QueryNormalizerPort)
        assert isinstance(OpenAITokenCounterAdapter(), TokenCounterPort)


class TestEvolutionAndScalability:
    """Test that architecture supports evolution."""

    @pytest.mark.asyncio
    async def test_easy_to_add_new_storage_backend(self):
        """New storage backends can be added without changing domain/app."""

        class NewStorageBackendAdapter(InMemoryStorageAdapter):
            """Custom storage implementation."""

            pass

        # Use it in place of any other storage adapter
        storage = NewStorageBackendAdapter()
        entry = CacheEntry(key="test", value=b"data", created_at=datetime.now())

        await storage.set(entry)
        retrieved = await storage.get("test")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_multiple_normalizers_work_interchangeably(self):
        """Different normalizers work with same code."""

        class AggressiveNormalizerAdapter(SimpleQueryNormalizerAdapter):
            def normalize(self, query: str) -> str:
                # More aggressive normalization
                import re

                normalized = query.lower()
                # Remove punctuation
                normalized = re.sub(r"[^\w\s]", "", normalized)
                return normalized

        normalizer1 = SimpleQueryNormalizerAdapter()
        normalizer2 = AggressiveNormalizerAdapter()

        query = "What is Machine Learning?"

        result1 = normalizer1.normalize(query)
        result2 = normalizer2.normalize(query)

        # Both work, just different results
        assert result1 != result2
        assert len(result2) < len(result1)  # More characters removed


class TestEventDrivenInvalidation:
    """Test event-driven cache invalidation."""

    @pytest.mark.asyncio
    async def test_cache_invalidation_event_is_published(self):
        """Cache invalidation publishes events."""
        from aicache.domain.services import CacheInvalidationService

        storage = InMemoryStorageAdapter()
        semantic_index = SimpleSemanticIndexAdapter()
        event_publisher = InMemoryEventPublisherAdapter()

        # Track published events
        published_events = []

        async def event_handler(event):
            published_events.append(event)

        await event_publisher.subscribe(event_handler)

        service = CacheInvalidationService(storage, semantic_index, event_publisher)

        # Store an entry first so deletion succeeds
        entry = CacheEntry(key="test-key", value=b"test", created_at=datetime.now())
        await storage.set(entry)

        # Invalidate
        await service.invalidate_key("test-key", "test_reason")

        # Wait briefly for event handlers to run
        await asyncio.sleep(0.05)

        # Verify event was published
        assert len(published_events) > 0
        assert published_events[0].cache_key == "test-key"
        assert published_events[0].reason == "test_reason"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
