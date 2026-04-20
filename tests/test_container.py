"""Application-layer tests driven through the composition root.

These exercise the use cases with in-memory adapters so the domain +
application layers are covered without touching the filesystem. Per
2026 §3.2, application-layer tests mock ports only; we use the shipped
in-memory adapters as "fakes" rather than mocks.
"""

from __future__ import annotations

import pytest

from aicache.domain.models import (
    CacheEntry,
    CachePolicy,
    CacheResult,
    EvictionPolicy,
)
from aicache.infrastructure import build_container


@pytest.fixture
def policy() -> CachePolicy:
    return CachePolicy(
        max_size_bytes=10_000_000,
        default_ttl_seconds=None,
        eviction_policy=EvictionPolicy.LRU,
        semantic_match_threshold=0.85,
        enable_compression=False,
        enable_semantic_caching=False,
    )


@pytest.fixture
def container(policy):
    return build_container(in_memory=True, policy=policy)


@pytest.mark.asyncio
async def test_container_exposes_expected_use_cases(container):
    for name in ("query_cache", "store_cache", "invalidate_cache", "cache_metrics"):
        assert getattr(container, name) is not None, f"missing use case: {name}"


@pytest.mark.asyncio
async def test_store_then_query_returns_hit_for_identical_query(container):
    query = "What is the capital of France?"
    response = b"Paris"

    cache_key = container.query_cache._generate_cache_key(
        container.query_normalizer.normalize(query), None
    )
    await container.store_cache.execute(key=cache_key, value=response)

    result = await container.query_cache.execute(query)
    assert result.hit is True
    assert result.value == response


@pytest.mark.asyncio
async def test_query_returns_miss_when_not_stored(container):
    result = await container.query_cache.execute("never seen before")
    assert result.hit is False
    assert result.value is None


@pytest.mark.asyncio
async def test_metrics_track_hits_and_misses(container):
    query = "python sorting tutorial"
    response = b"sorted() returns a new list"

    # miss — records miss
    first = await container.query_cache.execute(query)
    assert first.hit is False

    # store + hit — records hit
    cache_key = container.query_cache._generate_cache_key(
        container.query_normalizer.normalize(query), None
    )
    await container.store_cache.execute(key=cache_key, value=response)
    second = await container.query_cache.execute(query)
    assert second.hit is True

    metrics = await container.cache_metrics.get_metrics()
    assert metrics["total_hits"] == 1
    assert metrics["total_misses"] == 1
    assert metrics["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_invalidate_removes_entry(container):
    query = "invalidation target"
    cache_key = container.query_cache._generate_cache_key(
        container.query_normalizer.normalize(query), None
    )
    await container.store_cache.execute(key=cache_key, value=b"boom")

    assert (await container.query_cache.execute(query)).hit is True

    await container.invalidate_cache.invalidate_key(cache_key)

    assert (await container.query_cache.execute(query)).hit is False


@pytest.mark.asyncio
async def test_context_scopes_cache_entries(container):
    query = "greet user"
    prod_ctx = {"env": "prod"}
    dev_ctx = {"env": "dev"}

    prod_key = container.query_cache._generate_cache_key(
        container.query_normalizer.normalize(query), prod_ctx
    )
    await container.store_cache.execute(key=prod_key, value=b"hi prod")

    prod_result = await container.query_cache.execute(query, context=prod_ctx)
    dev_result = await container.query_cache.execute(query, context=dev_ctx)

    assert prod_result.hit is True
    assert prod_result.value == b"hi prod"
    assert dev_result.hit is False


@pytest.mark.asyncio
async def test_storage_port_used_directly_for_admin_ops(container):
    """Admin path: presentation can read through the StoragePort.

    This test fails if the port contract ever diverges from what
    modern_cli's list/stats/inspect rely on.
    """
    entry = CacheEntry(
        key="raw-key",
        value=b"raw-value",
        created_at=__import__("datetime").datetime.now(),
    )
    await container.storage.set(entry)

    assert await container.storage.exists("raw-key") is True
    keys = await container.storage.get_all_keys()
    assert "raw-key" in keys
    fetched = await container.storage.get("raw-key")
    assert fetched is not None
    assert fetched.value == b"raw-value"


@pytest.mark.asyncio
async def test_container_swaps_adapters_for_tests():
    """Custom-injected adapter overrides the default — lets any test
    stub a single port without building everything from scratch."""
    from aicache.infrastructure.adapters import InMemoryStorageAdapter

    shared_storage = InMemoryStorageAdapter()
    c1 = build_container(in_memory=True, storage=shared_storage)
    c2 = build_container(in_memory=True, storage=shared_storage)

    cache_key = c1.query_cache._generate_cache_key(c1.query_normalizer.normalize("shared"), None)
    await c1.store_cache.execute(key=cache_key, value=b"same")

    fetched = await c2.query_cache.execute("shared")
    assert fetched.hit is True
    assert fetched.value == b"same"


def test_build_container_is_deterministic_given_same_policy(policy):
    c1 = build_container(in_memory=True, policy=policy)
    c2 = build_container(in_memory=True, policy=policy)
    assert c1.policy == c2.policy
    assert type(c1.storage) is type(c2.storage)


def test_cache_result_factory_methods_roundtrip():
    """Guard: use case paths rely on these helpers staying consistent."""
    hit = CacheResult.create_hit(b"v", "k", 1.0)
    miss = CacheResult.create_miss(2.0)
    sem = CacheResult.create_semantic_hit(b"v", "k", 0.9, 0.9, 3.0)

    assert hit.hit and hit.value == b"v" and hit.entry_key == "k"
    assert not miss.hit
    assert sem.hit and sem.similarity_score == 0.9
