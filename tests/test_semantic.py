"""Tests for the semantic cache path.

We drive the ``QueryCacheUseCase`` / ``StoreCacheUseCase`` wiring with
controllable fake adapters — that exercises the real semantic branch
without dragging in chromadb / sentence-transformers. A real-adapter
integration test is gated on the extras being installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from aicache.domain.models import CachePolicy, EvictionPolicy, SemanticMatch
from aicache.domain.ports import EmbeddingGeneratorPort, SemanticIndexPort
from aicache.infrastructure import build_container

# ---------------------------------------------------------------------
# Fake semantic infrastructure — fully deterministic
# ---------------------------------------------------------------------


class _KeywordEmbedder(EmbeddingGeneratorPort):
    """Returns identical vectors for texts that share a keyword.

    Lets us assert semantic matches without ML dependencies: if two
    prompts both contain "sort", they get the same vector and therefore
    perfect cosine similarity.
    """

    KEYWORDS = ("sort", "format", "print", "parse")

    async def generate_embedding(self, text: str) -> list[float]:
        text = text.lower()
        vec = [1.0 if kw in text else 0.0 for kw in self.KEYWORDS]
        # Avoid all-zero vectors (undefined cosine similarity)
        if all(v == 0.0 for v in vec):
            vec = [0.1] + [0.0] * (len(self.KEYWORDS) - 1)
        return vec

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [await self.generate_embedding(t) for t in texts]

    def get_embedding_dimension(self) -> int:
        return len(self.KEYWORDS)


class _InMemoryCosineIndex(SemanticIndexPort):
    """Cosine-similarity index using only stdlib."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict[str, Any]]] = {}

    async def index_embedding(
        self, key: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        self._store[key] = (embedding, metadata)

    async def find_similar(
        self, embedding: list[float], threshold: float = 0.85
    ) -> list[SemanticMatch]:
        matches = []
        for key, (vec, _) in self._store.items():
            sim = _cosine(embedding, vec)
            if sim >= threshold:
                matches.append(
                    SemanticMatch(similarity_score=sim, matched_entry_key=key, confidence=sim)
                )
        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    async def remove_embedding(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def clear(self) -> None:
        self._store.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _semantic_policy() -> CachePolicy:
    return CachePolicy(
        max_size_bytes=10_000_000,
        default_ttl_seconds=None,
        eviction_policy=EvictionPolicy.LRU,
        semantic_match_threshold=0.85,
        enable_compression=False,
        enable_semantic_caching=True,
    )


@pytest.fixture
def container():
    return build_container(
        in_memory=True,
        policy=_semantic_policy(),
        semantic_index=_InMemoryCosineIndex(),
        embedding_generator=_KeywordEmbedder(),
    )


# ---------------------------------------------------------------------
# Behavioural assertions
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rephrased_query_hits_semantic_cache(container):
    await container.store_cache.execute(
        key="k1",
        value=b"use sorted() or list.sort()",
        query_text="How do I sort a list in Python?",
    )

    result = await container.query_cache.execute(
        "python sort list example",
        query_text="python sort list example",
    )
    assert result.hit is True
    assert result.value == b"use sorted() or list.sort()"
    assert result.similarity_score is not None
    assert result.similarity_score >= 0.85


@pytest.mark.asyncio
async def test_unrelated_query_is_a_miss(container):
    await container.store_cache.execute(
        key="k1",
        value=b"irrelevant",
        query_text="How do I sort a list?",
    )
    result = await container.query_cache.execute(
        "format a datetime",
        query_text="format a datetime",
    )
    assert result.hit is False


@pytest.mark.asyncio
async def test_semantic_caching_disabled_in_policy_skips_the_branch():
    # Identical infra but with semantic flag off
    policy = CachePolicy(
        max_size_bytes=10_000_000,
        default_ttl_seconds=None,
        eviction_policy=EvictionPolicy.LRU,
        semantic_match_threshold=0.85,
        enable_compression=False,
        enable_semantic_caching=False,
    )
    c = build_container(
        in_memory=True,
        policy=policy,
        semantic_index=_InMemoryCosineIndex(),
        embedding_generator=_KeywordEmbedder(),
    )
    await c.store_cache.execute(key="k1", value=b"v", query_text="How do I sort a list?")

    # Even though embeddings would match, semantic path is disabled.
    result = await c.query_cache.execute(
        "python sort list example", query_text="python sort list example"
    )
    assert result.hit is False


@pytest.mark.asyncio
async def test_semantic_match_tracked_in_metrics(container):
    await container.store_cache.execute(key="k1", value=b"v", query_text="sort this list")

    await container.query_cache.execute("python sort tutorial", query_text="python sort tutorial")

    metrics = await container.cache_metrics.get_metrics()
    assert metrics["total_hits"] == 1


# ---------------------------------------------------------------------
# Integration test with real adapters — skipped unless installed
# ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_chroma_plus_sentence_transformers_end_to_end():
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")

    container = build_container(in_memory=True, semantic=True)

    await container.store_cache.execute(
        key="k1",
        value=b"Use sorted() for a new list or list.sort() in place.",
        query_text="How do I sort a list in Python?",
    )

    result = await container.query_cache.execute(
        "python sort list example",
        query_text="python sort list example",
    )
    assert result.hit is True
