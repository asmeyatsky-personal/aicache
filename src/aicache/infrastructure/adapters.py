"""
Infrastructure Layer: Adapter Implementations

These adapters implement the port interfaces defined in the domain layer.
They handle interactions with external services and storage backends.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..domain.models import CacheEntry, CacheInvalidationEvent, SemanticMatch
from ..domain.ports import (
    CacheMetricsPort,
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
)

logger = logging.getLogger(__name__)


class InMemoryStorageAdapter(StoragePort):
    """In-memory cache storage adapter."""

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cache entry by key."""
        return self._cache.get(key)

    async def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        self._cache[entry.key] = entry

    async def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if cache entry exists."""
        return key in self._cache

    async def get_all_keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self._cache.keys())

    async def get_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry.get_size_bytes() for entry in self._cache.values())

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


class FileSystemStorageAdapter(StoragePort):
    """File system-based cache storage adapter."""

    def __init__(self, cache_dir: str = "~/.cache/aicache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cache entry by key."""
        cache_file = self.cache_dir / key
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
                return self._deserialize_entry(data)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error reading cache file {key}: {e}")
            return None

    async def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        cache_file = self.cache_dir / entry.key
        try:
            with open(cache_file, "w") as f:
                json.dump(self._serialize_entry(entry), f)
        except OSError as e:
            logger.error(f"Error writing cache file {entry.key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        cache_file = self.cache_dir / key
        if cache_file.exists():
            try:
                cache_file.unlink()
                return True
            except OSError as e:
                logger.error(f"Error deleting cache file {key}: {e}")
        return False

    async def exists(self, key: str) -> bool:
        """Check if cache entry exists."""
        return (self.cache_dir / key).exists()

    async def get_all_keys(self) -> list[str]:
        """Get all cache keys."""
        return [f.name for f in self.cache_dir.iterdir() if f.is_file()]

    async def get_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        return sum(f.stat().st_size for f in self.cache_dir.iterdir() if f.is_file())

    async def clear(self) -> None:
        """Clear all cache entries."""
        for f in self.cache_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError as e:
                    logger.error(f"Error clearing cache file {f.name}: {e}")

    @staticmethod
    def _serialize_entry(entry: CacheEntry) -> dict[str, Any]:
        """Serialize cache entry to JSON."""
        return {
            "key": entry.key,
            "value": entry.value.hex() if entry.value else None,
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "ttl_seconds": entry.ttl_seconds,
            "context": entry.context,
        }

    @staticmethod
    def _deserialize_entry(data: dict[str, Any]) -> CacheEntry:
        """Deserialize cache entry from JSON."""
        return CacheEntry(
            key=data["key"],
            value=bytes.fromhex(data["value"]) if data.get("value") else b"",
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            ttl_seconds=data.get("ttl_seconds"),
            context=data.get("context"),
        )


class SimpleQueryNormalizerAdapter(QueryNormalizerPort):
    """Simple query normalization adapter."""

    def normalize(self, query: str) -> str:
        """Normalize query for comparison."""
        return query.lower().strip()

    def extract_intent(self, query: str) -> str:
        """Extract semantic intent from query."""
        # Simple implementation: take first few words
        words = query.lower().split()
        return " ".join(words[:3]) if words else ""

    def similarity_score(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        # Simple Jaccard similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return 1.0 if query1 == query2 else 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union


class OpenAITokenCounterAdapter(TokenCounterPort):
    """OpenAI token counter adapter."""

    # Simplified token counts (real implementation would use tiktoken)
    def count_prompt_tokens(self, text: str, model: str) -> int:
        """Count tokens in prompt text."""
        # Rough approximation: 1 token ~ 4 characters
        return len(text) // 4

    def count_completion_tokens(self, text: str, model: str) -> int:
        """Count tokens in completion text."""
        return len(text) // 4

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate API cost for token usage."""
        # Pricing varies by model, this is approximate for GPT-4
        if "gpt-4" in model.lower():
            prompt_cost = prompt_tokens * 0.00003  # $0.03 per 1K tokens
            completion_cost = completion_tokens * 0.0006  # $0.06 per 1K tokens
        else:
            prompt_cost = prompt_tokens * 0.0000005  # $0.5 per 1M tokens
            completion_cost = completion_tokens * 0.0000015  # $1.5 per 1M tokens

        return prompt_cost + completion_cost

    def get_supported_models(self) -> list[str]:
        """Get list of supported models."""
        return ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]


class InMemoryEventPublisherAdapter(EventPublisherPort):
    """In-memory event publisher adapter."""

    def __init__(self):
        self._subscribers: list = []

    async def publish(self, event: CacheInvalidationEvent) -> None:
        """Publish a cache invalidation event."""
        logger.info(f"Publishing event: {event.cache_key} - {event.reason}")
        for handler in self._subscribers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error handling event: {e}")

    async def subscribe(self, handler) -> None:
        """Subscribe to cache invalidation events."""
        self._subscribers.append(handler)


class InMemoryCacheMetricsAdapter(CacheMetricsPort):
    """In-memory cache metrics adapter."""

    def __init__(self):
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._response_times = []
        self._tokens_saved = 0
        self._cost_saved = 0.0
        self._hit_times = []
        self._memory_used = 0
        self._semantic_matches = 0
        self._false_positives = 0

    async def record_hit(
        self, entry_key: str, response_time_ms: float, tokens_saved: int, cost_saved: float
    ) -> None:
        """Record a cache hit."""
        self._hits += 1
        self._response_times.append(response_time_ms)
        self._tokens_saved += tokens_saved
        self._cost_saved += cost_saved
        self._hit_times.append(datetime.now())

    async def record_miss(self, query: str, reason: str) -> None:
        """Record a cache miss."""
        self._misses += 1

    async def record_eviction(self, entry_key: str, policy: str) -> None:
        """Record a cache eviction."""
        self._evictions += 1

    async def get_metrics(self) -> dict[str, Any]:
        """Get aggregate metrics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        avg_response_time = (
            sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
        )

        return {
            "total_hits": self._hits,
            "total_misses": self._misses,
            "total_evictions": self._evictions,
            "hit_rate": hit_rate,
            "average_response_time_ms": avg_response_time,
            "total_tokens_saved": self._tokens_saved,
            "total_cost_saved": self._cost_saved,
            "memory_usage_bytes": self._memory_used,
            "semantic_matches": self._semantic_matches,
            "false_positives": self._false_positives,
        }


class SimpleSemanticIndexAdapter(SemanticIndexPort):
    """Simple in-memory semantic index adapter."""

    def __init__(self):
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    async def index_embedding(
        self, key: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Index an embedding with metadata."""
        self._embeddings[key] = embedding
        self._metadata[key] = metadata

    async def find_similar(
        self, embedding: list[float], threshold: float = 0.85
    ) -> list[SemanticMatch]:
        """Find semantically similar indexed embeddings."""
        matches = []

        for key, indexed_embedding in self._embeddings.items():
            similarity = self._cosine_similarity(embedding, indexed_embedding)

            if similarity >= threshold:
                matches.append(
                    SemanticMatch(
                        similarity_score=similarity, matched_entry_key=key, confidence=similarity
                    )
                )

        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    async def remove_embedding(self, key: str) -> bool:
        """Remove an embedding from the index."""
        if key in self._embeddings:
            del self._embeddings[key]
            if key in self._metadata:
                del self._metadata[key]
            return True
        return False

    async def clear(self) -> None:
        """Clear all embeddings."""
        self._embeddings.clear()
        self._metadata.clear()

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = sum(a**2 for a in vec1) ** 0.5
        magnitude2 = sum(b**2 for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class SimpleEmbeddingGeneratorAdapter(EmbeddingGeneratorPort):
    """Simple embedding generator adapter (returns random embeddings)."""

    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Simple hash-based embedding (not semantically meaningful)
        import hashlib

        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        embeddings = []
        for i in range(self._dimension):
            # Generate pseudo-random values from hash
            embeddings.append((hash_int >> i) % 1000 / 1000.0)

        return embeddings

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts]

    def get_embedding_dimension(self) -> int:
        """Get dimensionality of embeddings."""
        return self._dimension
