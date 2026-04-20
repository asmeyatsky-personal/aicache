"""
Ports (Interface Abstractions) for external dependencies.

These ports define contracts that infrastructure adapters must implement.
This separation ensures domain logic is independent of specific technologies.
"""

from abc import ABC, abstractmethod
from typing import Any

from .models import CacheEntry, CacheInvalidationEvent, SemanticMatch


class StoragePort(ABC):
    """Port for persistent cache storage."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cache entry by key."""
        pass

    @abstractmethod
    async def set(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if cache entry exists."""
        pass

    @abstractmethod
    async def get_all_keys(self) -> list[str]:
        """Get all cache keys."""
        pass

    @abstractmethod
    async def get_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass


class SemanticIndexPort(ABC):
    """Port for semantic similarity indexing and search."""

    @abstractmethod
    async def index_embedding(
        self, key: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Index an embedding with metadata."""
        pass

    @abstractmethod
    async def find_similar(
        self, embedding: list[float], threshold: float = 0.85
    ) -> list[SemanticMatch]:
        """Find semantically similar indexed embeddings."""
        pass

    @abstractmethod
    async def remove_embedding(self, key: str) -> bool:
        """Remove an embedding from the index."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all embeddings."""
        pass


class TokenCounterPort(ABC):
    """Port for token counting and cost estimation."""

    @abstractmethod
    def count_prompt_tokens(self, text: str, model: str) -> int:
        """Count tokens in prompt text."""
        pass

    @abstractmethod
    def count_completion_tokens(self, text: str, model: str) -> int:
        """Count tokens in completion text."""
        pass

    @abstractmethod
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate API cost for token usage."""
        pass

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get list of supported models."""
        pass


class EventPublisherPort(ABC):
    """Port for publishing domain events."""

    @abstractmethod
    async def publish(self, event: CacheInvalidationEvent) -> None:
        """Publish a cache invalidation event."""
        pass

    @abstractmethod
    async def subscribe(self, handler) -> None:
        """Subscribe to cache invalidation events."""
        pass


class QueryNormalizerPort(ABC):
    """Port for query normalization."""

    @abstractmethod
    def normalize(self, query: str) -> str:
        """Normalize query for comparison."""
        pass

    @abstractmethod
    def extract_intent(self, query: str) -> str:
        """Extract semantic intent from query."""
        pass

    @abstractmethod
    def similarity_score(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        pass


class CacheMetricsPort(ABC):
    """Port for cache metrics collection and reporting."""

    @abstractmethod
    async def record_hit(
        self, entry_key: str, response_time_ms: float, tokens_saved: int, cost_saved: float
    ) -> None:
        """Record a cache hit."""
        pass

    @abstractmethod
    async def record_miss(self, query: str, reason: str) -> None:
        """Record a cache miss."""
        pass

    @abstractmethod
    async def record_eviction(self, entry_key: str, policy: str) -> None:
        """Record a cache eviction."""
        pass

    @abstractmethod
    async def get_metrics(self) -> dict[str, Any]:
        """Get aggregate metrics."""
        pass


class EmbeddingGeneratorPort(ABC):
    """Port for generating embeddings from text."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get dimensionality of embeddings."""
        pass


class TOONRepositoryPort(ABC):
    """Port for TOON persistence."""

    @abstractmethod
    async def save_toon(self, toon) -> bool:
        """Save TOON operation."""
        pass

    @abstractmethod
    async def get_toon(self, operation_id: str):
        """Retrieve TOON by operation ID."""
        pass

    @abstractmethod
    async def get_all_toons(self, limit: int = 100) -> list:
        """Retrieve all TOONs."""
        pass

    @abstractmethod
    async def get_toons_by_type(self, operation_type: str, limit: int = 100) -> list:
        """Retrieve TOONs by type."""
        pass

    @abstractmethod
    async def clear_toons(self) -> int:
        """Clear all TOONs."""
        pass


class RepositoryPort(ABC):
    """Repository port for cache entry persistence (convenience abstraction)."""

    @abstractmethod
    async def save(self, entry: CacheEntry) -> None:
        """Save a cache entry."""
        pass

    @abstractmethod
    async def get_by_key(self, key: str) -> CacheEntry | None:
        """Retrieve by key."""
        pass

    @abstractmethod
    async def delete_by_key(self, key: str) -> bool:
        """Delete by key."""
        pass

    @abstractmethod
    async def find_expired(self) -> list[CacheEntry]:
        """Find all expired entries."""
        pass

    @abstractmethod
    async def find_by_policy(self, policy_name: str) -> list[CacheEntry]:
        """Find entries matching eviction policy criteria."""
        pass
