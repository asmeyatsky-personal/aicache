"""
Domain layer: Immutable cache entry aggregate and value objects.

This module defines the core domain models for the AI caching system following DDD principles.
All cache entries are immutable to prevent accidental state corruption.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    CONDITIONAL = "conditional"


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    CUSTOM = "custom"


@dataclass(frozen=True)
class CacheMetadata:
    """Immutable metadata for cache entries."""

    created_at: datetime
    accessed_count: int = 0
    last_accessed_at: datetime | None = None
    normalized_query: str | None = None
    semantic_tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.semantic_tags is None:
            object.__setattr__(self, "semantic_tags", [])
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def touch(self) -> "CacheMetadata":
        """Record access without mutation."""
        return replace(
            self, accessed_count=self.accessed_count + 1, last_accessed_at=datetime.now()
        )


@dataclass(frozen=True)
class CachePolicy:
    """Immutable cache policy value object."""

    max_size_bytes: int
    default_ttl_seconds: int | None
    eviction_policy: EvictionPolicy
    semantic_match_threshold: float = 0.85
    enable_compression: bool = True
    enable_semantic_caching: bool = True

    def validate(self) -> bool:
        """Validate policy constraints."""
        return (
            self.max_size_bytes > 0
            and self.semantic_match_threshold > 0
            and self.semantic_match_threshold <= 1.0
        )


@dataclass(frozen=True)
class CacheEntry:
    """
    Immutable cache entry aggregate root.

    This is the core domain entity that represents a cached query response.
    All state is immutable - operations return new instances.

    Invariants:
    - key cannot be empty
    - value cannot be empty
    - expires_at must be after created_at if set
    """

    key: str
    value: bytes
    created_at: datetime
    expires_at: datetime | None = None
    embedding: list[float] | None = None
    metadata: CacheMetadata | None = None
    ttl_seconds: int | None = None
    context: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate invariants."""
        if not self.key:
            raise ValueError("Cache key cannot be empty")
        if not self.value:
            raise ValueError("Cache value cannot be empty")
        if self.metadata is None:
            object.__setattr__(self, "metadata", CacheMetadata(created_at=self.created_at))
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError("Expiration time must be after creation time")

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def touch(self) -> "CacheEntry":
        """Record access without mutation."""
        if self.metadata is None:
            return self
        return replace(self, metadata=self.metadata.touch())

    def refresh_ttl(self) -> "CacheEntry":
        """Refresh expiration time based on TTL."""
        if self.ttl_seconds is None:
            return self
        return replace(self, expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds))

    def get_size_bytes(self) -> int:
        """Estimate memory size of cache entry."""
        return len(self.key.encode()) + len(self.value)

    def calculate_age_seconds(self) -> float:
        """Calculate age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass(frozen=True)
class CacheInvalidationEvent:
    """Domain event for cache invalidation."""

    cache_key: str
    reason: str
    triggered_by: str
    timestamp: datetime
    strategy: InvalidationStrategy
    affected_entries: int = 0

    def __post_init__(self):
        if not self.cache_key:
            raise ValueError("Cache key required for invalidation event")


@dataclass(frozen=True)
class SemanticMatch:
    """Result of semantic similarity matching."""

    similarity_score: float
    matched_entry_key: str
    confidence: float

    def __post_init__(self):
        if not 0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0 and 1")
        if not 0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class CacheResult:
    """Result of cache query operation."""

    hit: bool
    value: bytes | None = None
    entry_key: str | None = None
    similarity_score: float | None = None
    confidence: float | None = None
    response_time_ms: float = 0.0

    @classmethod
    def create_hit(
        cls, value: bytes, entry_key: str, response_time_ms: float = 0.0
    ) -> "CacheResult":
        """Create a cache hit result."""
        return cls(hit=True, value=value, entry_key=entry_key, response_time_ms=response_time_ms)

    @classmethod
    def create_semantic_hit(
        cls,
        value: bytes,
        entry_key: str,
        similarity_score: float,
        confidence: float,
        response_time_ms: float = 0.0,
    ) -> "CacheResult":
        """Create a semantic cache hit result."""
        return cls(
            hit=True,
            value=value,
            entry_key=entry_key,
            similarity_score=similarity_score,
            confidence=confidence,
            response_time_ms=response_time_ms,
        )

    @classmethod
    def create_miss(cls, response_time_ms: float = 0.0) -> "CacheResult":
        """Create a cache miss result."""
        return cls(hit=False, response_time_ms=response_time_ms)


@dataclass(frozen=True)
class TokenUsageMetrics:
    """Immutable token usage metrics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

    def __post_init__(self):
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("Token counts cannot be negative")


@dataclass(frozen=True)
class CacheMetrics:
    """Immutable aggregate cache metrics."""

    total_hits: int
    total_misses: int
    total_evictions: int
    average_response_time_ms: float
    total_tokens_saved: int
    total_cost_saved: float
    hit_rate: float
    memory_usage_bytes: int
    semantic_matches: int
    false_positives: int

    def calculate_roi(self) -> float:
        """Calculate return on investment."""
        total_operations = self.total_hits + self.total_misses
        if total_operations == 0:
            return 0.0
        return self.total_cost_saved / total_operations if self.total_cost_saved > 0 else 0.0
