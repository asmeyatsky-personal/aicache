"""
Token Optimization Object Notation (TOON) - Domain Models

TOON is a structured notation for representing token optimization metadata and analytics.
Every cache operation produces a TOON object that captures:
- Token usage without cache vs. with cache
- Cost savings and optimization decisions
- Semantic match confidence and decisions
- Cache metadata and optimization insights

This module defines immutable TOON domain models following DDD principles.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class TOONOperationType(Enum):
    """Type of cache operation captured by TOON."""

    EXACT_HIT = "exact_hit"  # Exact match cache hit
    SEMANTIC_HIT = "semantic_hit"  # Semantic similarity match
    INTENT_HIT = "intent_hit"  # Intent-based match
    EXACT_MISS = "exact_miss"  # Exact match miss, no cache entry
    SEMANTIC_MISS = "semantic_miss"  # Semantic match attempted but threshold not met
    CACHE_ERROR = "cache_error"  # Cache operation failed


class TOONOptimizationLevel(Enum):
    """Overall optimization effectiveness level."""

    CRITICAL = "critical"  # Very high savings opportunity
    HIGH = "high"  # Significant savings
    MEDIUM = "medium"  # Moderate savings
    LOW = "low"  # Minimal savings
    NONE = "none"  # No optimization


class TOONStrategy(Enum):
    """Cache matching strategy used."""

    EXACT = "exact"  # Exact key match with normalization
    SEMANTIC = "semantic"  # Semantic similarity matching
    INTENT = "intent"  # Intent-based matching
    NONE = "none"  # No cache strategy (miss)


@dataclass(frozen=True)
class TOONQueryMetadata:
    """Immutable query metadata for TOON."""

    original_query: str
    normalized_query: str
    query_hash: str
    embedding_dimension: int | None = None
    intent: str | None = None
    semantic_tags: list[str] | None = None

    def __post_init__(self):
        if not self.original_query:
            raise ValueError("Original query cannot be empty")
        if not self.normalized_query:
            raise ValueError("Normalized query cannot be empty")
        if not self.query_hash:
            raise ValueError("Query hash cannot be empty")


@dataclass(frozen=True)
class TOONTokenDelta:
    """Immutable representation of token usage delta."""

    without_cache_prompt: int
    without_cache_completion: int
    without_cache_total: int
    with_cache_prompt: int
    with_cache_completion: int
    with_cache_total: int
    saved_prompt: int
    saved_completion: int
    saved_total: int
    saved_percent: float
    cost_without_cache: float
    cost_with_cache: float
    cost_saved: float
    model: str

    def __post_init__(self):
        if self.without_cache_total <= 0:
            raise ValueError("Token counts must be positive")
        if not 0 <= self.saved_percent <= 100:
            raise ValueError("Saved percent must be between 0 and 100")
        if not self.model:
            raise ValueError("Model name required")


@dataclass(frozen=True)
class TOONSemanticMatchData:
    """Immutable semantic match decision data."""

    enabled: bool
    similarity_score: float | None
    confidence: float | None
    matched_entry_key: str | None
    semantic_distance: float | None
    embedding_dimension: int | None
    similarity_threshold_used: float
    threshold_met: bool

    def __post_init__(self):
        if self.enabled and self.similarity_score is None:
            raise ValueError("Similarity score required when semantic matching enabled")
        if self.similarity_score is not None and not 0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0 and 1")
        if self.confidence is not None and not 0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class TOONCacheMetadata:
    """Immutable cache entry metadata snapshot for TOON."""

    cache_key: str
    cache_age_seconds: float
    ttl_remaining_seconds: float | None
    access_count: int
    last_accessed: datetime | None
    created_at: datetime
    memory_size_bytes: int
    eviction_policy: str

    def __post_init__(self):
        if not self.cache_key:
            raise ValueError("Cache key required")
        if self.cache_age_seconds < 0:
            raise ValueError("Cache age cannot be negative")
        if self.access_count < 0:
            raise ValueError("Access count cannot be negative")


@dataclass(frozen=True)
class TOONOptimizationInsight:
    """Immutable optimization insights and recommendations."""

    optimization_level: TOONOptimizationLevel
    roi_score: float
    suggested_actions: list[str]
    eviction_risk: str
    cache_efficiency_score: float
    predictability_score: float
    pattern_detected: bool
    similar_queries_found: int

    def __post_init__(self):
        if not 0 <= self.roi_score <= 1.0:
            raise ValueError("ROI score must be between 0 and 1")
        if not 0 <= self.cache_efficiency_score <= 1.0:
            raise ValueError("Cache efficiency score must be between 0 and 1")
        if not 0 <= self.predictability_score <= 1.0:
            raise ValueError("Predictability score must be between 0 and 1")


@dataclass(frozen=True)
class TOONCacheOperation:
    """Immutable representation of a single cache operation with full optimization context."""

    # Operation metadata
    operation_id: str
    timestamp: datetime
    operation_type: TOONOperationType
    strategy_used: TOONStrategy
    duration_ms: float

    # Query information
    query_metadata: TOONQueryMetadata

    # Token economics
    token_delta: TOONTokenDelta

    # Semantic matching decision
    semantic_data: TOONSemanticMatchData

    # Cache state snapshot
    cache_metadata: TOONCacheMetadata

    # Insights and recommendations
    optimization_insight: TOONOptimizationInsight

    # Optional context
    context: dict[str, Any] | None = None
    error_message: str | None = None

    def __post_init__(self):
        if not self.operation_id:
            raise ValueError("Operation ID required")
        if self.duration_ms < 0:
            raise ValueError("Duration cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert TOON to dictionary representation."""
        return {
            "version": "1.0",
            "operation_id": self.operation_id,
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type.value,
            "strategy_used": self.strategy_used.value,
            "duration_ms": self.duration_ms,
            "query": {
                "original": self.query_metadata.original_query,
                "normalized": self.query_metadata.normalized_query,
                "hash": self.query_metadata.query_hash,
                "embedding_dimension": self.query_metadata.embedding_dimension,
                "intent": self.query_metadata.intent,
                "semantic_tags": self.query_metadata.semantic_tags or [],
            },
            "tokens": {
                "without_cache": {
                    "prompt": self.token_delta.without_cache_prompt,
                    "completion": self.token_delta.without_cache_completion,
                    "total": self.token_delta.without_cache_total,
                },
                "with_cache": {
                    "prompt": self.token_delta.with_cache_prompt,
                    "completion": self.token_delta.with_cache_completion,
                    "total": self.token_delta.with_cache_total,
                },
                "saved": {
                    "prompt": self.token_delta.saved_prompt,
                    "completion": self.token_delta.saved_completion,
                    "total": self.token_delta.saved_total,
                    "percent": round(self.token_delta.saved_percent, 2),
                },
                "costs": {
                    "without_cache": round(self.token_delta.cost_without_cache, 6),
                    "with_cache": round(self.token_delta.cost_with_cache, 6),
                    "saved": round(self.token_delta.cost_saved, 6),
                },
                "model": self.token_delta.model,
            },
            "semantic_match": {
                "enabled": self.semantic_data.enabled,
                "similarity_score": round(self.semantic_data.similarity_score, 4)
                if self.semantic_data.similarity_score
                else None,
                "confidence": round(self.semantic_data.confidence, 4)
                if self.semantic_data.confidence
                else None,
                "matched_entry_key": self.semantic_data.matched_entry_key,
                "semantic_distance": round(self.semantic_data.semantic_distance, 4)
                if self.semantic_data.semantic_distance
                else None,
                "threshold_used": self.semantic_data.similarity_threshold_used,
                "threshold_met": self.semantic_data.threshold_met,
            },
            "cache_metadata": {
                "cache_key": self.cache_metadata.cache_key,
                "cache_age_seconds": round(self.cache_metadata.cache_age_seconds, 2),
                "ttl_remaining_seconds": round(self.cache_metadata.ttl_remaining_seconds, 2)
                if self.cache_metadata.ttl_remaining_seconds
                else None,
                "access_count": self.cache_metadata.access_count,
                "last_accessed": self.cache_metadata.last_accessed.isoformat()
                if self.cache_metadata.last_accessed
                else None,
                "created_at": self.cache_metadata.created_at.isoformat(),
                "memory_size_bytes": self.cache_metadata.memory_size_bytes,
                "eviction_policy": self.cache_metadata.eviction_policy,
            },
            "optimization_insights": {
                "optimization_level": self.optimization_insight.optimization_level.value,
                "roi_score": round(self.optimization_insight.roi_score, 4),
                "suggested_actions": self.optimization_insight.suggested_actions,
                "eviction_risk": self.optimization_insight.eviction_risk,
                "cache_efficiency_score": round(
                    self.optimization_insight.cache_efficiency_score, 4
                ),
                "predictability_score": round(self.optimization_insight.predictability_score, 4),
                "pattern_detected": self.optimization_insight.pattern_detected,
                "similar_queries_found": self.optimization_insight.similar_queries_found,
            },
            "context": self.context or {},
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Serialize TOON to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_compact_dict(self) -> dict[str, Any]:
        """Convert to compact representation for storage efficiency."""
        return {
            "v": "1.0",
            "id": self.operation_id,
            "ts": self.timestamp.isoformat(),
            "op": self.operation_type.value,
            "st": self.strategy_used.value,
            "dur": self.duration_ms,
            "q_orig": self.query_metadata.original_query,
            "q_norm": self.query_metadata.normalized_query,
            "q_hash": self.query_metadata.query_hash,
            "tok_saved": self.token_delta.saved_total,
            "tok_pct": round(self.token_delta.saved_percent, 1),
            "cost_saved": round(self.token_delta.cost_saved, 6),
            "sem_score": round(self.semantic_data.similarity_score, 2)
            if self.semantic_data.similarity_score
            else None,
            "opt_level": self.optimization_insight.optimization_level.value,
            "roi": round(self.optimization_insight.roi_score, 2),
        }


@dataclass(frozen=True)
class TOONAnalytics:
    """Immutable aggregated TOON analytics."""

    total_operations: int
    exact_hits: int
    semantic_hits: int
    intent_hits: int
    misses: int
    total_tokens_saved: int
    total_cost_saved: float
    average_token_savings_percent: float
    average_roi_score: float
    operations: list[TOONCacheOperation]
    time_period_start: datetime
    time_period_end: datetime
    cache_efficiency_trend: float  # -1.0 to 1.0

    def __post_init__(self):
        if self.total_operations < 0:
            raise ValueError("Total operations cannot be negative")
        if self.total_tokens_saved < 0:
            raise ValueError("Total tokens saved cannot be negative")
        if self.total_cost_saved < 0:
            raise ValueError("Total cost saved cannot be negative")

    def hit_rate(self) -> float:
        """Calculate overall hit rate."""
        if self.total_operations == 0:
            return 0.0
        hits = self.exact_hits + self.semantic_hits + self.intent_hits
        return (hits / self.total_operations) * 100

    def semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate."""
        if self.total_operations == 0:
            return 0.0
        return (self.semantic_hits / self.total_operations) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": "1.0",
            "period": {
                "start": self.time_period_start.isoformat(),
                "end": self.time_period_end.isoformat(),
            },
            "operations": {
                "total": self.total_operations,
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "intent_hits": self.intent_hits,
                "misses": self.misses,
                "hit_rate_percent": round(self.hit_rate(), 2),
                "semantic_hit_rate_percent": round(self.semantic_hit_rate(), 2),
            },
            "tokens": {
                "total_saved": self.total_tokens_saved,
                "average_savings_percent": round(self.average_token_savings_percent, 2),
            },
            "costs": {
                "total_saved": round(self.total_cost_saved, 6),
            },
            "insights": {
                "average_roi_score": round(self.average_roi_score, 4),
                "cache_efficiency_trend": round(self.cache_efficiency_trend, 4),
            },
        }

    def to_json(self) -> str:
        """Serialize analytics to JSON."""
        data = self.to_dict()
        # Remove operations list for compact representation
        data.pop("operations", None)
        return json.dumps(data, indent=2)
