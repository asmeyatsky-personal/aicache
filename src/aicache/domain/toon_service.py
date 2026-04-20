"""
TOON Service: Token Optimization Object Notation generation and analytics.

This domain service generates TOON objects from cache operations and provides
optimization insights. It orchestrates multiple domain services to capture
comprehensive token optimization metadata.

Architecture:
- No dependencies on infrastructure layer
- Uses port abstractions for storage and metrics
- Generates immutable TOON objects
- Aggregates TOON data for analytics
"""

import logging
from datetime import datetime
from typing import Any

from .models import CacheEntry, CacheResult, EvictionPolicy
from .ports import CacheMetricsPort, StoragePort, TokenCounterPort
from .toon import (
    TOONAnalytics,
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

logger = logging.getLogger(__name__)


class TOONGenerationService:
    """
    Generates TOON objects from cache operations.

    This service bridges the gap between cache operations and TOON representation,
    capturing comprehensive optimization metadata for every cache interaction.
    """

    def __init__(
        self, storage: StoragePort, token_counter: TokenCounterPort, metrics_port: CacheMetricsPort
    ):
        self.storage = storage
        self.token_counter = token_counter
        self.metrics_port = metrics_port

    async def generate_toon_from_cache_hit(
        self,
        operation_id: str,
        original_query: str,
        normalized_query: str,
        query_hash: str,
        cache_result: CacheResult,
        cache_entry: CacheEntry | None,
        prompt_tokens_without_cache: int,
        model: str,
        duration_ms: float,
        semantic_match: bool = False,
        context: dict[str, Any] | None = None,
        semantic_tags: list[str] | None = None,
        intent: str | None = None,
    ) -> TOONCacheOperation:
        """
        Generate a TOON object from a cache hit.

        Captures all optimization metadata for a successful cache lookup.
        """
        # Determine operation type
        if semantic_match:
            operation_type = (
                TOONOperationType.SEMANTIC_HIT
                if cache_result.similarity_score and cache_result.similarity_score >= 0.85
                else TOONOperationType.SEMANTIC_MISS
            )
            strategy = TOONStrategy.SEMANTIC
        else:
            operation_type = TOONOperationType.EXACT_HIT
            strategy = TOONStrategy.EXACT

        # Calculate token metrics
        cache_result.value.decode() if cache_result.value else ""
        with_cache_prompt = 0  # Cache hit = no new prompt tokens
        cost_with_cache = 0.0

        saved_prompt = prompt_tokens_without_cache
        saved_total = prompt_tokens_without_cache
        cost_without_cache = self.token_counter.estimate_cost(model, prompt_tokens_without_cache, 0)
        cost_saved = cost_without_cache

        token_delta = TOONTokenDelta(
            without_cache_prompt=prompt_tokens_without_cache,
            without_cache_completion=0,
            without_cache_total=prompt_tokens_without_cache,
            with_cache_prompt=with_cache_prompt,
            with_cache_completion=0,
            with_cache_total=with_cache_prompt,
            saved_prompt=saved_prompt,
            saved_completion=0,
            saved_total=saved_total,
            saved_percent=100.0 if prompt_tokens_without_cache > 0 else 0.0,
            cost_without_cache=cost_without_cache,
            cost_with_cache=cost_with_cache,
            cost_saved=cost_saved,
            model=model,
        )

        # Semantic match data
        semantic_data = TOONSemanticMatchData(
            enabled=semantic_match,
            similarity_score=cache_result.similarity_score if semantic_match else None,
            confidence=cache_result.confidence if semantic_match else None,
            matched_entry_key=cache_result.entry_key,
            semantic_distance=1.0 - (cache_result.similarity_score or 0.0)
            if semantic_match
            else None,
            embedding_dimension=384 if semantic_match else None,
            similarity_threshold_used=0.85,
            threshold_met=semantic_match,
        )

        # Cache metadata
        cache_age = cache_entry.calculate_age_seconds() if cache_entry else 0
        ttl_remaining = None
        if cache_entry and cache_entry.expires_at:
            ttl_remaining = (cache_entry.expires_at - datetime.now()).total_seconds()

        cache_metadata = TOONCacheMetadata(
            cache_key=cache_result.entry_key or query_hash,
            cache_age_seconds=cache_age,
            ttl_remaining_seconds=ttl_remaining,
            access_count=cache_entry.metadata.accessed_count
            if cache_entry and cache_entry.metadata
            else 0,
            last_accessed=cache_entry.metadata.last_accessed_at
            if cache_entry and cache_entry.metadata
            else None,
            created_at=cache_entry.created_at if cache_entry else datetime.now(),
            memory_size_bytes=cache_entry.get_size_bytes() if cache_entry else 0,
            eviction_policy=EvictionPolicy.LRU.value,
        )

        # Optimization insights
        optimization_insight = self._generate_optimization_insight(
            token_delta, cache_entry, semantic_match
        )

        # Query metadata
        query_metadata = TOONQueryMetadata(
            original_query=original_query,
            normalized_query=normalized_query,
            query_hash=query_hash,
            embedding_dimension=384 if semantic_match else None,
            intent=intent,
            semantic_tags=semantic_tags,
        )

        return TOONCacheOperation(
            operation_id=operation_id,
            timestamp=datetime.now(),
            operation_type=operation_type,
            strategy_used=strategy,
            duration_ms=duration_ms,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=optimization_insight,
            context=context,
        )

    async def generate_toon_from_cache_miss(
        self,
        operation_id: str,
        original_query: str,
        normalized_query: str,
        query_hash: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        duration_ms: float,
        semantic_attempted: bool = False,
        context: dict[str, Any] | None = None,
        semantic_tags: list[str] | None = None,
        intent: str | None = None,
    ) -> TOONCacheOperation:
        """
        Generate a TOON object from a cache miss.

        Captures metadata for failed cache lookups (both exact and semantic).
        """
        # Determine operation type
        if semantic_attempted:
            operation_type = TOONOperationType.SEMANTIC_MISS
            strategy = TOONStrategy.SEMANTIC
        else:
            operation_type = TOONOperationType.EXACT_MISS
            strategy = TOONStrategy.EXACT

        # Token metrics (all tokens charged since no cache hit)
        total_tokens = prompt_tokens + completion_tokens
        cost = self.token_counter.estimate_cost(model, prompt_tokens, completion_tokens)

        token_delta = TOONTokenDelta(
            without_cache_prompt=prompt_tokens,
            without_cache_completion=completion_tokens,
            without_cache_total=total_tokens,
            with_cache_prompt=prompt_tokens,
            with_cache_completion=completion_tokens,
            with_cache_total=total_tokens,
            saved_prompt=0,
            saved_completion=0,
            saved_total=0,
            saved_percent=0.0,
            cost_without_cache=cost,
            cost_with_cache=cost,
            cost_saved=0.0,
            model=model,
        )

        # No semantic match data for miss
        semantic_data = TOONSemanticMatchData(
            enabled=semantic_attempted,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=384 if semantic_attempted else None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        # No cache metadata for miss
        cache_metadata = TOONCacheMetadata(
            cache_key=query_hash,
            cache_age_seconds=0,
            ttl_remaining_seconds=None,
            access_count=0,
            last_accessed=None,
            created_at=datetime.now(),
            memory_size_bytes=0,
            eviction_policy=EvictionPolicy.LRU.value,
        )

        # Optimization insights for miss
        optimization_insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.NONE,
            roi_score=0.0,
            suggested_actions=["add_to_cache", "monitor_similar_queries"],
            eviction_risk="none",
            cache_efficiency_score=0.0,
            predictability_score=0.0,
            pattern_detected=False,
            similar_queries_found=0,
        )

        # Query metadata
        query_metadata = TOONQueryMetadata(
            original_query=original_query,
            normalized_query=normalized_query,
            query_hash=query_hash,
            embedding_dimension=384 if semantic_attempted else None,
            intent=intent,
            semantic_tags=semantic_tags,
        )

        return TOONCacheOperation(
            operation_id=operation_id,
            timestamp=datetime.now(),
            operation_type=operation_type,
            strategy_used=strategy,
            duration_ms=duration_ms,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=optimization_insight,
            context=context,
        )

    def _generate_optimization_insight(
        self,
        token_delta: TOONTokenDelta,
        cache_entry: CacheEntry | None,
        semantic_match: bool = False,
    ) -> TOONOptimizationInsight:
        """Generate optimization insights from cache hit data."""
        # Calculate optimization level based on token savings
        saved_percent = token_delta.saved_percent
        if saved_percent >= 80:
            optimization_level = TOONOptimizationLevel.CRITICAL
        elif saved_percent >= 60:
            optimization_level = TOONOptimizationLevel.HIGH
        elif saved_percent >= 40:
            optimization_level = TOONOptimizationLevel.MEDIUM
        elif saved_percent > 0:
            optimization_level = TOONOptimizationLevel.LOW
        else:
            optimization_level = TOONOptimizationLevel.NONE

        # ROI score (0 to 1)
        roi_score = saved_percent / 100.0

        # Suggested actions
        suggested_actions = []
        if semantic_match:
            suggested_actions.append("semantic_match_effective")
        if cache_entry and cache_entry.ttl_seconds is None:
            suggested_actions.append("enable_ttl_for_freshness")
        if cache_entry and cache_entry.metadata and cache_entry.metadata.accessed_count > 10:
            suggested_actions.append("high_reuse_detected")

        # Cache efficiency score
        cache_efficiency = 1.0 if cache_entry else 0.0

        # Eviction risk (higher access count = lower risk)
        access_count = (
            cache_entry.metadata.accessed_count if cache_entry and cache_entry.metadata else 0
        )
        eviction_risk = "low" if access_count > 5 else "medium" if access_count > 2 else "high"

        return TOONOptimizationInsight(
            optimization_level=optimization_level,
            roi_score=roi_score,
            suggested_actions=suggested_actions,
            eviction_risk=eviction_risk,
            cache_efficiency_score=cache_efficiency,
            predictability_score=0.7 if semantic_match else 0.9,
            pattern_detected=False,
            similar_queries_found=0,
        )


class TOONAnalyticsService:
    """
    Aggregates TOON objects into analytics and insights.

    Processes collections of TOON operations to generate:
    - Aggregate statistics
    - Trends and patterns
    - Performance insights
    - ROI calculations
    """

    def aggregate_toons(
        self,
        toon_operations: list[TOONCacheOperation],
        time_period_start: datetime,
        time_period_end: datetime,
    ) -> TOONAnalytics:
        """
        Aggregate TOON operations into comprehensive analytics.

        Calculates hit rates, token savings, ROI, and trends.
        """
        if not toon_operations:
            return TOONAnalytics(
                total_operations=0,
                exact_hits=0,
                semantic_hits=0,
                intent_hits=0,
                misses=0,
                total_tokens_saved=0,
                total_cost_saved=0.0,
                average_token_savings_percent=0.0,
                average_roi_score=0.0,
                operations=toon_operations,
                time_period_start=time_period_start,
                time_period_end=time_period_end,
                cache_efficiency_trend=0.0,
            )

        # Count operation types
        exact_hits = sum(
            1 for op in toon_operations if op.operation_type == TOONOperationType.EXACT_HIT
        )
        semantic_hits = sum(
            1 for op in toon_operations if op.operation_type == TOONOperationType.SEMANTIC_HIT
        )
        intent_hits = sum(
            1 for op in toon_operations if op.operation_type == TOONOperationType.INTENT_HIT
        )
        misses = sum(1 for op in toon_operations if "miss" in op.operation_type.value)

        # Aggregate token savings
        total_tokens_saved = sum(op.token_delta.saved_total for op in toon_operations)
        total_cost_saved = sum(op.token_delta.cost_saved for op in toon_operations)

        # Calculate average savings percent
        avg_savings_percent = (
            sum(op.token_delta.saved_percent for op in toon_operations) / len(toon_operations)
            if toon_operations
            else 0.0
        )

        # Calculate average ROI
        avg_roi = (
            sum(op.optimization_insight.roi_score for op in toon_operations) / len(toon_operations)
            if toon_operations
            else 0.0
        )

        # Calculate efficiency trend (simplified)
        first_half = toon_operations[: len(toon_operations) // 2]
        second_half = toon_operations[len(toon_operations) // 2 :]
        first_half_roi = (
            sum(op.optimization_insight.roi_score for op in first_half) / len(first_half)
            if first_half
            else 0.0
        )
        second_half_roi = (
            sum(op.optimization_insight.roi_score for op in second_half) / len(second_half)
            if second_half
            else 0.0
        )
        efficiency_trend = second_half_roi - first_half_roi

        return TOONAnalytics(
            total_operations=len(toon_operations),
            exact_hits=exact_hits,
            semantic_hits=semantic_hits,
            intent_hits=intent_hits,
            misses=misses,
            total_tokens_saved=total_tokens_saved,
            total_cost_saved=total_cost_saved,
            average_token_savings_percent=avg_savings_percent,
            average_roi_score=avg_roi,
            operations=toon_operations,
            time_period_start=time_period_start,
            time_period_end=time_period_end,
            cache_efficiency_trend=efficiency_trend,
        )

    def extract_insights(self, analytics: TOONAnalytics) -> dict[str, Any]:
        """
        Extract actionable insights from TOON analytics.

        Returns recommendations and performance summaries.
        """
        hit_rate = analytics.hit_rate()
        semantic_hit_rate = analytics.semantic_hit_rate()

        insights = {
            "summary": {
                "total_operations": analytics.total_operations,
                "hit_rate_percent": round(hit_rate, 2),
                "semantic_hit_rate_percent": round(semantic_hit_rate, 2),
                "miss_rate_percent": round(100 - hit_rate, 2),
            },
            "savings": {
                "total_tokens_saved": analytics.total_tokens_saved,
                "total_cost_saved": round(analytics.total_cost_saved, 6),
                "average_tokens_per_operation": round(
                    analytics.total_tokens_saved / analytics.total_operations, 1
                )
                if analytics.total_operations > 0
                else 0,
            },
            "efficiency": {
                "roi_score": round(analytics.average_roi_score, 4),
                "efficiency_trend": "improving"
                if analytics.cache_efficiency_trend > 0
                else "declining"
                if analytics.cache_efficiency_trend < 0
                else "stable",
                "trend_magnitude": round(abs(analytics.cache_efficiency_trend), 4),
            },
            "recommendations": self._generate_recommendations(analytics),
        }

        return insights

    def _generate_recommendations(self, analytics: TOONAnalytics) -> list[str]:
        """Generate recommendations based on analytics."""
        recommendations = []

        if analytics.hit_rate() < 30:
            recommendations.append(
                "Low cache hit rate - consider expanding semantic threshold or enabling intent matching"
            )

        if (
            analytics.semantic_hit_rate() > 50
            and analytics.semantic_hit_rate() > analytics.hit_rate() * 0.5
        ):
            recommendations.append(
                "Semantic caching is effective - increase similarity threshold to reduce false positives"
            )

        if analytics.average_token_savings_percent > 70:
            recommendations.append(
                "Excellent token savings - monitor cache retention and TTL settings"
            )

        if analytics.cache_efficiency_trend < 0:
            recommendations.append(
                "Cache efficiency declining - investigate cache staleness and eviction policy"
            )

        if not recommendations:
            recommendations.append("Cache performance is optimal")

        return recommendations
