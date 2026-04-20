"""
Tests for AI-Native Structured Output Schemas (2026)

Tests the Pydantic schemas for structured AI output following skill2026.md.
"""

from datetime import datetime

import pytest

from aicache.application.schemas import (
    CacheAnalysis,
    CacheEntryMetadata,
    CacheHealthReport,
    CacheHitType,
    CacheQueryRequest,
    CacheQueryResponse,
    CacheStatsReport,
    CacheTier,
    CacheWarmupPlan,
    ContextBuilderConfig,
    InvalidationPattern,
    MultiProviderCacheStatus,
)


class TestCacheHitType:
    """Test cache hit type enum."""

    def test_hit_types(self):
        """All hit types defined."""
        assert CacheHitType.EXACT.value == "exact"
        assert CacheHitType.SEMANTIC.value == "semantic"
        assert CacheHitType.PREFIX.value == "prefix"
        assert CacheHitType.CONTEXT.value == "context"


class TestCacheTier:
    """Test cache tier enum."""

    def test_tiers(self):
        """All tiers defined."""
        assert CacheTier.HOT.value == "hot"
        assert CacheTier.WARM.value == "warm"
        assert CacheTier.COLD.value == "cold"


class TestCacheAnalysis:
    """Test cache analysis schema."""

    def test_valid_analysis(self):
        """Valid analysis passes validation."""
        analysis = CacheAnalysis(
            cache_efficiency_score=0.85,
            hit_rate_prediction=0.75,
            recommended_ttl_seconds=3600,
            suggested_optimizations=["increase semantic threshold"],
            confidence=0.90,
        )

        assert analysis.cache_efficiency_score == 0.85
        assert analysis.confidence == 0.90

    def test_invalid_score_range(self):
        """Scores must be 0-1."""
        with pytest.raises(ValueError):
            CacheAnalysis(
                cache_efficiency_score=1.5,  # Invalid
                hit_rate_prediction=0.5,
                recommended_ttl_seconds=3600,
                confidence=0.5,
            )


class TestCacheEntryMetadata:
    """Test cache entry metadata schema."""

    def test_valid_metadata(self):
        """Valid metadata passes."""
        now = datetime.now()
        metadata = CacheEntryMetadata(
            entry_key="test-key",
            created_at=now,
            last_accessed=now,
            access_count=10,
            tier=CacheTier.HOT,
            size_bytes=1024,
            expires_at=now,
            is_expired=False,
            embedding_available=True,
        )

        assert metadata.entry_key == "test-key"
        assert metadata.tier == CacheTier.HOT

    def test_optional_fields(self):
        """Optional fields can be None."""
        metadata = CacheEntryMetadata(
            entry_key="test-key",
            created_at=datetime.now(),
            access_count=0,
            tier=CacheTier.COLD,
            size_bytes=100,
            is_expired=False,
            embedding_available=False,
        )

        assert metadata.last_accessed is None
        assert metadata.expires_at is None


class TestCacheHealthReport:
    """Test cache health report schema."""

    def test_healthy_status(self):
        """Health report with healthy status."""
        report = CacheHealthReport(
            status="healthy",
            hit_rate=0.85,
            avg_response_time_ms=5.0,
            memory_usage_percent=45.0,
            issues=[],
            recommendations=["Continue monitoring"],
        )

        assert report.status == "healthy"
        assert report.hit_rate == 0.85

    def test_critical_status(self):
        """Health report with critical status."""
        report = CacheHealthReport(
            status="critical",
            hit_rate=0.10,
            avg_response_time_ms=500.0,
            memory_usage_percent=95.0,
            issues=["Memory nearly full", "Low hit rate"],
            recommendations=["Increase eviction", "Reduce cache size"],
        )

        assert report.status == "critical"
        assert len(report.issues) == 2


class TestCacheQueryRequest:
    """Test cache query request schema."""

    def test_valid_request(self):
        """Valid request passes."""
        request = CacheQueryRequest(
            query="What is machine learning?",
            context={"user_id": "123"},
            enable_semantic=True,
            threshold=0.85,
        )

        assert request.query == "What is machine learning?"
        assert request.threshold == 0.85

    def test_default_threshold(self):
        """Default threshold is 0.85."""
        request = CacheQueryRequest(query="test")

        assert request.threshold == 0.85
        assert request.enable_semantic is True


class TestCacheQueryResponse:
    """Test cache query response schema."""

    def test_hit_response(self):
        """Cache hit response."""
        response = CacheQueryResponse(
            hit=True,
            hit_type=CacheHitType.EXACT,
            value="Cached response",
            confidence=1.0,
            cache_key="abc123",
            latency_ms=5.0,
            tokens_saved=100,
            cost_saved=0.01,
        )

        assert response.hit is True
        assert response.hit_type == CacheHitType.EXACT

    def test_miss_response(self):
        """Cache miss response."""
        response = CacheQueryResponse(hit=False, latency_ms=1500.0)

        assert response.hit is False
        assert response.value is None


class TestCacheStatsReport:
    """Test comprehensive stats report."""

    def test_complete_stats(self):
        """Complete stats report."""
        now = datetime.now()
        report = CacheStatsReport(
            period_start=now,
            period_end=now,
            total_requests=1000,
            total_hits=750,
            total_misses=250,
            hit_rate=0.75,
            exact_hits=500,
            semantic_hits=200,
            prefix_hits=30,
            context_hits=20,
            total_tokens_saved=50000,
            total_cost_saved=50.0,
            avg_latency_ms=5.0,
            p50_latency_ms=4.0,
            p95_latency_ms=10.0,
            p99_latency_ms=20.0,
            cache_size_bytes=10_000_000,
            cache_entry_count=5000,
            eviction_count=100,
        )

        assert report.hit_rate == 0.75
        assert report.prefix_hits == 30
        assert report.context_hits == 20

    def test_percentages(self):
        """Percentages validated."""
        with pytest.raises(ValueError):
            CacheStatsReport(
                period_start=datetime.now(),
                period_end=datetime.now(),
                total_requests=100,
                total_hits=100,
                total_misses=0,
                hit_rate=1.5,  # Invalid - must be <=1.0
                exact_hits=100,
                semantic_hits=0,
                prefix_hits=0,
                context_hits=0,
                total_tokens_saved=0,
                total_cost_saved=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                cache_size_bytes=0,
                cache_entry_count=0,
                eviction_count=0,
            )


class TestCacheWarmupPlan:
    """Test cache warmup plan schema."""

    def test_valid_plan(self):
        """Valid warmup plan."""
        plan = CacheWarmupPlan(
            priority_queries=[
                "What is Python?",
                "How does machine learning work?",
                "Explain neural networks",
            ],
            estimated_time_seconds=30.0,
            expected_hit_rate_after_warmup=0.80,
            concurrent_warmup_limit=10,
        )

        assert len(plan.priority_queries) == 3
        assert plan.concurrent_warmup_limit == 10


class TestInvalidationPattern:
    """Test invalidation pattern schema."""

    def test_pattern_types(self):
        """Pattern types validated."""
        pattern = InvalidationPattern(
            pattern="user:*",
            pattern_type="prefix",
            reason="User logout",
            affected_entries=50,
        )

        assert pattern.pattern_type == "prefix"
        assert pattern.affected_entries == 50


class TestMultiProviderCacheStatus:
    """Test multi-provider cache status."""

    def test_provider_status(self):
        """Multi-provider status."""
        status = MultiProviderCacheStatus(
            providers={"openai": True, "anthropic": True, "google": False},
            active_provider="anthropic",
            failover_enabled=True,
            total_cost_saved=100.0,
            provider_savings={"openai": 60.0, "anthropic": 40.0},
        )

        assert status.active_provider == "anthropic"
        assert status.failover_enabled is True


class TestContextBuilderConfig:
    """Test context builder configuration."""

    def test_default_config(self):
        """Default configuration values."""
        config = ContextBuilderConfig()

        assert config.max_tokens == 100_000
        assert config.include_cache_stats is True
        assert config.history_length == 10

    def test_custom_config(self):
        """Custom configuration."""
        config = ContextBuilderConfig(
            max_tokens=50_000, system_priority=10, include_recent_history=False
        )

        assert config.max_tokens == 50_000
        assert config.system_priority == 10
        assert config.include_recent_history is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
