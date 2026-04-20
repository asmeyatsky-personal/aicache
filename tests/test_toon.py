"""
Comprehensive Test Suite for TOON (Token Optimization Object Notation)

Tests cover:
- Domain models (immutability, validation)
- TOON generation service
- TOON analytics service
- Repository persistence
- Export formats
- Query builder
"""

import json
from datetime import datetime, timedelta

import pytest

# Import domain models for cache
# Import domain models
from aicache.domain.toon import (
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

# Import services
from aicache.domain.toon_service import TOONAnalyticsService

# Import adapters
from aicache.infrastructure.toon_adapters import (
    InMemoryTOONRepositoryAdapter,
    TOONExportService,
)


class TestTOONDomainModels:
    """Test TOON domain models for correctness and immutability."""

    def test_toon_query_metadata_creation(self):
        """Test creating a valid TOONQueryMetadata."""
        metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        assert metadata.original_query == "What is AI?"
        assert metadata.normalized_query == "what is ai"
        assert metadata.query_hash == "sha256:abc123"

    def test_toon_query_metadata_immutability(self):
        """Test that TOONQueryMetadata is immutable."""
        metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        with pytest.raises(AttributeError):
            metadata.original_query = "Modified"

    def test_toon_query_metadata_validation(self):
        """Test TOONQueryMetadata validation."""
        with pytest.raises(ValueError):
            TOONQueryMetadata(
                original_query="",  # Empty
                normalized_query="what is ai",
                query_hash="sha256:abc123",
            )

    def test_toon_token_delta_creation(self):
        """Test creating a valid TOONTokenDelta."""
        delta = TOONTokenDelta(
            without_cache_prompt=15,
            without_cache_completion=10,
            without_cache_total=25,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=15,
            saved_completion=10,
            saved_total=25,
            saved_percent=100.0,
            cost_without_cache=0.000375,
            cost_with_cache=0.0,
            cost_saved=0.000375,
            model="claude-3-opus",
        )

        assert delta.saved_total == 25
        assert delta.saved_percent == 100.0
        assert delta.cost_saved == 0.000375

    def test_toon_token_delta_validation(self):
        """Test TOONTokenDelta validation."""
        # Invalid: zero tokens
        with pytest.raises(ValueError):
            TOONTokenDelta(
                without_cache_prompt=0,
                without_cache_completion=0,
                without_cache_total=0,
                with_cache_prompt=0,
                with_cache_completion=0,
                with_cache_total=0,
                saved_prompt=0,
                saved_completion=0,
                saved_total=0,
                saved_percent=0.0,
                cost_without_cache=0.0,
                cost_with_cache=0.0,
                cost_saved=0.0,
                model="claude-3-opus",
            )

    def test_toon_semantic_match_data_creation(self):
        """Test creating TOONSemanticMatchData."""
        data = TOONSemanticMatchData(
            enabled=True,
            similarity_score=0.92,
            confidence=0.95,
            matched_entry_key="sha256:xyz789",
            semantic_distance=0.08,
            embedding_dimension=384,
            similarity_threshold_used=0.85,
            threshold_met=True,
        )

        assert data.similarity_score == 0.92
        assert data.confidence == 0.95
        assert data.threshold_met is True

    def test_toon_semantic_match_data_validation(self):
        """Test semantic match data validation."""
        # Invalid: similarity score out of range
        with pytest.raises(ValueError):
            TOONSemanticMatchData(
                enabled=True,
                similarity_score=1.5,  # Invalid: > 1.0
                confidence=0.95,
                matched_entry_key="sha256:xyz789",
                semantic_distance=0.08,
                embedding_dimension=384,
                similarity_threshold_used=0.85,
                threshold_met=True,
            )

    def test_toon_optimization_insight_creation(self):
        """Test creating TOONOptimizationInsight."""
        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=["maintain_settings"],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=5,
        )

        assert insight.optimization_level == TOONOptimizationLevel.HIGH
        assert insight.roi_score == 0.85

    def test_toon_cache_operation_creation(self):
        """Test creating a complete TOONCacheOperation."""
        query_metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=15,
            without_cache_completion=0,
            without_cache_total=15,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=15,
            saved_completion=0,
            saved_total=15,
            saved_percent=100.0,
            cost_without_cache=0.000225,
            cost_with_cache=0.0,
            cost_saved=0.000225,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:abc123",
            cache_age_seconds=3600,
            ttl_remaining_seconds=82800,
            access_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            memory_size_bytes=1024,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=["maintain_settings"],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="550e8400-e29b-41d4-a716-446655440000",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=4.5,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        assert toon.operation_id == "550e8400-e29b-41d4-a716-446655440000"
        assert toon.operation_type == TOONOperationType.EXACT_HIT

    def test_toon_to_dict(self):
        """Test TOON serialization to dict."""
        query_metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=15,
            without_cache_completion=0,
            without_cache_total=15,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=15,
            saved_completion=0,
            saved_total=15,
            saved_percent=100.0,
            cost_without_cache=0.000225,
            cost_with_cache=0.0,
            cost_saved=0.000225,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:abc123",
            cache_age_seconds=3600,
            ttl_remaining_seconds=82800,
            access_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            memory_size_bytes=1024,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=["maintain_settings"],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="550e8400-e29b-41d4-a716-446655440000",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=4.5,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        toon_dict = toon.to_dict()

        assert toon_dict["version"] == "1.0"
        assert toon_dict["operation_type"] == "exact_hit"
        assert toon_dict["tokens"]["saved"]["total"] == 15
        assert toon_dict["tokens"]["saved"]["percent"] == 100.0

    def test_toon_to_json(self):
        """Test TOON serialization to JSON string."""
        query_metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=15,
            without_cache_completion=0,
            without_cache_total=15,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=15,
            saved_completion=0,
            saved_total=15,
            saved_percent=100.0,
            cost_without_cache=0.000225,
            cost_with_cache=0.0,
            cost_saved=0.000225,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:abc123",
            cache_age_seconds=3600,
            ttl_remaining_seconds=82800,
            access_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            memory_size_bytes=1024,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=["maintain_settings"],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="550e8400-e29b-41d4-a716-446655440000",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=4.5,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        json_str = toon.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["operation_type"] == "exact_hit"


class TestTOONRepository:
    """Test TOON repository adapters."""

    @pytest.mark.asyncio
    async def test_in_memory_repository_save_and_get(self):
        """Test saving and retrieving TOON from memory."""
        repo = InMemoryTOONRepositoryAdapter()

        query_metadata = TOONQueryMetadata(
            original_query="What is AI?",
            normalized_query="what is ai",
            query_hash="sha256:abc123",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=15,
            without_cache_completion=0,
            without_cache_total=15,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=15,
            saved_completion=0,
            saved_total=15,
            saved_percent=100.0,
            cost_without_cache=0.000225,
            cost_with_cache=0.0,
            cost_saved=0.000225,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:abc123",
            cache_age_seconds=3600,
            ttl_remaining_seconds=82800,
            access_count=5,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            memory_size_bytes=1024,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=["maintain_settings"],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="test-op-001",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=4.5,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        # Save
        assert await repo.save_toon(toon) is True

        # Retrieve
        retrieved = await repo.get_toon("test-op-001")
        assert retrieved is not None
        assert retrieved.operation_id == "test-op-001"
        assert retrieved.operation_type == TOONOperationType.EXACT_HIT

    @pytest.mark.asyncio
    async def test_in_memory_repository_delete(self):
        """Test deleting TOON from memory."""
        repo = InMemoryTOONRepositoryAdapter()

        # Create and save a TOON
        query_metadata = TOONQueryMetadata(
            original_query="Test",
            normalized_query="test",
            query_hash="sha256:test",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=10,
            without_cache_completion=0,
            without_cache_total=10,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=10,
            saved_completion=0,
            saved_total=10,
            saved_percent=100.0,
            cost_without_cache=0.00015,
            cost_with_cache=0.0,
            cost_saved=0.00015,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:test",
            cache_age_seconds=0,
            ttl_remaining_seconds=None,
            access_count=0,
            last_accessed=None,
            created_at=datetime.now(),
            memory_size_bytes=100,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=[],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="delete-test",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=1.0,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        await repo.save_toon(toon)

        # Delete
        assert await repo.delete_toon("delete-test") is True
        assert await repo.get_toon("delete-test") is None


class TestTOONAnalytics:
    """Test TOON analytics service."""

    def test_analytics_aggregation(self):
        """Test aggregating TOON operations into analytics."""
        analytics_service = TOONAnalyticsService()

        # Create sample TOONs
        toons = []
        for i in range(10):
            query_metadata = TOONQueryMetadata(
                original_query=f"Query {i}",
                normalized_query=f"query {i}",
                query_hash=f"sha256:{i}",
            )

            token_delta = TOONTokenDelta(
                without_cache_prompt=15 + i,
                without_cache_completion=0,
                without_cache_total=15 + i,
                with_cache_prompt=0,
                with_cache_completion=0,
                with_cache_total=0,
                saved_prompt=15 + i,
                saved_completion=0,
                saved_total=15 + i,
                saved_percent=100.0,
                cost_without_cache=0.000225 + (i * 0.00001),
                cost_with_cache=0.0,
                cost_saved=0.000225 + (i * 0.00001),
                model="claude-3-opus",
            )

            semantic_data = TOONSemanticMatchData(
                enabled=False,
                similarity_score=None,
                confidence=None,
                matched_entry_key=None,
                semantic_distance=None,
                embedding_dimension=None,
                similarity_threshold_used=0.85,
                threshold_met=False,
            )

            cache_metadata = TOONCacheMetadata(
                cache_key=f"sha256:{i}",
                cache_age_seconds=3600,
                ttl_remaining_seconds=82800,
                access_count=5 + i,
                last_accessed=datetime.now(),
                created_at=datetime.now(),
                memory_size_bytes=1024,
                eviction_policy="lru",
            )

            insight = TOONOptimizationInsight(
                optimization_level=TOONOptimizationLevel.HIGH,
                roi_score=0.85,
                suggested_actions=[],
                eviction_risk="low",
                cache_efficiency_score=0.95,
                predictability_score=0.8,
                pattern_detected=False,
                similar_queries_found=0,
            )

            toon = TOONCacheOperation(
                operation_id=f"op-{i}",
                timestamp=datetime.now(),
                operation_type=TOONOperationType.EXACT_HIT
                if i % 2 == 0
                else TOONOperationType.SEMANTIC_HIT,
                strategy_used=TOONStrategy.EXACT if i % 2 == 0 else TOONStrategy.SEMANTIC,
                duration_ms=4.5 + i,
                query_metadata=query_metadata,
                token_delta=token_delta,
                semantic_data=semantic_data,
                cache_metadata=cache_metadata,
                optimization_insight=insight,
            )
            toons.append(toon)

        # Aggregate
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        analytics = analytics_service.aggregate_toons(toons, start_time, end_time)

        assert analytics.total_operations == 10
        assert analytics.exact_hits == 5
        assert analytics.semantic_hits == 5
        assert analytics.total_tokens_saved > 0
        assert analytics.total_cost_saved > 0

    def test_analytics_insights(self):
        """Test extracting insights from analytics."""
        analytics_service = TOONAnalyticsService()

        # Create sample TOONs
        toons = []
        for i in range(20):
            query_metadata = TOONQueryMetadata(
                original_query=f"Query {i}",
                normalized_query=f"query {i}",
                query_hash=f"sha256:{i}",
            )

            # Vary tokens for different operations
            tokens = 20 + (i * 2) if i < 10 else 10 + (i % 5)

            token_delta = TOONTokenDelta(
                without_cache_prompt=tokens,
                without_cache_completion=0,
                without_cache_total=tokens,
                with_cache_prompt=0,
                with_cache_completion=0,
                with_cache_total=0,
                saved_prompt=tokens,
                saved_completion=0,
                saved_total=tokens,
                saved_percent=100.0,
                cost_without_cache=tokens * 0.000015,
                cost_with_cache=0.0,
                cost_saved=tokens * 0.000015,
                model="claude-3-opus",
            )

            semantic_data = TOONSemanticMatchData(
                enabled=False,
                similarity_score=None,
                confidence=None,
                matched_entry_key=None,
                semantic_distance=None,
                embedding_dimension=None,
                similarity_threshold_used=0.85,
                threshold_met=False,
            )

            cache_metadata = TOONCacheMetadata(
                cache_key=f"sha256:{i}",
                cache_age_seconds=3600,
                ttl_remaining_seconds=82800,
                access_count=5,
                last_accessed=datetime.now(),
                created_at=datetime.now(),
                memory_size_bytes=1024,
                eviction_policy="lru",
            )

            insight = TOONOptimizationInsight(
                optimization_level=TOONOptimizationLevel.HIGH,
                roi_score=0.85,
                suggested_actions=[],
                eviction_risk="low",
                cache_efficiency_score=0.95,
                predictability_score=0.8,
                pattern_detected=False,
                similar_queries_found=0,
            )

            toon = TOONCacheOperation(
                operation_id=f"op-{i}",
                timestamp=datetime.now(),
                operation_type=TOONOperationType.EXACT_HIT,
                strategy_used=TOONStrategy.EXACT,
                duration_ms=4.5,
                query_metadata=query_metadata,
                token_delta=token_delta,
                semantic_data=semantic_data,
                cache_metadata=cache_metadata,
                optimization_insight=insight,
            )
            toons.append(toon)

        # Get insights
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        analytics = analytics_service.aggregate_toons(toons, start_time, end_time)
        insights = analytics_service.extract_insights(analytics)

        assert "summary" in insights
        assert "savings" in insights
        assert "efficiency" in insights
        assert "recommendations" in insights
        assert len(insights["recommendations"]) > 0


class TestTOONExport:
    """Test TOON export functionality."""

    @pytest.mark.asyncio
    async def test_export_to_json(self):
        """Test exporting TOONs to JSON."""
        repo = InMemoryTOONRepositoryAdapter()
        export_service = TOONExportService(repo)

        # Create and save a TOON
        query_metadata = TOONQueryMetadata(
            original_query="Test Query",
            normalized_query="test query",
            query_hash="sha256:test",
        )

        token_delta = TOONTokenDelta(
            without_cache_prompt=20,
            without_cache_completion=0,
            without_cache_total=20,
            with_cache_prompt=0,
            with_cache_completion=0,
            with_cache_total=0,
            saved_prompt=20,
            saved_completion=0,
            saved_total=20,
            saved_percent=100.0,
            cost_without_cache=0.0003,
            cost_with_cache=0.0,
            cost_saved=0.0003,
            model="claude-3-opus",
        )

        semantic_data = TOONSemanticMatchData(
            enabled=False,
            similarity_score=None,
            confidence=None,
            matched_entry_key=None,
            semantic_distance=None,
            embedding_dimension=None,
            similarity_threshold_used=0.85,
            threshold_met=False,
        )

        cache_metadata = TOONCacheMetadata(
            cache_key="sha256:test",
            cache_age_seconds=0,
            ttl_remaining_seconds=None,
            access_count=0,
            last_accessed=None,
            created_at=datetime.now(),
            memory_size_bytes=500,
            eviction_policy="lru",
        )

        insight = TOONOptimizationInsight(
            optimization_level=TOONOptimizationLevel.HIGH,
            roi_score=0.85,
            suggested_actions=[],
            eviction_risk="low",
            cache_efficiency_score=0.95,
            predictability_score=0.8,
            pattern_detected=False,
            similar_queries_found=0,
        )

        toon = TOONCacheOperation(
            operation_id="export-test",
            timestamp=datetime.now(),
            operation_type=TOONOperationType.EXACT_HIT,
            strategy_used=TOONStrategy.EXACT,
            duration_ms=2.0,
            query_metadata=query_metadata,
            token_delta=token_delta,
            semantic_data=semantic_data,
            cache_metadata=cache_metadata,
            optimization_insight=insight,
        )

        await repo.save_toon(toon)

        # Export to JSON
        json_data = await export_service.export_to_json()

        assert isinstance(json_data, str)
        parsed = json.loads(json_data)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["operation_id"] == "export-test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
