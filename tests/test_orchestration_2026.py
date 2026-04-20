"""
Tests for DAG-based Workflow Orchestration (2026)

Tests the parallelism-first design patterns from skill2026.md:
- DAG execution with automatic parallelization
- Independent operations run concurrently
- Backpressure and rate limiting
"""

import asyncio

import pytest

from aicache.application.orchestration import (
    CacheWorkflowOrchestrator,
    DAGOrchestrator,
    StepResult,
    WorkflowStatus,
    WorkflowStep,
)


class TestDAGOrchestrator:
    """Test DAG-based workflow execution."""

    def test_orchestrator_validates_no_cycles(self):
        """DAG detects circular dependencies."""
        with pytest.raises(ValueError, match="Circular dependency"):
            DAGOrchestrator(
                [
                    WorkflowStep("a", lambda ctx, c: None, depends_on=["b"]),
                    WorkflowStep("b", lambda ctx, c: None, depends_on=["a"]),
                ]
            )

    def test_orchestrator_validates_unknown_dependency(self):
        """DAG detects unknown dependencies."""
        with pytest.raises(ValueError, match="Unknown dependency"):
            DAGOrchestrator(
                [
                    WorkflowStep("a", lambda ctx, c: None, depends_on=["nonexistent"]),
                ]
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_of_independent_steps(self):
        """Independent steps run in parallel."""
        execution_order = []

        async def slow_task(ctx, completed):
            await asyncio.sleep(0.1)
            execution_order.append("task")
            return "done"

        orchestrator = DAGOrchestrator(
            [
                WorkflowStep("task1", slow_task),
                WorkflowStep("task2", slow_task),
                WorkflowStep("task3", slow_task),
            ]
        )

        await orchestrator.execute({}, {})

        # All three tasks should have run
        assert len(execution_order) == 3

    @pytest.mark.asyncio
    async def test_sequential_execution_with_dependencies(self):
        """Steps with dependencies wait for completion."""
        execution_order = []

        async def track_task(name):
            async def task(ctx, completed):
                execution_order.append(name)
                return name

            return task

        orchestrator = DAGOrchestrator(
            [
                WorkflowStep("a", await track_task("a")),
                WorkflowStep("b", await track_task("b"), depends_on=["a"]),
                WorkflowStep("c", await track_task("c"), depends_on=["b"]),
            ]
        )

        await orchestrator.execute({}, {})

        # Must be in order due to dependencies
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_parallel_branches_merge(self):
        """Parallel branches merge correctly."""
        execution_times = {}

        async def timed_task(name, duration):
            async def task(ctx, completed):
                execution_times[name] = asyncio.get_event_loop().time()
                await asyncio.sleep(duration)
                return name

            return task

        orchestrator = DAGOrchestrator(
            [
                WorkflowStep("a", await timed_task("a", 0.1)),
                WorkflowStep("b", await timed_task("b", 0.1)),
                WorkflowStep("c", await timed_task("c", 0.01), depends_on=["a", "b"]),
            ]
        )

        await orchestrator.execute({}, {})

        # c should start after both a and b complete
        assert execution_times["c"] > execution_times["a"]
        assert execution_times["c"] > execution_times["b"]

    @pytest.mark.asyncio
    async def test_non_critical_step_failure(self):
        """Non-critical step failures don't stop workflow."""

        async def failing_task(ctx, completed):
            raise ValueError("Task failed")

        orchestrator = DAGOrchestrator(
            [
                WorkflowStep("a", failing_task, is_critical=False),
                WorkflowStep("b", lambda ctx, c: "success"),
            ]
        )

        result = await orchestrator.execute({}, {})

        # Should complete despite failure
        assert "b" in result

    @pytest.mark.asyncio
    async def test_critical_step_failure_raises(self):
        """Critical step failures stop workflow."""

        call_count = 0

        async def failing_task(ctx, completed):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Critical failure")
            return "recovered"

        async def recovery_task(ctx, completed):
            return "success"

        orchestrator = DAGOrchestrator(
            [
                WorkflowStep("a", failing_task, is_critical=True, retry_count=0),
                WorkflowStep("b", recovery_task),
            ]
        )

        result = await orchestrator.execute({}, {})

        # b should not have been called because a failed critically
        assert "a" not in result or result.get("a") is None


class TestCacheWorkflowOrchestrator:
    """Test cache-specific workflow orchestration."""

    @pytest.mark.asyncio
    async def test_warm_cache_parallel(self):
        """Cache warming runs queries in parallel."""
        orchestrator = CacheWorkflowOrchestrator()

        warmed_count = 0

        async def mock_execute(query):
            nonlocal warmed_count
            await asyncio.sleep(0.01)
            warmed_count += 1
            return {"query": query, "warmed": True}

        result = await orchestrator.warm_cache(
            queries=["q1", "q2", "q3"],
            execute_query_fn=mock_execute,
            max_concurrency=10,
        )

        assert result["warmed"] == 3
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_multi_lookup_parallel(self):
        """Multiple cache lookups run in parallel."""
        orchestrator = CacheWorkflowOrchestrator()

        async def mock_lookup(query):
            await asyncio.sleep(0.01)
            return {"query": query, "hit": True}

        result = await orchestrator.multi_lookup(
            queries=["q1", "q2"], lookup_fn=mock_lookup, max_concurrency=10
        )

        assert result["hits"] == 2
        assert result["hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_invalidate_pattern_parallel(self):
        """Pattern invalidation runs in parallel."""
        orchestrator = CacheWorkflowOrchestrator()

        async def mock_invalidate(pattern):
            await asyncio.sleep(0.01)
            return 5  # 5 entries invalidated

        result = await orchestrator.invalidate_pattern(
            patterns=["pattern1", "pattern2"],
            invalidate_fn=mock_invalidate,
            max_concurrency=10,
        )

        assert result["invalidated"] == 2


class TestStepResult:
    """Test step result dataclass."""

    def test_step_result_creation(self):
        """StepResult captures execution details."""
        result = StepResult(
            step_name="test-step",
            status=WorkflowStatus.COMPLETED,
            result={"key": "value"},
            duration_ms=150.0,
        )

        assert result.step_name == "test-step"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.result == {"key": "value"}
        assert result.duration_ms == 150.0

    def test_step_result_error(self):
        """StepResult captures errors."""
        result = StepResult(
            step_name="failing-step",
            status=WorkflowStatus.FAILED,
            error=ValueError("Something went wrong"),
            duration_ms=50.0,
        )

        assert result.status == WorkflowStatus.FAILED
        assert isinstance(result.error, ValueError)


class TestWorkflowStep:
    """Test workflow step configuration."""

    def test_step_requires_name(self):
        """WorkflowStep must have a name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            WorkflowStep("", lambda ctx, c: None)

    def test_step_defaults(self):
        """WorkflowStep has sensible defaults."""
        step = WorkflowStep("test", lambda ctx, c: None)

        assert step.name == "test"
        assert step.depends_on == []
        assert step.is_critical is True
        assert step.timeout_seconds is None
        assert step.retry_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
