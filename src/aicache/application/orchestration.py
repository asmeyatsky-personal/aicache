"""
DAG-based Workflow Orchestration - Parallelism-First Design (2026)

This module implements parallel execution patterns following skill2026.md.
DAG-based orchestration for multi-step workflows with automatic parallelization.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """
    A single step in a workflow DAG.

    Each step declares its dependencies, enabling the orchestrator
    to automatically parallelize independent steps.
    """

    name: str
    execute: Callable[..., Any]
    depends_on: list[str] = field(default_factory=list)
    is_critical: bool = True
    timeout_seconds: float | None = None
    retry_count: int = 0

    def __post_init__(self):
        if not self.name:
            raise ValueError("Step name cannot be empty")


@dataclass
class StepResult:
    """Result of a workflow step execution."""

    step_name: str
    status: WorkflowStatus
    result: Any = None
    error: Exception | None = None
    duration_ms: float = 0.0


class DAGOrchestrator:
    """
    Executes workflow steps respecting dependency order.

    2026 Pattern: Parallelizes independent steps automatically.
    Uses asyncio.gather for concurrent execution of ready steps.
    """

    def __init__(self, steps: list[WorkflowStep]):
        self.steps: dict[str, WorkflowStep] = {s.name: s for s in steps}
        self._validate_dag()
        self._backpressure_limit = 10
        self._semaphore: asyncio.Semaphore | None = None

    def _validate_dag(self) -> None:
        """Validate DAG has no cycles."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in self.steps[node].depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Unknown dependency: {dep} for step {node}")
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_name in self.steps:
            if step_name not in visited:
                if has_cycle(step_name):
                    raise ValueError(f"Circular dependency detected involving {step_name}")

    def _get_ready_steps(self, pending: set[str], completed: set[str]) -> list[str]:
        """Find all steps whose dependencies are satisfied."""
        return [
            name for name in pending if all(dep in completed for dep in self.steps[name].depends_on)
        ]

    async def execute(
        self, context: dict[str, Any], initial_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute workflow with parallelization.

        Steps are grouped by dependency level - all ready steps
        at each level execute concurrently.
        """
        self._semaphore = asyncio.Semaphore(self._backpressure_limit)

        completed: dict[str, Any] = {}
        if initial_data:
            completed.update(initial_data)

        pending: set[str] = set(self.steps.keys())
        results: dict[str, StepResult] = {}

        while pending:
            ready = self._get_ready_steps(pending, set(completed.keys()))

            if not ready:
                raise RuntimeError(
                    f"Deadlock: no ready steps but {len(pending)} pending. Pending: {pending}"
                )

            logger.info(f"Executing {len(ready)} steps in parallel: {ready}")

            # Execute ready steps concurrently
            tasks = [self._execute_step(name, context, completed) for name in ready]

            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(ready, step_results, strict=False):
                if isinstance(result, Exception):
                    step = self.steps[name]
                    if step.is_critical:
                        raise RuntimeError(f"Critical step '{name}' failed: {result}")
                    else:
                        logger.warning(f"Non-critical step '{name}' failed: {result}")
                        results[name] = StepResult(
                            step_name=name, status=WorkflowStatus.FAILED, error=result
                        )
                        completed[name] = None
                else:
                    results[name] = result
                    completed[name] = result.result

                pending.discard(name)

        return completed

    async def _execute_step(
        self, name: str, context: dict[str, Any], completed: dict[str, Any]
    ) -> StepResult:
        """Execute a single step with timeout and error handling."""
        step = self.steps[name]
        start_time = asyncio.get_event_loop().time()

        try:
            async with self._semaphore:
                if asyncio.iscoroutinefunction(step.execute):
                    if step.timeout_seconds:
                        result = await asyncio.wait_for(
                            step.execute(context, completed),
                            timeout=step.timeout_seconds,
                        )
                    else:
                        result = await step.execute(context, completed)
                else:
                    result = step.execute(context, completed)

            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return StepResult(
                step_name=name,
                status=WorkflowStatus.COMPLETED,
                result=result,
                duration_ms=duration_ms,
            )

        except TimeoutError:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            error = TimeoutError(f"Step '{name}' timed out after {step.timeout_seconds}s")

            if step.retry_count > 0:
                logger.warning(f"Step '{name}' timed out, retrying...")
                for attempt in range(step.retry_count):
                    try:
                        result = await step.execute(context, completed)
                        return StepResult(
                            step_name=name,
                            status=WorkflowStatus.COMPLETED,
                            result=result,
                            duration_ms=duration_ms,
                        )
                    except Exception as e:
                        logger.warning(f"Retry {attempt + 1} failed: {e}")

            return StepResult(
                step_name=name,
                status=WorkflowStatus.FAILED,
                error=error,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return StepResult(
                step_name=name,
                status=WorkflowStatus.FAILED,
                error=e,
                duration_ms=duration_ms,
            )


class CacheWorkflowOrchestrator:
    """
    Specialized orchestrator for cache operations.

    2026 Pattern: Fan-out/fan-in for parallel cache operations.
    """

    def __init__(self):
        self._orchestrator: DAGOrchestrator | None = None

    async def warm_cache(
        self,
        queries: list[str],
        execute_query_fn: Callable[[str], Any],
        max_concurrency: int = 10,
    ) -> dict[str, Any]:
        """
        Warm cache for multiple queries concurrently.

        2026 Pattern: Fan-out to parallelize independent cache warming.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def warm_single(query: str) -> dict[str, Any]:
            async with semaphore:
                result = await execute_query_fn(query)
                return {"query": query, "result": result, "warmed": True}

        results = await asyncio.gather(*[warm_single(q) for q in queries], return_exceptions=True)

        return {
            "total": len(queries),
            "warmed": sum(1 for r in results if not isinstance(r, Exception)),
            "failed": sum(1 for r in results if isinstance(r, Exception)),
            "results": [r for r in results if not isinstance(r, Exception)],
        }

    async def multi_lookup(
        self,
        queries: list[str],
        lookup_fn: Callable[[str], Any],
        max_concurrency: int = 20,
    ) -> dict[str, Any]:
        """
        Perform multiple cache lookups in parallel.

        2026 Pattern: Pipeline parallelism with concurrency limit.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def lookup_single(query: str) -> dict[str, Any]:
            async with semaphore:
                return await lookup_fn(query)

        results = await asyncio.gather(*[lookup_single(q) for q in queries], return_exceptions=True)

        hits = sum(1 for r in results if not isinstance(r, Exception) and r.get("hit"))

        return {
            "total": len(queries),
            "hits": hits,
            "misses": len(queries) - hits,
            "hit_rate": hits / len(queries) if queries else 0,
            "results": [r for r in results if not isinstance(r, Exception)],
        }

    async def invalidate_pattern(
        self,
        patterns: list[str],
        invalidate_fn: Callable[[str], Any],
        max_concurrency: int = 10,
    ) -> dict[str, Any]:
        """
        Invalidate cache entries matching multiple patterns in parallel.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def invalidate_single(pattern: str) -> dict[str, Any]:
            async with semaphore:
                result = await invalidate_fn(pattern)
                return {"pattern": pattern, "invalidated": result}

        results = await asyncio.gather(
            *[invalidate_single(p) for p in patterns], return_exceptions=True
        )

        return {
            "total": len(patterns),
            "invalidated": sum(1 for r in results if not isinstance(r, Exception)),
            "failed": sum(1 for r in results if isinstance(r, Exception)),
        }


class AgentTaskDecomposer:
    """
    Decomposes complex tasks into parallel subtasks for AI agents.

    2026 Pattern: Multi-agent coordination with scoped MCP access.
    """

    def __init__(self):
        self._cache_lookup = None
        self._semantic_search = None

    async def coordinate_research(
        self,
        query: str,
        sub_queries: list[str],
        lookup_fn: Callable[[str], Any],
        synthesis_fn: Callable[[list[Any]], Any],
    ) -> dict[str, Any]:
        """
        Phase 1: Parallel research across sub-queries.

        2026 Pattern: Fan-out to independent research agents.
        """
        results = await asyncio.gather(
            *[lookup_fn(sq) for sq in sub_queries], return_exceptions=True
        )

        return {
            "query": query,
            "sub_results": [r for r in results if not isinstance(r, Exception)],
            "synthesis": synthesis_fn(results),
        }

    async def coordinate_validation(
        self, synthesis_result: Any, validators: list[Callable[[Any], Any]]
    ) -> dict[str, Any]:
        """
        Phase 2: Parallel validation.

        2026 Pattern: Multiple validators check different aspects.
        """
        results = await asyncio.gather(
            *[v(synthesis_result) for v in validators], return_exceptions=True
        )

        return {
            "validations": [
                {
                    "validator": v.__name__,
                    "passed": not isinstance(r, Exception),
                    "result": r,
                }
                for v, r in zip(validators, results, strict=False)
            ],
            "all_passed": all(not isinstance(r, Exception) for r in results),
        }
