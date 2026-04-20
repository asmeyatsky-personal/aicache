"""Reproducible demo: measure aicache's impact on a realistic workload.

Runs a fixed set of Anthropic-like calls twice against the same cached
client. The first pass populates the cache; the second pass should hit
every entry. We print a before/after summary using the telemetry log
so the numbers aren't fabricated — they come out of the same pipe a
production install would use.

If the anthropic SDK and an API key are available (and
``AICACHE_DEMO_LIVE=1``), we route through a real client. Otherwise a
deterministic stub stands in so the demo runs anywhere in ~1 second.

Usage::

    python examples/claude_code_agent_demo.py
    AICACHE_DEMO_LIVE=1 python examples/claude_code_agent_demo.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from aicache.infrastructure import build_container, summarise
from aicache.integrations.anthropic import cached_client

# ---------------------------------------------------------------------
# Stub client — used when no Anthropic API key is available
# ---------------------------------------------------------------------


@dataclass
class _StubUsage:
    input_tokens: int = 512
    output_tokens: int = 384


@dataclass
class _StubResponse:
    id: str
    content: list[dict[str, Any]]
    model: str
    usage: _StubUsage

    def model_dump_json(self) -> str:
        import json

        return json.dumps(
            {
                "id": self.id,
                "content": self.content,
                "model": self.model,
                "usage": {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                },
            }
        )


@dataclass
class _StubMessages:
    call_count: int = 0

    def create(self, **params: Any) -> _StubResponse:
        self.call_count += 1
        # Simulate network latency so before/after numbers look real.
        time.sleep(0.05)
        return _StubResponse(
            id=f"msg_{self.call_count:04d}",
            content=[{"type": "text", "text": "ok"}],
            model=params["model"],
            usage=_StubUsage(),
        )


@dataclass
class _StubAnthropic:
    messages: _StubMessages = field(default_factory=_StubMessages)


# ---------------------------------------------------------------------
# Workload — the thing we want to measure
# ---------------------------------------------------------------------


WORKLOAD: list[dict[str, Any]] = [
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Summarise the ports and adapters pattern."}],
    },
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "List five common Python web frameworks."}],
    },
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "Explain the difference between hit rate and MRU."}
        ],
    },
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "What does the Unix `tee` command do?"}],
    },
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "Give me three good names for a caching library."}
        ],
    },
]


def _make_client() -> Any:
    if os.environ.get("AICACHE_DEMO_LIVE") == "1":
        try:
            from anthropic import Anthropic

            return Anthropic()
        except ImportError:  # pragma: no cover — demo-only
            print("[warn] anthropic SDK not installed — falling back to stub.")
    return _StubAnthropic()


def _run_pass(client: Any, label: str) -> float:
    start = time.perf_counter()
    for req in WORKLOAD:
        client.messages.create(**req)
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed * 1000:.1f} ms across {len(WORKLOAD)} calls")
    return elapsed


async def _summarise(container: Any) -> dict[str, Any]:
    events = await container.telemetry.read_recent(days=1)
    return summarise(events)


def main() -> None:
    print("aicache demo — two passes over the same workload\n")

    container = build_container(in_memory=True)
    upstream = _make_client()
    client = cached_client(upstream, container=container)

    print("Pass 1 (populating cache):")
    t1 = _run_pass(client, "first")

    print("Pass 2 (expected hits):")
    t2 = _run_pass(client, "second")

    import asyncio

    report = asyncio.run(_summarise(container))

    print("\n--- Telemetry (from the shared events log) ---")
    print(f"  Calls          : {report['total_calls']}")
    print(f"  Hits           : {report['hits']}")
    print(f"  Misses         : {report['misses']}")
    print(f"  Hit rate       : {report['hit_rate']:.1%}")
    print(f"  Tokens saved   : {report['tokens_saved']:,}")
    print(f"  $ saved        : ${report['cost_saved_usd']:.6f}")
    print(f"  $ spent        : ${report['cost_spent_usd']:.6f}")

    if t1 > 0:
        speedup = t1 / t2 if t2 > 0 else float("inf")
        print(f"\n  End-to-end pass speedup: {speedup:.1f}x")

    assert report["hits"] == len(WORKLOAD), "second pass should be all hits"
    assert report["misses"] == len(WORKLOAD), "first pass should be all misses"
    print("\n✓ All second-pass calls were cache hits.")


if __name__ == "__main__":
    main()
