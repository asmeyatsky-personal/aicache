"""Micro-benchmark for the Python cache kernel.

Populates ``N`` entries in the in-memory adapter and measures latency
of :class:`QueryCacheUseCase.execute` for exact-match hits. Output is
the distribution we care about for the Rust-kernel ADR: p50 / p95 /
p99 / worst.

Run::

    python examples/bench_kernel.py
    python examples/bench_kernel.py --entries 100000 --iterations 10000
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from aicache.infrastructure import build_container


async def _populate(container, n: int) -> list[str]:
    queries = [f"prompt number {i}" for i in range(n)]
    for q in queries:
        key = container.query_cache._generate_cache_key(
            container.query_normalizer.normalize(q), None
        )
        await container.store_cache.execute(key=key, value=b"cached response")
    return queries


async def _measure(container, queries: list[str], iterations: int) -> list[float]:
    samples: list[float] = []
    # Round-robin over the populated queries so we aren't hot-keying one slot.
    for i in range(iterations):
        q = queries[i % len(queries)]
        start = time.perf_counter_ns()
        result = await container.query_cache.execute(q)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        assert result.hit, "benchmark precondition: all lookups should hit"
        samples.append(elapsed_us)
    return samples


def _quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    idx = min(int(q * len(s)), len(s) - 1)
    return s[idx]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=5_000)
    args = parser.parse_args()

    container = build_container(in_memory=True)

    print(f"populating {args.entries:,} entries…")
    t0 = time.perf_counter()
    queries = await _populate(container, args.entries)
    populate_s = time.perf_counter() - t0
    print(f"  done in {populate_s * 1000:.0f} ms")

    print(f"measuring {args.iterations:,} lookups (all hits)…")
    samples = await _measure(container, queries, args.iterations)

    samples.sort()
    print(
        "\nLatency (µs, in-memory storage, Python 3.12):\n"
        f"  min  : {samples[0]:8.1f}\n"
        f"  p50  : {_quantile(samples, 0.50):8.1f}\n"
        f"  p95  : {_quantile(samples, 0.95):8.1f}\n"
        f"  p99  : {_quantile(samples, 0.99):8.1f}\n"
        f"  max  : {samples[-1]:8.1f}\n"
        f"  mean : {statistics.fmean(samples):8.1f}\n"
        f"  n    : {len(samples):,}"
    )


if __name__ == "__main__":
    asyncio.run(main())
