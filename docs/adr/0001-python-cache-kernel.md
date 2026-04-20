# ADR 0001: Python cache kernel now, Rust deferred

- **Status**: Accepted
- **Date**: 2026-04-20
- **Context**: Phase 7 of [plan200426.md](../../plan200426.md)
- **Relates to**: [Architectural Rules — 2026](../../Architectural%20Rules%20%E2%80%94%202026.md) §1

## Context

Architectural Rules 2026 §1 mandates Rust for "ledgers, parsers,
kernels, hot-path APIs (p99 < 50ms), cryptography." The aicache query
path satisfies the "hot-path API" qualifier: every SDK wrapper call
routes through `QueryCacheUseCase.execute()` on both hit and miss
before the caller sees a response.

The rule requires a one-paragraph ADR when we deviate from the stack
default. This is that ADR.

## Decision

**Keep the cache kernel in Python 3.12 for v0.3.0.** Write the Rust
kernel only when measured p99 on a representative workload exceeds the
trigger criteria below.

## Measured baseline (2026-04-20)

Ran `examples/bench_kernel.py` on the author's workstation
(Darwin 25.3.0, Python 3.12.8, in-memory storage adapter). 10,000
entries pre-populated; 5,000 round-robin exact-match lookups, every
call a hit:

| Percentile | Latency |
|-----------:|--------:|
| min  | 14.2 µs |
| p50  | 16.4 µs |
| p95  | 19.7 µs |
| p99  | **33.5 µs** |
| max  | 124.7 µs |
| mean | 17.1 µs |

p99 is three orders of magnitude below the §1 budget of 50 ms. A Rust
rewrite would save ~30 µs on a path that is already invisible inside
the cheapest network round-trip. The engineering cost — maintaining a
PyO3 boundary, a second test matrix, a second release pipeline —
massively outweighs the benefit at current scale.

## Rewrite trigger criteria

Re-open this decision when **any** of the following hold on a release
workload (not micro-benchmark):

1. **p99 on exact-match hits > 10 ms** at ≥ 10,000 cache entries on the
   filesystem adapter (not in-memory).
2. **p99 on semantic-match hits > 50 ms** at ≥ 10,000 entries with the
   Chroma adapter — half the §1 budget, so we have headroom before
   breaching it.
3. **CPU profile shows > 30% time in the Python kernel**, measured over
   a representative agentic workload (see the demo at
   `examples/claude_code_agent_demo.py` as a starting harness).
4. A named user reports latency-sensitive use that the current kernel
   can't serve, with a reproducible workload.

Criteria 1 and 2 are directly checkable with `examples/bench_kernel.py`
once filesystem + semantic options are added to that script — see the
"Next" item below.

## Non-deferrable work this ADR depends on

- Extend `examples/bench_kernel.py` to cover filesystem storage and
  semantic lookup, so criteria 1 and 2 can be measured rather than
  guessed. Landed whenever the filesystem path becomes hot; not
  required for v0.3.0.
- Keep the in-memory baseline measurement above refreshed whenever the
  domain services or the query path changes. The bench is fast enough
  (< 10 s) to run in CI.

## Consequences

- **Accepted**: we ship v0.3.0 in Python with measured margin against
  the §1 budget.
- **Accepted**: the port/adapter boundaries (StoragePort,
  SemanticIndexPort, TokenCounterPort) mean a future Rust kernel slots
  in behind the same interfaces without changes to the application or
  integration layers.
- **Rejected**: pre-emptive Rust rewrite for rule conformance. The rule
  is expressed in terms of a latency budget, and we are three orders of
  magnitude inside it.

## Revisit

- Automatic: when any trigger criterion fires in a release bench run.
- Manual: at the v0.5 cut, regardless of trigger, as a sanity check.
