# aicache

**Drop-in caching for Claude and OpenAI SDK calls — with semantic hits, cost telemetry, and an MCP server for agents.**

One import change. Second identical call is free, tracked, and ~1 ms.
Rephrased calls hit too, when semantic caching is on.

```python
from anthropic import Anthropic
from aicache.integrations.anthropic import cached_client

client = cached_client(Anthropic())

# First call → real API, tokens billed, event logged.
# Second identical call → cached, $0, tracked in savings report.
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain the Fourier transform"}],
)
```

## Why this exists

Anthropic's and OpenAI's built-in prompt caching helps with prefix reuse
within a session. It does not help with:

- **Rerunning the same agent flow twice** in development — you pay twice.
- **Rephrased prompts** that should return the same answer — no hit.
- **Cross-provider cost visibility** — you have to stitch together
  dashboards from each vendor.

aicache sits one layer above the SDK and fills those gaps. Same
fingerprint + same params → cache hit. Similar meaning → semantic hit
(optional). Every call, hit or miss, written to
`~/.cache/aicache/events.jsonl` with `tokens_in`, `tokens_out`,
`cost_usd`, `model`, `latency_ms`, `cache_hit`, `match_type`.

## Install

```bash
pip install aicache                      # base: exact-match cache + MCP server
pip install aicache[anthropic]           # + Anthropic SDK integration
pip install aicache[openai]              # + OpenAI SDK integration
pip install aicache[semantic]            # + Chroma + sentence-transformers
pip install aicache[observability]       # + OpenTelemetry tracing shim
pip install aicache[all]                 # everything incl. dev extras
```

Python 3.12+.

## Quick tour

### 1. SDK wrapper — the 80% case

```python
from anthropic import Anthropic
from aicache.integrations.anthropic import cached_client

client = cached_client(Anthropic())
client.messages.create(model="claude-sonnet-4-6", max_tokens=512,
                       messages=[{"role": "user", "content": "hello"}])
```

Works for sync and async clients. Same pattern for OpenAI:

```python
from openai import OpenAI
from aicache.integrations.openai import cached_client

client = cached_client(OpenAI())
client.chat.completions.create(model="gpt-4o",
                               messages=[{"role": "user", "content": "hi"}])
```

Full reference: [`docs/SDK_QUICKSTART.md`](docs/SDK_QUICKSTART.md).

### 2. Semantic hits across rephrasings (optional)

```python
from aicache.infrastructure import build_container
from aicache.integrations.anthropic import cached_client
from anthropic import Anthropic

client = cached_client(Anthropic(), container=build_container(semantic=True))
# "How do I sort a list in Python?" and "python sort a list example"
# resolve to the same cache entry above the similarity threshold.
```

Uses ChromaDB + sentence-transformers locally. No network, no API key.

### 3. MCP server for agents

```bash
aicache mcp start           # stdio; point Claude Desktop / Claude Code at it
```

One MCP server exposes cache tools (writes) and resources (reads)
following [Architectural Rules 2026 §3.5](Architectural%20Rules%20%E2%80%94%202026.md).

### 4. Rolling savings report

```bash
aicache savings --days 30
```

Reads `~/.cache/aicache/events.jsonl` and prints hit rate, tokens saved,
`$` saved vs `$` spent, per-model breakdown. `--json` for CI.

## Architecture

Hexagonal / ports-and-adapters. Aligned to [Architectural Rules — 2026](Architectural%20Rules%20%E2%80%94%202026.md).

```
domain/         ← models, ports, services (no SDK imports, no I/O)
application/    ← use cases (imports only domain)
infrastructure/ ← adapters, container, MCP server, telemetry
integrations/   ← Anthropic / OpenAI SDK wrappers (presentation)
```

Enforced in CI by [`import-linter`](pyproject.toml). Add a
domain→infrastructure import and the build fails.

## Roadmap

See [`plan200426.md`](plan200426.md). Delivered to date: Phases 0–5
(guardrails, hex wiring, SDK wedge, MCP consolidation, semantic
caching, observability). Phase 6 is the README + demo you are looking
at; Phase 7 is a deferred Rust-kernel ADR.

## Contributing

1. `make dev` to install with the `[dev]` extras.
2. `make ci` runs lint + import-linter + tests with coverage gate.
3. `make type` shows the current mypy backlog (not blocking until
   Phase 1 coverage catches up — tracked in
   [`docs/COVERAGE_RATCHET.md`](docs/COVERAGE_RATCHET.md)).

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
