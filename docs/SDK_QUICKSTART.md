# SDK quickstart

Drop-in caching for the Anthropic and OpenAI SDKs. One import change,
identical request body → cache hit.

## Install

```bash
pip install aicache[anthropic]
# or
pip install aicache[openai]
```

## Anthropic

```python
from anthropic import Anthropic
from aicache.integrations.anthropic import cached_client

client = cached_client(Anthropic())

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain Fourier transforms"}],
)

# Second identical call — cache hit, zero API cost, ~1 ms latency.
cached = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain Fourier transforms"}],
)
```

The async client works the same way via `messages.acreate`:

```python
from anthropic import AsyncAnthropic
from aicache.integrations.anthropic import cached_client

client = cached_client(AsyncAnthropic())
response = await client.messages.acreate(model="claude-sonnet-4-6", ...)
```

## OpenAI

```python
from openai import OpenAI
from aicache.integrations.openai import cached_client

client = cached_client(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Translate 'good morning' to Japanese"}],
)
```

## What varies the cache key

A request is considered identical (and therefore a hit) when these
fields match, regardless of key ordering:

| Field | Varies cache key |
|-------|:----------------:|
| `model` | ✅ |
| `messages` | ✅ |
| `system` | ✅ |
| `max_tokens` | ✅ |
| `temperature` | ✅ |
| `top_p` / `top_k` / `seed` | ✅ |
| `tools` / `tool_choice` | ✅ |
| `response_format` | ✅ |
| `stop` / `stop_sequences` | ✅ |
| `stream` | ❌ — cached response satisfies both |
| timeouts / transport options | ❌ |

## Customising the cache

By default every cached client uses the process-wide default
[`Container`](../src/aicache/infrastructure/container.py). Pass a custom
one for tests or multi-tenant setups:

```python
from aicache.infrastructure import build_container
from aicache.integrations.anthropic import cached_client

container = build_container(cache_dir="/tmp/aicache-tests", in_memory=False)
client = cached_client(Anthropic(), container=container)
```

For tests specifically, `build_container(in_memory=True)` wires the
in-memory storage adapter so nothing hits disk.

## What is not yet supported

- **Streaming**: the request fingerprint already ignores `stream=True`,
  but the wrapper doesn't yet replay chunks from cache. Phase 5 ships
  that; until then, streaming calls always hit the upstream API.
- **Provider-native prompt caching interop**: the Anthropic and OpenAI
  prompt-caching features work alongside aicache (they're transparent
  to us), but we don't yet aggregate the savings into one report.
  Phase 5 ships that.
- **Cost attribution per model/project**: TOON analytics currently track
  hit/miss counts. Per-model `$` savings land in Phase 5.
