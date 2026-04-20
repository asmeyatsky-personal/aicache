# Roadmap

Three horizons. Everything else — "sentient assistant", federated
learning, living brain, IDE extensions — is deferred indefinitely and
won't be picked up without a concrete user pull.

## Now (v0.3.0, shipping)

- SDK-wrapper caching for Anthropic + OpenAI with token / cost
  telemetry.
- Hex architecture enforced by `import-linter` in CI.
- MCP server (one per bounded context) over stdio.
- Semantic caching via Chroma + sentence-transformers (optional extra).
- `aicache savings` CLI over a rolling events log.
- Reproducible demo at `examples/claude_code_agent_demo.py`.

## Next (v0.4.x)

- **Streaming**: replay cached chunks on hit; buffer-then-store on miss
  (fingerprint already ignores `stream=True`).
- **Provider-native prompt-cache aggregation**: include Anthropic's and
  OpenAI's built-in prompt caching signals in the savings report so the
  numbers reflect total savings, not just aicache's contribution.
- **Per-project attribution**: optional project/tenant dimension on
  `AICallEvent` for multi-repo developer environments.
- **Semantic invalidation**: drop all entries whose embeddings fall
  within a threshold of a given prompt.
- **Coverage ratchet to §5 targets**: domain ≥95%, application ≥85%,
  overall ≥80% (tracked in `docs/COVERAGE_RATCHET.md`).

## Later (v0.5+)

- **Rust hot-path kernel** if the Python kernel ever exceeds the p99
  budget at typical cache sizes. Trigger criteria and baseline will
  live in `docs/adr/0001-python-cache-kernel.md` (Phase 7 of
  `plan200426.md`, not yet written).
- **Redis / S3 shared backend** for teams that need a single cache
  across machines.
- **Additional providers** only with user pull and a real workload to
  validate against (no speculative integrations).

## Deferred indefinitely

IDE/editor integrations, federated learning, behavioural analytics,
"autonomous learning", decentralised identity, team management UI,
web dashboard, public cache browser. Previously scoped but never
delivered — see `plan200426.md` and the deletion PR
(`2a371f6`) for the cleanup rationale.
