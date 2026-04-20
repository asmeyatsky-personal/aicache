Architectural Rules — 2026
Hard rules for AI-assisted code generation. Anything not in this file is a suggestion.
Conflict handling (read first)
Idiom mismatch (PRD language ≠ any example): PRD wins.
Principle mismatch (PRD violates §2, §3, or §4): principle wins, revise the PRD.
Record resolutions as ADRs in memory://decisions.
1. Stack defaults
Rust — ledgers, parsers, kernels, hot-path APIs (p99 < 50ms), cryptography
Python 3.12+ — AI/ML, agentic orchestration, data, CLI, prototypes
TypeScript + React — all frontends
Postgres primary • Firestore for documents • Redis cache • BigQuery analytics
GCP — Cloud Run, Secret Manager, IAM, Workload Identity
Protobuf or JSON Schema for cross-service contracts
Deviation requires a one-paragraph ADR citing why the default fails.
2. Layer direction
domain ← application ← infrastructure
← presentation
Domain imports nothing from infrastructure, presentation, or any SDK.
Application imports only domain.
Infrastructure implements domain ports.
MCP servers live in infrastructure and wrap application use cases.
Enforce mechanically: import-linter (Py) / module boundaries or cargo-deny (Rust) /
eslint-plugin-boundaries (TS). Rule not in CI = rule not real.
3. Non-negotiable architecture rules
1. 2. 3. 4. 5. 6. 7. No business logic in repositories, adapters, or MCP servers.
Every external dependency has a port. Adapters implement it. Tests use in-memory
adapters.
Domain models are immutable. State changes return new instances.
Invariants enforced in constructors/factories — never setters.
One MCP server per bounded context. Tools = writes. Resources = reads.
Independent steps run concurrently. Dependencies are explicit (DAG). Every external
call has a timeout.
Every module header documents: layer, ports, MCP integration, and stack choice if non-
canonical.
4. Security (hard rules, CI-enforced)
No secrets in code, config, env defaults, or repo history. Secret Manager + Workload
Identity only.
Validate all external input at presentation and domain boundaries. Schema-based
(Pydantic / serde / Zod). Reject by default.
Every write emits an audit event: actor, action, before/after hash. Append-only,
separate IAM.
Every external call has a timeout and circuit breaker. No unbounded waits.
AI output that mutates state must be validated against an explicit schema first.
Agents get scoped MCP tool access — minimum needed, time-boxed.
No AI-generated code executes without a sandbox: container, no net egress by
default, CPU/memory caps.
Supply chain: lockfiles committed, dependency scan in CI ( cargo-audit / pip-audit /
npm audit ), signed commits on protected branches, SBOM at build.
5. Testing floor
Domain: ≥95% coverage, zero mocks (pure logic).
Application: ≥85% coverage, mock ports only.
Overall: ≥80%.
MCP servers: schema compliance + round-trip tests.
CI blocks merges below threshold.
6. Production observability
OpenTelemetry tracing, propagated through MCP calls.
RED metrics (Rate, Errors, Duration) per endpoint and per MCP tool.
Structured JSON logs with correlation IDs. Zero PII.
Per AI call log: model ID, version, prompt hash, tokens in/out, latency, cost.
7. Anti-patterns (rejected at review)
1. 2. 3. 4. 5. 6. 7. 8. 9. 10. Anemic domain models (getters/setters only, no behavior).
Fat services + empty domain.
Domain importing SDKs or frameworks.
MCP sprawl — one server per function instead of per context.
Sequential execution of independent operations.
Untyped strings from AI used as-is.
Secrets in code or config.
Missing timeouts / unbounded waits.
Language lock-in reasoning (“the skill showed Python”).
Rules declared but not mechanically enforced.
Changelog: v3 (2026-04) — cut to rules-only. Long-form patterns, appendices, and
reference code moved to per-language reference repos.