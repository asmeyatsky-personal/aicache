# Coverage ratchet

`Architectural Rules — 2026.md` §5 targets:

| Layer | Target |
|-------|--------|
| `domain` | ≥95% |
| `application` | ≥85% |
| Overall | ≥80% |

Current `--cov-fail-under` gate is lower than the target to prevent a
blocked main branch while the refactor lands. It ratchets up each phase.

| Phase | Overall gate | Domain | Application |
|-------|-------------:|-------:|------------:|
| 0 (now) | 50% (line only) | — | — |
| 1 | 60% | 80% | 75% |
| 2 | 65% | 85% | 80% |
| 3 | 70% | 90% | 82% |
| 4 | 75% | 92% | 85% |
| 5 | 80% | 95% | 85% |
| 6 | 80% (locked to §5 target) | 95% | 85% |

Per-layer gates are added alongside the overall one in Phase 1 when the
composition root makes application-layer coverage tractable.
