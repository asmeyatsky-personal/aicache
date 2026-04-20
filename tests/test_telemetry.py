"""Telemetry adapter + integration event tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from aicache.domain.models import AICallEvent
from aicache.infrastructure import (
    InMemoryTelemetryAdapter,
    JSONLTelemetryAdapter,
    build_container,
    summarise,
)
from aicache.integrations.anthropic import cached_client as anthropic_cached_client

# ---------------------------------------------------------------------
# Domain event validation
# ---------------------------------------------------------------------


def _event(**overrides: Any) -> AICallEvent:
    defaults: dict[str, Any] = dict(
        timestamp=datetime.now(tz=timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        prompt_hash="abc",
        tokens_in=100,
        tokens_out=50,
        latency_ms=12.3,
        cost_usd=0.001,
        cache_hit=False,
        match_type="miss",
    )
    defaults.update(overrides)
    return AICallEvent(**defaults)


def test_event_rejects_invalid_match_type():
    with pytest.raises(ValueError):
        _event(match_type="cached")


def test_event_rejects_negative_tokens():
    with pytest.raises(ValueError):
        _event(tokens_in=-1)


# ---------------------------------------------------------------------
# In-memory adapter
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_adapter_records_and_reads_back():
    telem = InMemoryTelemetryAdapter()
    await telem.record_ai_call(_event())
    recent = await telem.read_recent(days=30)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_in_memory_adapter_filters_by_age():
    telem = InMemoryTelemetryAdapter()
    old = _event(timestamp=datetime.now(tz=timezone.utc) - timedelta(days=60))
    new = _event(timestamp=datetime.now(tz=timezone.utc))
    await telem.record_ai_call(old)
    await telem.record_ai_call(new)
    recent = await telem.read_recent(days=30)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_in_memory_adapter_caps_at_max_events():
    telem = InMemoryTelemetryAdapter(max_events=3)
    for _ in range(10):
        await telem.record_ai_call(_event())
    assert len(await telem.read_recent(days=30)) == 3


# ---------------------------------------------------------------------
# JSONL adapter
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jsonl_adapter_round_trips(tmp_path):
    path = tmp_path / "events.jsonl"
    telem = JSONLTelemetryAdapter(path)
    await telem.record_ai_call(_event(cache_hit=True, match_type="exact"))
    await telem.record_ai_call(_event(cache_hit=False, match_type="miss"))

    recovered = await telem.read_recent(days=30)
    assert [e.cache_hit for e in recovered] == [True, False]
    # The file is newline-delimited JSON — ingestible by DuckDB / jq.
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["provider"] == "anthropic" for line in lines)


@pytest.mark.asyncio
async def test_jsonl_adapter_skips_malformed_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"not":"an event"}\nnot json at all\n')
    telem = JSONLTelemetryAdapter(path)
    assert await telem.read_recent(days=30) == []


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def test_summarise_buckets_by_model_and_sums_cost():
    events = [
        _event(cache_hit=True, match_type="exact", cost_usd=0.002, model="claude-sonnet-4-6"),
        _event(cache_hit=True, match_type="semantic", cost_usd=0.003, model="claude-sonnet-4-6"),
        _event(cache_hit=False, match_type="miss", cost_usd=0.010, model="gpt-4o"),
    ]
    report = summarise(events)
    assert report["total_calls"] == 3
    assert report["hits"] == 2
    assert report["misses"] == 1
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["cost_saved_usd"] == pytest.approx(0.005)
    assert report["cost_spent_usd"] == pytest.approx(0.010)
    assert "claude-sonnet-4-6" in report["per_model"]
    assert report["per_model"]["claude-sonnet-4-6"]["hits"] == 2


def test_summarise_empty_events():
    assert summarise([])["total_calls"] == 0


# ---------------------------------------------------------------------
# Integration: SDK wrapper emits events
# ---------------------------------------------------------------------


@dataclass
class _StubUsage:
    input_tokens: int = 17
    output_tokens: int = 23


@dataclass
class _StubMessage:
    id: str = "msg_01"
    content: list[dict[str, str]] = field(default_factory=lambda: [{"type": "text", "text": "Hi"}])
    model: str = "claude-sonnet-4-6"
    usage: _StubUsage = field(default_factory=_StubUsage)

    def model_dump_json(self) -> str:
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

    def create(self, **_params: Any) -> _StubMessage:
        self.call_count += 1
        return _StubMessage()


@dataclass
class _StubAnthropic:
    messages: _StubMessages = field(default_factory=_StubMessages)


@pytest.mark.asyncio
async def test_sdk_miss_emits_event_with_tokens_and_cost():
    container = build_container(in_memory=True)
    stub = _StubAnthropic()
    client = anthropic_cached_client(stub, container=container)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )

    events = await container.telemetry.read_recent(days=1)
    assert len(events) == 1
    event = events[0]
    assert event.cache_hit is False
    assert event.match_type == "miss"
    assert event.tokens_in == 17  # from stub.usage
    assert event.tokens_out == 23
    assert event.cost_usd > 0.0


@pytest.mark.asyncio
async def test_sdk_second_call_emits_cache_hit_event():
    container = build_container(in_memory=True)
    stub = _StubAnthropic()
    client = anthropic_cached_client(stub, container=container)

    params = dict(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )
    client.messages.create(**params)
    client.messages.create(**params)

    events = await container.telemetry.read_recent(days=1)
    assert [e.cache_hit for e in events] == [False, True]
    assert events[1].match_type == "exact"
    assert stub.messages.call_count == 1
