"""Unit tests for the OpenAI integration (stubbed SDK)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from aicache.infrastructure import build_container
from aicache.integrations.openai import cached_client


@dataclass
class _StubChoice:
    message: dict[str, str]


@dataclass
class _StubCompletion:
    id: str
    choices: list[_StubChoice]
    model: str

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "choices": [{"message": c.message} for c in self.choices],
                "model": self.model,
            }
        )


@dataclass
class _StubCompletions:
    response_factory: Any
    call_count: int = 0

    def create(self, **params: Any) -> _StubCompletion:
        self.call_count += 1
        return self.response_factory(params)


@dataclass
class _StubChat:
    completions: _StubCompletions


@dataclass
class _StubOpenAI:
    chat: _StubChat


def _default_factory(params: dict[str, Any]) -> _StubCompletion:
    return _StubCompletion(
        id="chatcmpl_01",
        choices=[_StubChoice(message={"role": "assistant", "content": "ack"})],
        model=params["model"],
    )


def _make_stub(response_factory: Any) -> _StubOpenAI:
    return _StubOpenAI(
        chat=_StubChat(completions=_StubCompletions(response_factory=response_factory))
    )


@pytest.fixture
def container():
    return build_container(in_memory=True)


def test_second_call_is_cache_hit(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)
    params = dict(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    first = client.chat.completions.create(**params)
    second = client.chat.completions.create(**params)

    assert stub.chat.completions.call_count == 1
    first_id = first.id if hasattr(first, "id") else first["id"]
    second_id = second.id if hasattr(second, "id") else second["id"]
    assert first_id == second_id


def test_distinct_messages_do_not_collide(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "foo"}],
    )
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "bar"}],
    )
    assert stub.chat.completions.call_count == 2


def test_temperature_varies_cache_key(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    base = dict(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    client.chat.completions.create(**base, temperature=0.0)
    client.chat.completions.create(**base, temperature=0.7)
    client.chat.completions.create(**base, temperature=0.0)  # dup — cache hit

    assert stub.chat.completions.call_count == 2
