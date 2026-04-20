"""Unit tests for the Anthropic integration.

These don't require the anthropic SDK — we duck-type a stub client so
we can verify cache hit/miss behaviour without the network or the
optional dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from aicache.infrastructure import build_container
from aicache.integrations._common import build_request_fingerprint
from aicache.integrations.anthropic import cached_client


@dataclass
class _StubResponse:
    id: str
    content: list[dict[str, str]]
    model: str

    def model_dump_json(self) -> str:
        return json.dumps({"id": self.id, "content": self.content, "model": self.model})


@dataclass
class _StubMessages:
    response_factory: Any
    call_count: int = 0
    seen_params: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **params: Any) -> _StubResponse:
        self.call_count += 1
        self.seen_params.append(params)
        return self.response_factory(params)


@dataclass
class _StubAnthropic:
    messages: _StubMessages


def _make_stub(response_factory: Any) -> _StubAnthropic:
    return _StubAnthropic(messages=_StubMessages(response_factory=response_factory))


@pytest.fixture
def container():
    return build_container(in_memory=True)


def _default_factory(params: dict[str, Any]) -> _StubResponse:
    return _StubResponse(
        id="msg_01",
        content=[{"type": "text", "text": "Hi"}],
        model=params["model"],
    )


def test_first_call_hits_upstream(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert stub.messages.call_count == 1
    assert response.content[0]["text"] == "Hi"


def test_second_identical_call_is_a_cache_hit(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    params = dict(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )
    first = client.messages.create(**params)
    second = client.messages.create(**params)

    assert stub.messages.call_count == 1, "upstream called twice — cache missed"
    # Cached result round-trips through JSON; deserialisation yields a
    # plain dict when the anthropic SDK isn't available.
    assert (
        second.content[0]["text"] if hasattr(second, "content") else second["content"][0]["text"]
    ) == "Hi"
    assert first.id == (second.id if hasattr(second, "id") else second["id"])


def test_distinct_models_do_not_collide(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)
    base_params = dict(
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )

    client.messages.create(model="claude-opus-4-7", **base_params)
    client.messages.create(model="claude-sonnet-4-6", **base_params)
    client.messages.create(model="claude-opus-4-7", **base_params)

    assert stub.messages.call_count == 2


def test_distinct_messages_do_not_collide(container):
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hello"}],
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "goodbye"}],
    )

    assert stub.messages.call_count == 2


def test_streaming_flag_does_not_vary_cache_key(container):
    """Cached response should satisfy a streaming follow-up — we'll
    still need stream support in a later phase, but the fingerprint
    itself must not diverge.
    """
    stub = _make_stub(_default_factory)
    client = cached_client(stub, container=container)

    params = dict(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": "hi"}],
    )
    client.messages.create(**params)
    client.messages.create(**params, stream=False)

    assert stub.messages.call_count == 1


def test_fingerprint_is_deterministic_across_dict_ordering():
    a = build_request_fingerprint(
        "anthropic",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7},
    )
    b = build_request_fingerprint(
        "anthropic",
        {"temperature": 0.7, "messages": [{"role": "user", "content": "hi"}], "model": "m"},
    )
    assert a == b


def test_fingerprint_distinguishes_meaningful_params():
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
    }
    tweaked = {**base, "temperature": 0.0}
    assert build_request_fingerprint("anthropic", base) != build_request_fingerprint(
        "anthropic", tweaked
    )


def test_upstream_exception_does_not_cache_result(container):
    calls = {"n": 0}

    def factory(_params: dict[str, Any]) -> _StubResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return _StubResponse(id="m", content=[{"type": "text", "text": "ok"}], model="m")

    stub = _make_stub(factory)
    client = cached_client(stub, container=container)

    params = dict(
        model="m",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(RuntimeError):
        client.messages.create(**params)

    response = client.messages.create(**params)
    assert stub.messages.call_count == 2  # both calls hit upstream
    text = (
        response.content[0]["text"]
        if hasattr(response, "content")
        else response["content"][0]["text"]
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_async_path_hits_cache_on_second_call(container):
    class _AsyncStubMessages:
        def __init__(self) -> None:
            self.call_count = 0

        async def create(self, **params: Any) -> _StubResponse:
            self.call_count += 1
            return _StubResponse(
                id="msg_async", content=[{"type": "text", "text": "hi"}], model=params["model"]
            )

    class _AsyncStubAnthropic:
        def __init__(self) -> None:
            self.messages = _AsyncStubMessages()

    stub = _AsyncStubAnthropic()
    client = cached_client(stub, container=container)

    params = dict(
        model="claude-sonnet-4-6",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
    )

    first = await client.messages.acreate(**params)
    second = await client.messages.acreate(**params)

    assert stub.messages.call_count == 1
    assert first.id == (second.id if hasattr(second, "id") else second["id"])
