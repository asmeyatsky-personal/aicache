"""MCP server tests — schema compliance + round-trip per §5.

The server is a pure JSON-RPC dispatcher over application use cases.
Tests drive it at that protocol boundary with in-memory adapters so we
exercise the real async handler path without stdio or file I/O.
"""

from __future__ import annotations

import json

import pytest

from aicache.infrastructure import build_container
from aicache.infrastructure.mcp_server import (
    _RESOURCE_SCHEMAS,
    _TOOL_SCHEMAS,
    MCPCacheServer,
    MCPRequest,
)


@pytest.fixture
def server() -> MCPCacheServer:
    return MCPCacheServer(container=build_container(in_memory=True))


# ---------------------------------------------------------------------
# Schema / protocol compliance
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_reports_name_and_capabilities(server):
    response = await server.handle(MCPRequest(id=1, method="initialize"))
    assert response.error is None
    assert response.result["serverInfo"]["name"] == "aicache"
    assert "tools" in response.result["capabilities"]
    assert "resources" in response.result["capabilities"]


@pytest.mark.asyncio
async def test_tools_list_exposes_only_write_tools(server):
    response = await server.handle(MCPRequest(id=1, method="tools/list"))
    names = [t["name"] for t in response.result["tools"]]
    # §3.5 — tools = writes only
    assert names == [
        "cache.store",
        "cache.invalidate",
        "cache.purge_expired",
        "cache.clear_all",
    ]


@pytest.mark.asyncio
async def test_resources_list_exposes_only_read_resources(server):
    response = await server.handle(MCPRequest(id=1, method="resources/list"))
    uris = [r["uri"] for r in response.result["resources"]]
    # §3.5 — resources = reads only
    assert "cache://stats" in uris
    assert "cache://entries" in uris
    assert any(u.startswith("cache://entry/") for u in uris)
    assert any(u.startswith("cache://query/") for u in uris)


def test_tool_schemas_all_have_required_fields():
    for schema in _TOOL_SCHEMAS:
        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"


def test_resource_schemas_all_have_required_fields():
    for schema in _RESOURCE_SCHEMAS:
        for field in ("uri", "name", "description", "mimeType"):
            assert field in schema, f"resource missing {field}: {schema}"


@pytest.mark.asyncio
async def test_unknown_method_returns_jsonrpc_error(server):
    response = await server.handle(MCPRequest(id=1, method="does/not/exist"))
    assert response.result is None
    assert response.error is not None
    assert response.error["code"] == -32601  # Method not found


@pytest.mark.asyncio
async def test_invalid_params_return_invalid_params_error(server):
    # tools/call with no name
    response = await server.handle(MCPRequest(id=1, method="tools/call", params={"arguments": {}}))
    assert response.error is not None
    assert response.error["code"] == -32602


# ---------------------------------------------------------------------
# Round trip — store, then read back
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_then_query_round_trip(server):
    store = await server.handle(
        MCPRequest(
            id=1,
            method="tools/call",
            params={
                "name": "cache.store",
                "arguments": {"query": "capital of France", "response": "Paris"},
            },
        )
    )
    payload = json.loads(store.result["content"][0]["text"])
    assert payload["stored"] is True
    cache_key = payload["cache_key"]

    read = await server.handle(
        MCPRequest(
            id=2,
            method="resources/read",
            params={"uri": f"cache://entry/{cache_key}"},
        )
    )
    read_payload = json.loads(read.result["contents"][0]["text"])
    assert read_payload is not None
    assert read_payload["value"] == "Paris"


@pytest.mark.asyncio
async def test_stats_reflects_store_activity(server):
    await server.handle(
        MCPRequest(
            id=1,
            method="tools/call",
            params={
                "name": "cache.store",
                "arguments": {"query": "q", "response": "r"},
            },
        )
    )
    stats = await server.handle(
        MCPRequest(id=2, method="resources/read", params={"uri": "cache://stats"})
    )
    payload = json.loads(stats.result["contents"][0]["text"])
    # At minimum the schema should report a non-negative byte count.
    assert payload["cache_size_bytes"] >= 0


@pytest.mark.asyncio
async def test_invalidate_removes_entry(server):
    store = await server.handle(
        MCPRequest(
            id=1,
            method="tools/call",
            params={
                "name": "cache.store",
                "arguments": {"query": "delete me", "response": "r"},
            },
        )
    )
    cache_key = json.loads(store.result["content"][0]["text"])["cache_key"]

    invalidate = await server.handle(
        MCPRequest(
            id=2,
            method="tools/call",
            params={"name": "cache.invalidate", "arguments": {"cache_key": cache_key}},
        )
    )
    assert json.loads(invalidate.result["content"][0]["text"])["invalidated"] is True

    read = await server.handle(
        MCPRequest(
            id=3,
            method="resources/read",
            params={"uri": f"cache://entry/{cache_key}"},
        )
    )
    assert json.loads(read.result["contents"][0]["text"]) is None


@pytest.mark.asyncio
async def test_clear_all_requires_confirm(server):
    result = await server.handle(
        MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "cache.clear_all", "arguments": {"confirm": False}},
        )
    )
    payload = json.loads(result.result["content"][0]["text"])
    assert payload["cleared"] is False


@pytest.mark.asyncio
async def test_clear_all_with_confirm_empties_cache(server):
    # Seed two entries
    for query in ("a", "b"):
        await server.handle(
            MCPRequest(
                id=1,
                method="tools/call",
                params={
                    "name": "cache.store",
                    "arguments": {"query": query, "response": "r"},
                },
            )
        )

    await server.handle(
        MCPRequest(
            id=2,
            method="tools/call",
            params={"name": "cache.clear_all", "arguments": {"confirm": True}},
        )
    )

    listed = await server.handle(
        MCPRequest(id=3, method="resources/read", params={"uri": "cache://entries"})
    )
    assert json.loads(listed.result["contents"][0]["text"])["count"] == 0


@pytest.mark.asyncio
async def test_query_resource_reports_miss_when_empty(server):
    result = await server.handle(
        MCPRequest(
            id=1,
            method="resources/read",
            params={"uri": "cache://query/never-seen"},
        )
    )
    payload = json.loads(result.result["contents"][0]["text"])
    assert payload["hit"] is False


@pytest.mark.asyncio
async def test_unknown_tool_name_rejected(server):
    result = await server.handle(
        MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "cache.does_not_exist", "arguments": {}},
        )
    )
    assert result.error is not None
    assert result.error["code"] == -32602


@pytest.mark.asyncio
async def test_unknown_resource_uri_rejected(server):
    result = await server.handle(
        MCPRequest(id=1, method="resources/read", params={"uri": "bogus://thing"})
    )
    assert result.error is not None
    assert result.error["code"] == -32602


@pytest.mark.asyncio
async def test_mcprequest_model_validate_json_round_trips():
    """Protocol surface: JSON strings on stdin round-trip through pydantic."""
    raw = '{"jsonrpc":"2.0","id":5,"method":"tools/list","params":null}'
    request = MCPRequest.model_validate_json(raw)
    assert request.id == 5
    assert request.method == "tools/list"
