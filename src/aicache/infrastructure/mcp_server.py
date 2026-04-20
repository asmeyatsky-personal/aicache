"""MCP server for the CacheManagement bounded context.

Per Architectural Rules 2026 §3.5: one MCP server per bounded context,
tools are writes and resources are reads. This server wraps the
application use cases wired in :class:`Container` — it contains no
business logic of its own (§3.1).

Transport: JSON-RPC over stdio so Claude Desktop, Claude Code and any
MCP-capable agent can connect directly. Runtime: request → handler →
use case → response.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from typing import Any

from pydantic import BaseModel

from .container import Container, build_container

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Protocol messages
# ---------------------------------------------------------------------


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


# ---------------------------------------------------------------------
# Tool and resource schemas — stable contract, exercised in round-trip tests
# ---------------------------------------------------------------------


_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "cache.store",
        "description": "Store a query/response pair in the cache.",
        "inputSchema": {
            "type": "object",
            "required": ["query", "response"],
            "properties": {
                "query": {"type": "string"},
                "response": {"type": "string"},
                "context": {"type": "object"},
                "ttl_seconds": {"type": "integer", "minimum": 0},
            },
        },
    },
    {
        "name": "cache.invalidate",
        "description": "Invalidate a specific cache entry by key.",
        "inputSchema": {
            "type": "object",
            "required": ["cache_key"],
            "properties": {
                "cache_key": {"type": "string"},
                "reason": {"type": "string", "default": "user_request"},
            },
        },
    },
    {
        "name": "cache.purge_expired",
        "description": "Purge all expired cache entries.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "cache.clear_all",
        "description": "Clear every cache entry. Requires confirm=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "default": False},
            },
        },
    },
]


_RESOURCE_SCHEMAS: list[dict[str, Any]] = [
    {
        "uri": "cache://stats",
        "name": "Cache statistics",
        "description": "Hit rate, totals, estimated savings.",
        "mimeType": "application/json",
    },
    {
        "uri": "cache://entries",
        "name": "Cache entries",
        "description": "List of stored entries (keys + metadata).",
        "mimeType": "application/json",
    },
    {
        "uri": "cache://entry/{cache_key}",
        "name": "Single cache entry",
        "description": "Fetch one entry by key.",
        "mimeType": "application/json",
    },
    {
        "uri": "cache://query/{prompt}",
        "name": "Query by prompt",
        "description": "Look up a cached response for a raw prompt.",
        "mimeType": "application/json",
    },
]


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------


class MCPCacheServer:
    """MCP server handler.

    Owns the wired :class:`Container` and translates JSON-RPC requests
    into use-case invocations. Stateful across a connection but safe to
    re-instantiate per process.
    """

    def __init__(self, container: Container | None = None) -> None:
        self._container = container or build_container()

    # ------------------------------------------------------------------
    # Tools (writes)
    # ------------------------------------------------------------------

    async def tool_cache_store(
        self,
        query: str,
        response: str,
        context: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        cache_key = _key_for(query, context)
        await self._container.store_cache.execute(
            key=cache_key,
            value=response.encode("utf-8"),
            ttl_seconds=ttl_seconds,
            context=context,
        )
        return {"cache_key": cache_key, "stored": True}

    async def tool_cache_invalidate(
        self, cache_key: str, reason: str = "user_request"
    ) -> dict[str, Any]:
        await self._container.invalidate_cache.invalidate_key(cache_key, reason)
        return {"cache_key": cache_key, "invalidated": True, "reason": reason}

    async def tool_cache_purge_expired(self) -> dict[str, Any]:
        count = await self._container.invalidate_cache.purge_expired()
        return {"purged": count}

    async def tool_cache_clear_all(self, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            return {"cleared": False, "reason": "confirm flag required"}
        await self._container.storage.clear()
        return {"cleared": True}

    # ------------------------------------------------------------------
    # Resources (reads)
    # ------------------------------------------------------------------

    async def resource_stats(self) -> dict[str, Any]:
        metrics = await self._container.cache_metrics.get_metrics()
        size_bytes = await self._container.storage.get_size_bytes()
        return {
            "hit_rate": metrics.get("hit_rate", 0.0),
            "total_hits": metrics.get("total_hits", 0),
            "total_misses": metrics.get("total_misses", 0),
            "total_evictions": metrics.get("total_evictions", 0),
            "cache_size_bytes": size_bytes,
            "total_cost_saved": metrics.get("total_cost_saved", 0.0),
            "total_tokens_saved": metrics.get("total_tokens_saved", 0),
        }

    async def resource_list_entries(
        self, limit: int = 50, include_expired: bool = False
    ) -> dict[str, Any]:
        keys = await self._container.storage.get_all_keys()
        entries: list[dict[str, Any]] = []
        for key in keys[:limit]:
            entry = await self._container.storage.get(key)
            if entry and (include_expired or not entry.is_expired()):
                entries.append(
                    {
                        "cache_key": entry.key,
                        "size_bytes": entry.get_size_bytes(),
                        "is_expired": entry.is_expired(),
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    }
                )
        return {"entries": entries, "count": len(entries)}

    async def resource_get_entry(self, cache_key: str) -> dict[str, Any] | None:
        entry = await self._container.storage.get(cache_key)
        if entry is None:
            return None
        return {
            "cache_key": entry.key,
            "value": entry.value.decode("utf-8", errors="replace"),
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "is_expired": entry.is_expired(),
            "size_bytes": entry.get_size_bytes(),
        }

    async def resource_query(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        result = await self._container.query_cache.execute(prompt, context=context)
        return {
            "hit": result.hit,
            "value": result.value.decode("utf-8", errors="replace")
            if result.hit and result.value
            else None,
            "cache_key": result.entry_key,
            "response_time_ms": result.response_time_ms,
        }

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

    async def handle(self, request: MCPRequest) -> MCPResponse:
        try:
            if request.method == "initialize":
                return MCPResponse(id=request.id, result=self._initialize_reply())
            if request.method == "tools/list":
                return MCPResponse(id=request.id, result={"tools": _TOOL_SCHEMAS})
            if request.method == "resources/list":
                return MCPResponse(id=request.id, result={"resources": _RESOURCE_SCHEMAS})
            if request.method == "tools/call":
                return MCPResponse(id=request.id, result=await self._dispatch_tool(request.params))
            if request.method == "resources/read":
                return MCPResponse(
                    id=request.id, result=await self._dispatch_resource(request.params)
                )
            return MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Unknown method: {request.method}"},
            )
        except ValueError as err:
            # Input validation error — distinct from internal failures (§4.2).
            return MCPResponse(
                id=request.id, error={"code": -32602, "message": f"Invalid params: {err}"}
            )
        except Exception:
            logger.exception("MCP request failed: %s", request.method)
            return MCPResponse(id=request.id, error={"code": -32603, "message": "Internal error"})

    def _initialize_reply(self) -> dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "aicache", "version": "0.3.0"},
        }

    async def _dispatch_tool(self, params: dict[str, Any] | None) -> dict[str, Any]:
        if not params or "name" not in params:
            raise ValueError("tools/call requires 'name'")
        name = params["name"]
        args = params.get("arguments") or {}

        handlers = {
            "cache.store": self.tool_cache_store,
            "cache.invalidate": self.tool_cache_invalidate,
            "cache.purge_expired": self.tool_cache_purge_expired,
            "cache.clear_all": self.tool_cache_clear_all,
        }
        if name not in handlers:
            raise ValueError(f"Unknown tool: {name}")
        result = await handlers[name](**args)
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    async def _dispatch_resource(self, params: dict[str, Any] | None) -> dict[str, Any]:
        if not params or "uri" not in params:
            raise ValueError("resources/read requires 'uri'")
        uri = params["uri"]

        if uri == "cache://stats":
            payload = await self.resource_stats()
        elif uri == "cache://entries":
            payload = await self.resource_list_entries(
                limit=params.get("limit", 50),
                include_expired=params.get("include_expired", False),
            )
        elif uri.startswith("cache://entry/"):
            cache_key = uri[len("cache://entry/") :]
            payload = await self.resource_get_entry(cache_key)
        elif uri.startswith("cache://query/"):
            prompt = uri[len("cache://query/") :]
            payload = await self.resource_query(prompt, params.get("context"))
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(payload),
                }
            ]
        }


# ---------------------------------------------------------------------
# Stdio transport — entry point for `python -m aicache.infrastructure.mcp_server`
# ---------------------------------------------------------------------


def run_stdio(server: MCPCacheServer | None = None) -> None:  # pragma: no cover
    """Blocking stdio event loop for MCP clients."""
    import asyncio

    server = server or MCPCacheServer()
    loop = asyncio.new_event_loop()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = MCPRequest.model_validate_json(line)
            except Exception as err:
                logger.warning("Invalid request: %s", err)
                continue
            response = loop.run_until_complete(server.handle(request))
            sys.stdout.write(response.model_dump_json(exclude_none=True) + "\n")
            sys.stdout.flush()
    finally:
        loop.close()


def _key_for(query: str, context: dict[str, Any] | None) -> str:
    hasher = hashlib.sha256()
    hasher.update(query.encode("utf-8"))
    if context:
        hasher.update(json.dumps(context, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


# Back-compat exports — some tests still import MCPConnection from here
MCPConnection = MCPCacheServer


if __name__ == "__main__":  # pragma: no cover
    run_stdio()
