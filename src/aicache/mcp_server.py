"""
AI Cache MCP Server - Model Context Protocol Integration

This module provides an MCP (Model Context Protocol) server that exposes
aicache functionality as tools for LLMs like Claude Desktop or GPT-4o.

Usage:
    # Run as standalone server
    python -m aicache.mcp_server

    # Or integrate into your application
    from aicache.mcp_server import AICacheMCPServer

    server = AICacheMCPServer()
    server.run()
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# MCP Protocol types (simplified for compatibility)
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object

    class Field:
        def __init__(self, default=None, description=""):
            self.default = default
            self.description = description


from .config import get_config
from .core.cache import CoreCache

logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """MCP request message"""

    jsonrpc: str = "2.0"
    id: int | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    """MCP response message"""

    jsonrpc: str = "2.0"
    id: int | None = None
    result: Any | None = None
    error: dict[str, str] | None = None


class MCPTool(BaseModel):
    """MCP tool definition"""

    name: str
    description: str
    inputSchema: dict[str, Any]


class MCPConnection:
    """MCP server connection handler"""

    def __init__(self):
        self.cache = CoreCache()
        self.config = get_config()
        self._initialize()

    def _initialize(self):
        """Initialize cache directory"""
        cache_dir = self.config.get("cache_dir", "~/.cache/aicache")
        Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # ========== TOOL HANDLERS ==========

    def handle_aicache_get(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Get a cached response for a prompt.

        Args:
            prompt: The prompt to look up
            context: Optional context dictionary

        Returns:
            Cached response or null if not found
        """
        result = self.cache.get(prompt, context)

        if result:
            return {
                "found": True,
                "response": result,
                "cache_key": self.cache._get_cache_key(prompt, context),
            }
        return {"found": False, "response": None}

    def handle_aicache_set(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Cache a response for a prompt.

        Args:
            prompt: The prompt to cache
            response: The response to store
            context: Optional context dictionary
            ttl_seconds: Optional time-to-live in seconds

        Returns:
            Success status and cache key
        """
        self.cache.set(prompt, response, context, ttl_seconds)
        return {
            "success": True,
            "cache_key": self.cache._get_cache_key(prompt, context),
        }

    def handle_aicache_list(self, limit: int = 10, verbose: bool = False) -> dict[str, Any]:
        """
        List cached entries.

        Args:
            limit: Maximum number of entries to return
            verbose: Include detailed information

        Returns:
            List of cache entries
        """
        entries = self.cache.list(limit=limit, verbose=True)

        if verbose:
            return {"entries": entries, "total": len(entries)}

        # Simplified output for non-verbose
        simplified = []
        for entry in entries:
            simplified.append(
                {
                    "key": entry.get("cache_key", "")[:16] + "...",
                    "preview": entry.get("prompt_preview", "N/A")[:50],
                    "accesses": entry.get("access_count", 0),
                }
            )

        return {"entries": simplified, "total": len(simplified)}

    def handle_aicache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics including entries, size, and estimated savings
        """
        stats = self.cache.stats()

        # Add estimated savings (rough estimate: $0.002 per cached response)
        estimated_savings = stats["total_accesses"] * 0.002

        return {
            "total_entries": stats["total_entries"],
            "total_accesses": stats["total_accesses"],
            "cache_size_bytes": stats["cache_size_bytes"],
            "cache_size_mb": stats["cache_size_mb"],
            "estimated_cost_saved": round(estimated_savings, 4),
            "monthly_projection": round(estimated_savings * 30, 2),
        }

    def handle_aicache_clear(self, confirm: bool = False) -> dict[str, Any]:
        """
        Clear all cache entries.

        Args:
            confirm: Must be True to actually clear (safety feature)

        Returns:
            Number of entries cleared
        """
        if not confirm:
            return {"success": False, "message": "Must set confirm=true to clear cache"}

        count = self.cache.clear()
        return {"success": True, "cleared": count}

    def handle_aicache_delete(self, cache_key: str) -> dict[str, Any]:
        """
        Delete a specific cache entry.

        Args:
            cache_key: The cache key to delete

        Returns:
            Success status
        """
        success = self.cache.delete(cache_key)
        return {"success": success}

    def handle_aicache_prune(self, days: int = 30, confirm: bool = False) -> dict[str, Any]:
        """
        Prune old cache entries.

        Args:
            days: Delete entries older than this many days
            confirm: Must be True to actually prune

        Returns:
            Number of entries pruned
        """
        if not confirm:
            return {"success": False, "message": "Must set confirm=true to prune cache"}

        entries = self.cache.list()
        cutoff_time = datetime.now().timestamp() - (days * 86400)

        pruned = 0
        for entry in entries:
            created_at = entry.get("created_at", 0)
            if created_at < cutoff_time:
                if self.cache.delete(entry["cache_key"]):
                    pruned += 1

        return {"success": True, "pruned": pruned}

    # ========== MCP PROTOCOL HANDLERS ==========

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "aicache", "version": "0.2.0"},
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """Handle MCP tools/list request"""
        return {
            "tools": [
                {
                    "name": "aicache_get",
                    "description": "Get a cached response for a prompt. Returns cached response if found, or null if not cached.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to look up",
                            },
                            "context": {
                                "type": "object",
                                "description": "Optional context dictionary",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "aicache_set",
                    "description": "Cache a response for a prompt. The response will be retrieved on subsequent identical prompts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to cache",
                            },
                            "response": {
                                "type": "string",
                                "description": "The response to store",
                            },
                            "context": {
                                "type": "object",
                                "description": "Optional context",
                            },
                            "ttl_seconds": {
                                "type": "integer",
                                "description": "Optional TTL in seconds",
                            },
                        },
                        "required": ["prompt", "response"],
                    },
                },
                {
                    "name": "aicache_list",
                    "description": "List all cached entries with optional detail level.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Max entries to return",
                                "default": 10,
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Include full details",
                                "default": False,
                            },
                        },
                    },
                },
                {
                    "name": "aicache_stats",
                    "description": "Get cache statistics including entries, size, and estimated cost savings.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "aicache_clear",
                    "description": "Clear all cache entries. Requires confirm=true to actually clear.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "confirm": {
                                "type": "boolean",
                                "description": "Must be true to clear",
                                "default": False,
                            }
                        },
                    },
                },
                {
                    "name": "aicache_delete",
                    "description": "Delete a specific cache entry by key.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cache_key": {
                                "type": "string",
                                "description": "The cache key to delete",
                            }
                        },
                        "required": ["cache_key"],
                    },
                },
                {
                    "name": "aicache_prune",
                    "description": "Prune (delete) old cache entries. Requires confirm=true.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Delete entries older than N days",
                                "default": 30,
                            },
                            "confirm": {
                                "type": "boolean",
                                "description": "Must be true to prune",
                                "default": False,
                            },
                        },
                    },
                },
            ]
        }

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tools/call request"""
        handlers = {
            "aicache_get": self.handle_aicache_get,
            "aicache_set": self.handle_aicache_set,
            "aicache_list": self.handle_aicache_list,
            "aicache_stats": self.handle_aicache_stats,
            "aicache_clear": self.handle_aicache_clear,
            "aicache_delete": self.handle_aicache_delete,
            "aicache_prune": self.handle_aicache_prune,
        }

        if name not in handlers:
            return {"error": f"Unknown tool: {name}"}

        try:
            # Validate arguments against handler's expected parameters
            import inspect

            handler = handlers[name]
            sig = inspect.signature(handler)
            valid_params = set(sig.parameters.keys())
            filtered_args = {k: v for k, v in arguments.items() if k in valid_params}
            result = handler(**filtered_args)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            logger.error(f"Error in tool call '{name}': {e}")
            return {"error": "Internal error processing tool call"}

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request"""
        try:
            result = None

            if request.method == "initialize":
                result = self.handle_initialize(request.params or {})
            elif request.method == "tools/list":
                result = self.handle_tools_list()
            elif request.method == "tools/call":
                tool_name = request.params.get("name") if request.params else None
                tool_args = request.params.get("arguments", {}) if request.params else {}
                result = self.handle_tool_call(tool_name, tool_args)
            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": "method_not_found",
                        "message": f"Unknown method: {request.method}",
                    },
                )

            return MCPResponse(id=request.id, result=result)

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                id=request.id, error={"code": "internal_error", "message": "Internal server error"}
            )


class AICacheMCPServer:
    """MCP Server for AI Cache"""

    def __init__(self, port: int = 8765):
        self.port = port
        self.connection = MCPConnection()

    def run_stdio(self):
        """Run server using stdin/stdout"""
        logger.info("Starting aicache MCP server (stdio mode)")

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request_data = json.loads(line.strip())
                request = MCPRequest(**request_data)
                response = self.connection.handle_request(request)

                print(json.dumps(response.model_dump(exclude_none=True)), flush=True)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error: {e}")

    def run_tcp(self):
        """Run server using TCP"""
        from asyncio import start_server

        async def handle_client(reader, writer):
            try:
                data = await reader.read(4096)
                request_data = json.loads(data.decode())
                request = MCPRequest(**request_data)
                response = self.connection.handle_request(request)

                writer.write(json.dumps(response.model_dump(exclude_none=True)).encode())
                await writer.drain()
            except Exception as e:
                logger.error(f"Client error: {e}")
            finally:
                writer.close()

        async def main():
            server = await start_server(handle_client, "localhost", self.port)
            logger.info(f"MCP server listening on localhost:{self.port}")
            async with server:
                await server.serve_forever()

        asyncio.run(main())


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Cache MCP Server")
    parser.add_argument("--mode", choices=["stdio", "tcp"], default="stdio", help="Server mode")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (if using tcp mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    server = AICacheMCPServer(port=args.port)

    if args.mode == "stdio":
        server.run_stdio()
    else:
        server.run_tcp()


if __name__ == "__main__":
    main()
