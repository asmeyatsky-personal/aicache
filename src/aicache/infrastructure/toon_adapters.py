"""
TOON Infrastructure Adapters

Concrete implementations for TOON persistence and serialization.
Provides storage, retrieval, and export capabilities for TOON operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack

from ..domain.ports import TOONRepositoryPort
from ..domain.toon import TOONAnalytics, TOONCacheOperation, TOONOperationType

logger = logging.getLogger(__name__)


class InMemoryTOONRepositoryAdapter(TOONRepositoryPort):
    """
    In-memory TOON repository for development and testing.

    Stores TOON operations in memory with no persistence.
    """

    def __init__(self):
        self.toons: dict[str, TOONCacheOperation] = {}

    async def save_toon(self, toon: TOONCacheOperation) -> bool:
        """Save TOON operation to memory."""
        try:
            self.toons[toon.operation_id] = toon
            return True
        except Exception as e:
            logger.error(f"Error saving TOON: {e}")
            return False

    async def get_toon(self, operation_id: str) -> TOONCacheOperation | None:
        """Retrieve TOON from memory."""
        return self.toons.get(operation_id)

    async def get_all_toons(self, limit: int | None = None) -> list[TOONCacheOperation]:
        """Get all TOONs from memory."""
        toons = list(self.toons.values())
        if limit:
            toons = toons[-limit:]  # Return most recent
        return toons

    async def get_toons_by_type(
        self, operation_type: TOONOperationType
    ) -> list[TOONCacheOperation]:
        """Get TOONs filtered by operation type."""
        return [t for t in self.toons.values() if t.operation_type == operation_type]

    async def delete_toon(self, operation_id: str) -> bool:
        """Delete TOON from memory."""
        if operation_id in self.toons:
            del self.toons[operation_id]
            return True
        return False

    async def clear_toons(self) -> int:
        """Clear all TOONs from memory."""
        count = len(self.toons)
        self.toons.clear()
        return count


class FileSystemTOONRepositoryAdapter(TOONRepositoryPort):
    """
    File system based TOON repository.

    Stores TOON operations as JSON files in a directory hierarchy.
    Format: toon_data/YYYY/MM/DD/operation_id.json
    """

    def __init__(self, base_dir: str = "~/.cache/aicache/toon_data"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_toon_path(self, operation_id: str) -> Path:
        """Get file path for TOON operation."""
        return self.base_dir / operation_id[:2] / f"{operation_id}.json"

    async def save_toon(self, toon: TOONCacheOperation) -> bool:
        """Save TOON to JSON file."""
        try:
            path = self._get_toon_path(toon.operation_id)
            path.parent.mkdir(parents=True, exist_ok=True)

            toon_dict = toon.to_dict()
            with open(path, "w") as f:
                json.dump(toon_dict, f, indent=2)

            logger.debug(f"Saved TOON to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving TOON: {e}")
            return False

    async def get_toon(self, operation_id: str) -> TOONCacheOperation | None:
        """Retrieve TOON from JSON file."""
        try:
            path = self._get_toon_path(operation_id)
            if not path.exists():
                return None

            with open(path) as f:
                data = json.load(f)

            return self._dict_to_toon(data)
        except Exception as e:
            logger.error(f"Error loading TOON: {e}")
            return None

    async def get_all_toons(self, limit: int | None = None) -> list[TOONCacheOperation]:
        """Get all TOONs from files."""
        toons = []
        try:
            for json_file in sorted(self.base_dir.glob("*/*.json"), reverse=True):
                if limit and len(toons) >= limit:
                    break
                with open(json_file) as f:
                    data = json.load(f)
                    toon = self._dict_to_toon(data)
                    if toon:
                        toons.append(toon)
        except Exception as e:
            logger.error(f"Error loading TOONs: {e}")

        return toons

    async def get_toons_by_type(
        self, operation_type: TOONOperationType
    ) -> list[TOONCacheOperation]:
        """Get TOONs filtered by operation type."""
        toons = []
        try:
            for json_file in self.base_dir.glob("*/*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    if data.get("operation_type") == operation_type.value:
                        toon = self._dict_to_toon(data)
                        if toon:
                            toons.append(toon)
        except Exception as e:
            logger.error(f"Error filtering TOONs: {e}")

        return toons

    async def delete_toon(self, operation_id: str) -> bool:
        """Delete TOON file."""
        try:
            path = self._get_toon_path(operation_id)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting TOON: {e}")
            return False

    async def clear_toons(self) -> int:
        """Clear all TOON files."""
        count = 0
        try:
            for json_file in self.base_dir.glob("*/*.json"):
                json_file.unlink()
                count += 1
        except Exception as e:
            logger.error(f"Error clearing TOONs: {e}")

        return count

    @staticmethod
    def _dict_to_toon(data: dict[str, Any]) -> TOONCacheOperation | None:
        """Convert dictionary to TOON object."""
        try:
            from ..domain.toon import TokenDelta, TOONOperationType

            token_delta_data = data.get("token_delta", {})
            token_delta = TokenDelta(
                prompt_tokens=token_delta_data.get("prompt_tokens", 0),
                completion_tokens=token_delta_data.get("completion_tokens", 0),
                cached_tokens=token_delta_data.get("cached_tokens", 0),
                saved_total=token_delta_data.get("saved_total", 0),
                cost_without_cache=token_delta_data.get("cost_without_cache", 0.0),
                cost_with_cache=token_delta_data.get("cost_with_cache", 0.0),
                cost_saved=token_delta_data.get("cost_saved", 0.0),
            )
            return TOONCacheOperation(
                operation_id=data.get("operation_id", ""),
                timestamp=datetime.fromisoformat(data["timestamp"])
                if isinstance(data.get("timestamp"), str)
                else datetime.now(),
                operation_type=TOONOperationType(data.get("operation_type", "cache_miss")),
                original_query=data.get("original_query", ""),
                normalized_query=data.get("normalized_query", ""),
                query_hash=data.get("query_hash", ""),
                token_delta=token_delta,
                model=data.get("model", ""),
                cache_key=data.get("cache_key"),
                similarity_score=data.get("similarity_score"),
                duration_ms=data.get("duration_ms", 0.0),
                context=data.get("context"),
            )
        except Exception as e:
            logger.error(f"Error converting dict to TOON: {e}")
            return None


class TOONExportService:
    """
    Service for exporting TOON data in various formats.

    Supports JSON, CSV, and compact binary formats.
    """

    def __init__(self, repository: TOONRepositoryPort):
        self.repository = repository

    async def export_to_json(self, limit: int | None = None) -> str:
        """Export TOONs as JSON array."""
        toons = await self.repository.get_all_toons(limit)
        toon_dicts = [t.to_dict() for t in toons]
        return json.dumps(toon_dicts, indent=2)

    async def export_to_jsonl(self, limit: int | None = None) -> str:
        """Export TOONs as JSONL (one JSON per line)."""
        toons = await self.repository.get_all_toons(limit)
        lines = [json.dumps(t.to_dict()) for t in toons]
        return "\n".join(lines)

    async def export_to_csv(self, limit: int | None = None) -> str:
        """Export TOONs as CSV."""
        toons = await self.repository.get_all_toons(limit)
        if not toons:
            return ""

        # CSV header
        compact_dict = toons[0].to_compact_dict()
        headers = list(compact_dict.keys())
        csv_lines = [",".join(headers)]

        # CSV rows
        for toon in toons:
            row_dict = toon.to_compact_dict()
            row = [str(row_dict.get(h, "")) for h in headers]
            csv_lines.append(",".join(row))

        return "\n".join(csv_lines)

    async def export_to_msgpack(self, limit: int | None = None) -> bytes:
        """Export TOONs as binary msgpack format (compact)."""
        toons = await self.repository.get_all_toons(limit)
        compact_dicts = [t.to_compact_dict() for t in toons]
        return msgpack.packb(compact_dicts)

    async def export_analytics_json(self, analytics: TOONAnalytics) -> str:
        """Export analytics as JSON."""
        return analytics.to_json()

    async def export_analytics_csv(self, analytics: TOONAnalytics) -> str:
        """Export analytics summary as CSV."""
        data = analytics.to_dict()
        ops = data.get("operations", {})

        csv_lines = [
            "Metric,Value",
            f"Total Operations,{ops.get('total', 0)}",
            f"Exact Hits,{ops.get('exact_hits', 0)}",
            f"Semantic Hits,{ops.get('semantic_hits', 0)}",
            f"Intent Hits,{ops.get('intent_hits', 0)}",
            f"Misses,{ops.get('misses', 0)}",
            f"Hit Rate,{ops.get('hit_rate_percent', 0)}%",
            f"Semantic Hit Rate,{ops.get('semantic_hit_rate_percent', 0)}%",
            f"Total Tokens Saved,{data.get('tokens', {}).get('total_saved', 0)}",
            f"Total Cost Saved,${data.get('costs', {}).get('total_saved', 0)}",
            f"Average ROI Score,{data.get('insights', {}).get('average_roi_score', 0)}",
        ]

        return "\n".join(csv_lines)


class TOONQueryBuilder:
    """
    Builder for constructing complex TOON queries.

    Supports filtering by operation type, time range, token savings, etc.
    """

    def __init__(self, repository: TOONRepositoryPort):
        self.repository = repository
        self.filters = []

    def with_operation_type(self, operation_type: TOONOperationType) -> "TOONQueryBuilder":
        """Filter by operation type."""
        self.filters.append(lambda t: t.operation_type == operation_type)
        return self

    def with_min_tokens_saved(self, min_tokens: int) -> "TOONQueryBuilder":
        """Filter by minimum tokens saved."""
        self.filters.append(lambda t: t.token_delta.saved_total >= min_tokens)
        return self

    def with_min_similarity(self, min_score: float) -> "TOONQueryBuilder":
        """Filter by minimum similarity score."""
        self.filters.append(
            lambda t: (
                t.semantic_data.similarity_score and t.semantic_data.similarity_score >= min_score
            )
        )
        return self

    def with_time_range(self, start: datetime, end: datetime) -> "TOONQueryBuilder":
        """Filter by time range."""
        self.filters.append(lambda t: start <= t.timestamp <= end)
        return self

    def with_optimization_level(self, level) -> "TOONQueryBuilder":
        """Filter by optimization level."""
        self.filters.append(lambda t: t.optimization_insight.optimization_level == level)
        return self

    async def execute(self) -> list[TOONCacheOperation]:
        """Execute query with all filters."""
        toons = await self.repository.get_all_toons()

        for filter_fn in self.filters:
            toons = [t for t in toons if filter_fn(t)]

        return toons
