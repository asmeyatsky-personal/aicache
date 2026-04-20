"""
AI Cache Core - Lightweight caching module for AI CLI tools.

This module provides basic file-based caching functionality without requiring
heavy ML dependencies. Use this for simple exact-match caching.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Lightweight cache entry for basic caching."""

    key: str
    value: str
    timestamp: float
    ttl_seconds: int | None = None
    access_count: int = 0
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() > (self.timestamp + self.ttl_seconds)

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class CoreCache:
    """
    Lightweight file-based cache for AI CLI tools.

    Provides basic exact-match caching without semantic features.
    Perfect for getting started with minimal dependencies.
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/aicache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / ".index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    self._index = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f)
        except OSError:
            logger.warning("Failed to save cache index")

    def _get_cache_key(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        """Generate deterministic cache key."""
        hasher = hashlib.sha256()
        hasher.update(prompt.encode("utf-8"))
        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            hasher.update(sorted_context.encode("utf-8"))
        return hasher.hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for key, with path traversal protection."""
        cache_file = (self.cache_dir / f"{cache_key}.json").resolve()
        if not str(cache_file).startswith(str(self.cache_dir.resolve())):
            raise ValueError("Invalid cache key: path traversal detected")
        return cache_file

    def get(self, prompt: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """
        Get cached response for prompt (backward compatible).

        Args:
            prompt: The original prompt
            context: Optional context dictionary

        Returns:
            Cache entry dict or None if not found/expired
        """
        cache_key = self._get_cache_key(prompt, context)
        cache_file = self._get_cache_file(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            entry = CacheEntry(**data)

            if entry.is_expired():
                self.delete(cache_key)
                return None

            entry.touch()

            with open(cache_file, "w") as f:
                json.dump(asdict(entry), f)

            if cache_key in self._index:
                self._index[cache_key]["last_accessed"] = entry.last_accessed
                self._index[cache_key]["access_count"] = entry.access_count
                self._save_index()

            # Add backward compatibility keys
            data["prompt"] = prompt
            data["response"] = entry.value
            data["context"] = context

            return data

        except (OSError, json.JSONDecodeError, TypeError):
            self.delete(cache_key)
            return None

    def get_value(self, prompt: str, context: dict[str, Any] | None = None) -> str | None:
        """
        Get just the cached response value (new API).

        Returns:
            Cached response string or None if not found/expired
        """
        cache_key = self._get_cache_key(prompt, context)
        cache_file = self._get_cache_file(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            entry = CacheEntry(**data)

            if entry.is_expired():
                self.delete(cache_key)
                return None

            return entry.value

        except (OSError, json.JSONDecodeError, TypeError):
            self.delete(cache_key)
            return None

    def inspect(self, cache_key: str) -> dict[str, Any] | None:
        """
        Inspect a cache entry by key (backward compatible).
        """
        cache_file = self._get_cache_file(cache_key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
                # Add backward compatibility keys
                if "value" in data:
                    data["response"] = data["value"]
                # We don't have prompt stored, but we can derive something
                if "prompt" not in data:
                    data["prompt"] = ""  # Placeholder
                return data
        except (OSError, json.JSONDecodeError):
            return None

    def prune(self, max_age_days: int = 30, max_size_mb: int | None = None) -> int:
        """
        Prune expired cache entries (backward compatible).

        Args:
            max_age_days: Delete entries older than this many days
            max_size_mb: Optional size limit

        Returns:
            Number of entries pruned
        """
        import time

        cutoff_time = time.time() - (max_age_days * 86400)
        pruned = 0

        for cache_key in list(self._index.keys()):
            cache_file = self._get_cache_file(cache_key)
            if not cache_file.exists():
                continue

            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", 0)

                if timestamp < cutoff_time:
                    cache_file.unlink()
                    del self._index[cache_key]
                    pruned += 1
            except (OSError, json.JSONDecodeError):
                continue

        if pruned > 0:
            self._save_index()

        return pruned

    def set(
        self,
        prompt: str,
        response: str,
        context: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Cache a response for the given prompt.

        Args:
            prompt: The original prompt
            response: The response to cache
            context: Optional context dictionary
            ttl_seconds: Optional time-to-live in seconds
        """
        cache_key = self._get_cache_key(prompt, context)
        cache_file = self._get_cache_file(cache_key)

        entry = CacheEntry(
            key=cache_key,
            value=response,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds,
            access_count=0,
            last_accessed=time.time(),
        )

        try:
            # Store data - use asdict but add backward compat keys
            data = asdict(entry)

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            # Update index
            self._index[cache_key] = {
                "created_at": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response_length": len(response),
            }
            self._save_index()

        except OSError:
            logger.warning(f"Failed to write cache entry: {cache_key}")

    def delete(self, cache_key: str) -> bool:
        """Delete a cache entry by key."""
        cache_file = self._get_cache_file(cache_key)
        success = False

        if cache_file.exists():
            try:
                cache_file.unlink()
                success = True
            except OSError:
                pass

        if cache_key in self._index:
            del self._index[cache_key]
            self._save_index()

        return success

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == ".index.json":
                continue
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        self._index = {}
        self._save_index()
        return count

    def list(self, limit: int | None = None, verbose: bool = False) -> list[dict[str, Any]]:
        """
        List cache entries.

        Args:
            limit: Maximum number of entries to return
            verbose: If False, return just keys (backward compat). If True, return full dicts.

        Returns:
            List of cache entry metadata
        """
        entries = []
        for cache_key, metadata in list(self._index.items())[
            : limit if limit is not None else len(self._index)
        ]:
            cache_file = self._get_cache_file(cache_key)
            if cache_file.exists():
                if not verbose:
                    # Backward compatibility: return just the keys
                    entries.append(cache_key)
                else:
                    # Read prompt from file for backward compatibility
                    prompt = metadata.get("prompt_preview", "").replace("...", "")
                    try:
                        with open(cache_file) as f:
                            file_data = json.load(f)
                            prompt = file_data.get("prompt", prompt)
                    except (json.JSONDecodeError, OSError):
                        pass

                    entry_data = {
                        "cache_key": cache_key,
                        "prompt": prompt,
                        **metadata,
                        "created_at_readable": datetime.fromtimestamp(
                            metadata["created_at"]
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "last_accessed_readable": datetime.fromtimestamp(
                            metadata["last_accessed"]
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if metadata["last_accessed"]
                        else None,
                    }
                    entries.append(entry_data)

        return sorted(
            entries,
            key=lambda x: x["last_accessed"] if isinstance(x, dict) else 0,
            reverse=True,
        )

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._index)
        total_accesses = sum(meta.get("access_count", 0) for meta in self._index.values())

        # Calculate cache size
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == ".index.json":
                continue
            try:
                total_size += cache_file.stat().st_size
            except OSError:
                pass

        return {
            "total_entries": total_entries,
            "num_entries": total_entries,  # Backward compatibility
            "total_accesses": total_accesses,
            "total_size": total_size,  # Backward compatibility
            "cache_size_bytes": total_size,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }


# Convenience function for quick usage
def get_cache() -> CoreCache:
    """Get a default cache instance."""
    return CoreCache()
