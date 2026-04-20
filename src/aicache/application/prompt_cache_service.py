"""
Application Service: Prompt Caching Integration

Bridges provider-specific prompt caching (OpenAI/Anthropic/Google) with
the core cache system. Tracks savings across providers and generates reports.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..domain.prompt_caching import (
    CacheProvider,
    MultiProviderPromptCachePort,
    PromptCacheConfig,
    PromptCacheResult,
)

logger = logging.getLogger(__name__)


class PromptCacheService:
    """
    Application service that integrates provider-specific prompt caching
    with the core aicache system.

    Responsibilities:
    - Route cache operations to the correct provider adapter
    - Track cumulative savings across all providers
    - Generate cost reports
    - Persist savings history to disk
    """

    def __init__(self, data_dir: str | None = None):
        self._multi_provider = MultiProviderPromptCachePort()
        self._data_dir = Path(data_dir or Path.home() / ".cache" / "aicache" / "prompt_cache")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._data_dir / "savings_history.json"
        self._history = self._load_history()

    def _load_history(self) -> dict[str, Any]:
        """Load savings history from disk."""
        if self._history_file.exists():
            try:
                with open(self._history_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "total_queries": 0,
            "total_cache_hits": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "by_provider": {},
            "daily_savings": [],
        }

    def _save_history(self) -> None:
        """Persist savings history to disk."""
        try:
            with open(self._history_file, "w") as f:
                json.dump(self._history, f, indent=2)
        except OSError:
            logger.warning("Failed to save prompt cache history")

    def set_provider(self, provider: str) -> bool:
        """Set the active LLM provider."""
        try:
            cache_provider = CacheProvider(provider)
            self._multi_provider.set_provider(cache_provider)
            return True
        except ValueError:
            logger.error(f"Unknown provider: {provider}")
            return False

    @property
    def current_provider(self) -> str:
        """Get the current active provider name."""
        return self._multi_provider._current_provider.value

    def get_provider_config(self) -> PromptCacheConfig:
        """Get current provider's cache configuration."""
        return self._multi_provider.get_cache_config()

    async def check_and_cache(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> PromptCacheResult:
        """
        Check if messages can benefit from prompt caching,
        and cache them if possible.

        Returns the cache result with savings information.
        """
        start_time = time.time()

        # Check cache
        result = await self._multi_provider.check_cache(messages)

        # Attempt to cache for future use
        if not result.cached:
            await self._multi_provider.cache_prompt(messages)

        # Record in history
        duration_ms = (time.time() - start_time) * 1000
        self._record_operation(result, duration_ms)

        return result

    def _record_operation(self, result: PromptCacheResult, duration_ms: float) -> None:
        """Record a cache operation in history."""
        provider = result.provider.value
        self._history["total_queries"] += 1

        if result.cache_hit:
            self._history["total_cache_hits"] += 1
            self._history["total_tokens_saved"] += result.cached_tokens

        # Per-provider tracking
        if provider not in self._history["by_provider"]:
            self._history["by_provider"][provider] = {
                "queries": 0,
                "hits": 0,
                "tokens_saved": 0,
                "cost_saved": 0.0,
            }

        prov = self._history["by_provider"][provider]
        prov["queries"] += 1
        if result.cache_hit:
            prov["hits"] += 1
            prov["tokens_saved"] += result.cached_tokens

        # Daily rollup
        today = datetime.now().strftime("%Y-%m-%d")
        daily = self._history["daily_savings"]
        if not daily or daily[-1].get("date") != today:
            daily.append(
                {
                    "date": today,
                    "queries": 0,
                    "hits": 0,
                    "tokens_saved": 0,
                }
            )
        daily[-1]["queries"] += 1
        if result.cache_hit:
            daily[-1]["hits"] += 1
            daily[-1]["tokens_saved"] += result.cached_tokens

        # Keep only last 90 days
        if len(daily) > 90:
            self._history["daily_savings"] = daily[-90:]

        self._save_history()

    def get_savings_report(self, days: int = 30) -> dict[str, Any]:
        """Generate a savings report for the specified period."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        daily = self._history.get("daily_savings", [])
        period_data = [d for d in daily if d.get("date", "") >= cutoff]

        total_queries = sum(d["queries"] for d in period_data)
        total_hits = sum(d["hits"] for d in period_data)
        total_tokens = sum(d["tokens_saved"] for d in period_data)
        hit_rate = (total_hits / total_queries * 100) if total_queries > 0 else 0.0

        # Rough cost estimates per provider
        provider_savings = {}
        for provider, pdata in self._history.get("by_provider", {}).items():
            if provider == "openai":
                savings_per_1k = 0.075  # 50% savings
            elif provider == "anthropic":
                savings_per_1k = 0.0027  # 90% savings
            else:
                savings_per_1k = 0.000375  # 75% savings

            estimated_savings = (pdata["tokens_saved"] / 1000) * savings_per_1k
            provider_savings[provider] = {
                "queries": pdata["queries"],
                "hits": pdata["hits"],
                "tokens_saved": pdata["tokens_saved"],
                "estimated_savings": round(estimated_savings, 4),
                "hit_rate": (pdata["hits"] / pdata["queries"] * 100)
                if pdata["queries"] > 0
                else 0.0,
            }

        total_estimated_savings = sum(p["estimated_savings"] for p in provider_savings.values())

        return {
            "period_days": days,
            "total_queries": total_queries,
            "total_hits": total_hits,
            "hit_rate_percent": round(hit_rate, 1),
            "total_tokens_saved": total_tokens,
            "total_estimated_savings": round(total_estimated_savings, 4),
            "monthly_projection": round(total_estimated_savings * (30 / max(days, 1)), 2),
            "by_provider": provider_savings,
            "daily_trend": period_data[-7:] if period_data else [],
            "all_time": {
                "queries": self._history["total_queries"],
                "hits": self._history["total_cache_hits"],
                "tokens_saved": self._history["total_tokens_saved"],
            },
        }

    def get_provider_info(self) -> list[dict[str, Any]]:
        """Get info about all configured providers."""
        providers = []
        for provider_enum in CacheProvider:
            if provider_enum in self._multi_provider._adapters:
                adapter = self._multi_provider._adapters[provider_enum]
                config = adapter.get_cache_config()
                is_active = provider_enum == self._multi_provider._current_provider
                prov_stats = self._history.get("by_provider", {}).get(provider_enum.value, {})

                providers.append(
                    {
                        "name": provider_enum.value,
                        "active": is_active,
                        "auto_cache": config.auto_cache_enabled,
                        "min_tokens": config.cache_min_tokens,
                        "ttl_seconds": config.cache_ttl_seconds,
                        "queries": prov_stats.get("queries", 0),
                        "hits": prov_stats.get("hits", 0),
                        "tokens_saved": prov_stats.get("tokens_saved", 0),
                    }
                )
        return providers
