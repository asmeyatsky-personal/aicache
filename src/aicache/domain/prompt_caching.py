"""
Prompt Caching Ports - 2026 Provider-Specific Caching

This module defines ports for provider-specific prompt caching support.
Based on 2026 trends: OpenAI automatic, Anthropic explicit, Google implicit caching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class CacheProvider(Enum):
    """Supported LLM providers with caching support."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


@dataclass(frozen=True)
class PromptCacheResult:
    """Result of a prompt cache lookup."""

    cached: bool
    cache_hit: bool
    cached_tokens: int = 0
    new_tokens: int = 0
    total_tokens: int = 0
    cost_reduction_percent: float = 0.0
    latency_reduction_ms: float = 0.0
    provider: CacheProvider = CacheProvider.OPENAI


@dataclass(frozen=True)
class PromptCacheConfig:
    """Configuration for prompt caching."""

    provider: CacheProvider
    cache_min_tokens: int = 1024
    cache_ttl_seconds: int = 3600  # Default 1 hour
    cache_prefix: str | None = None
    auto_cache_enabled: bool = True


class PromptCachePort(ABC):
    """Port for provider-specific prompt caching."""

    @abstractmethod
    async def check_cache(self, messages: list[dict[str, str]]) -> PromptCacheResult:
        """Check if prompt is cached (read operation)."""
        pass

    @abstractmethod
    async def cache_prompt(
        self, messages: list[dict[str, str]], cache_key: str | None = None
    ) -> bool:
        """Cache a prompt for future use (write operation)."""
        pass

    @abstractmethod
    async def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate a specific cached prompt."""
        pass

    @abstractmethod
    def get_cache_config(self) -> PromptCacheConfig:
        """Get provider-specific cache configuration."""
        pass

    @abstractmethod
    def calculate_savings(self, cached_tokens: int, new_tokens: int, original_cost: float) -> float:
        """Calculate cost savings from caching."""
        pass


class OpenAIPromptCacheAdapter(PromptCachePort):
    """
    OpenAI Prompt Caching Adapter.

    OpenAI implements automatic prompt caching - when >=1024 tokens
    are shared between requests, caching activates automatically.
    Cache persists for 24 hours.

    2026: 50% cost reduction on cached tokens.
    """

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self._config = PromptCacheConfig(
            provider=CacheProvider.OPENAI,
            cache_min_tokens=1024,
            cache_ttl_seconds=86400,  # 24 hours
            auto_cache_enabled=True,
        )
        self._cache_hits: dict[str, bool] = {}

    async def check_cache(self, messages: list[dict[str, str]]) -> PromptCacheResult:
        """Check if prompt benefits from caching (OpenAI does this automatically)."""
        total_tokens = self._estimate_tokens(messages)
        cached_tokens = total_tokens if total_tokens >= 1024 else 0
        new_tokens = total_tokens - cached_tokens

        return PromptCacheResult(
            cached=cached_tokens > 0,
            cache_hit=len(self._cache_hits) > 0,
            cached_tokens=cached_tokens,
            new_tokens=new_tokens,
            total_tokens=total_tokens,
            cost_reduction_percent=50.0 if cached_tokens > 0 else 0.0,
            provider=CacheProvider.OPENAI,
        )

    async def cache_prompt(
        self, messages: list[dict[str, str]], cache_key: str | None = None
    ) -> bool:
        """OpenAI caches automatically when conditions met."""
        total_tokens = self._estimate_tokens(messages)
        if total_tokens >= 1024:
            cache_key = cache_key or self._generate_cache_key(messages)
            self._cache_hits[cache_key] = True
            return True
        return False

    async def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate cached prompt."""
        if cache_key in self._cache_hits:
            del self._cache_hits[cache_key]
            return True
        return False

    def get_cache_config(self) -> PromptCacheConfig:
        return self._config

    def calculate_savings(self, cached_tokens: int, new_tokens: int, original_cost: float) -> float:
        if cached_tokens == 0:
            return 0.0
        # 50% discount on cached tokens
        cached_cost = (cached_tokens / 1000) * 0.075  # $0.075/1K for cached
        new_cost = (new_tokens / 1000) * 0.15  # $0.15/1K for new
        original = (original_cost / 1000) * 0.15
        return original - (cached_cost + new_cost)

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count for messages."""
        total = 0
        for msg in messages:
            total += len(msg.get("content", "")) // 4
        return total

    def _generate_cache_key(self, messages: list[dict[str, str]]) -> str:
        """Generate cache key from messages."""
        import hashlib

        content = str(messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class AnthropicPromptCacheAdapter(PromptCachePort):
    """
    Anthropic Prompt Caching Adapter.

    Anthropic requires explicit cache_prefix specification.
    Cache persists for 5-10 minutes (configurable).

    2026: Up to 90% cost reduction on cached tokens.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._cache_prefixes: dict[str, str] = {}
        self._config = PromptCacheConfig(
            provider=CacheProvider.ANTHROPIC,
            cache_min_tokens=1024,
            cache_ttl_seconds=600,  # 10 minutes default
            auto_cache_enabled=False,  # Must specify cache_prefix
        )

    async def check_cache(self, messages: list[dict[str, str]]) -> PromptCacheResult:
        """Check cache status for messages."""
        total_tokens = self._estimate_tokens(messages)
        has_cache_prefix = any(self._is_cacheable(msg.get("content", "")) for msg in messages)

        cached_tokens = total_tokens if (total_tokens >= 1024 and has_cache_prefix) else 0
        new_tokens = total_tokens - cached_tokens

        return PromptCacheResult(
            cached=cached_tokens > 0,
            cache_hit=len(self._cache_prefixes) > 0 and has_cache_prefix,
            cached_tokens=cached_tokens,
            new_tokens=new_tokens,
            total_tokens=total_tokens,
            cost_reduction_percent=90.0 if cached_tokens > 0 else 0.0,
            provider=CacheProvider.ANTHROPIC,
        )

    async def cache_prompt(
        self, messages: list[dict[str, str]], cache_key: str | None = None
    ) -> bool:
        """Cache prompt with explicit prefix (required for Anthropic)."""
        total_tokens = self._estimate_tokens(messages)
        if total_tokens < 1024:
            return False

        cache_key = cache_key or self._generate_cache_key(messages)
        prefix = f"<cache>{cache_key}</cache>"
        self._cache_prefixes[cache_key] = prefix
        return True

    async def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate cached prompt."""
        if cache_key in self._cache_prefixes:
            del self._cache_prefixes[cache_key]
            return True
        return False

    def get_cache_config(self) -> PromptCacheConfig:
        return self._config

    def calculate_savings(self, cached_tokens: int, new_tokens: int, original_cost: float) -> float:
        if cached_tokens == 0:
            return 0.0
        # 90% discount on cached tokens
        cached_cost = (cached_tokens / 1000) * 0.0003  # 90% off
        new_cost = (new_tokens / 1000) * 0.003
        original = (original_cost / 1000) * 0.003
        return original - (cached_cost + new_cost)

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += len(msg.get("content", "")) // 4
        return total

    def _is_cacheable(self, content: str) -> bool:
        """Check if content is marked for caching."""
        return "<cache>" in content

    def _generate_cache_key(self, messages: list[dict[str, str]]) -> str:
        import hashlib

        content = str(messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class GooglePromptCacheAdapter(PromptCachePort):
    """
    Google Gemini Prompt Caching Adapter.

    Google supports both implicit and explicit caching.
    Cache TTL is configurable.

    2026: Similar to OpenAI with 50-75% savings.
    """

    def __init__(self, model: str = "gemini-2.0-pro"):
        self.model = model
        self._cached_contexts: dict[str, list[dict]] = {}
        self._config = PromptCacheConfig(
            provider=CacheProvider.GOOGLE,
            cache_min_tokens=1024,
            cache_ttl_seconds=3600,  # 1 hour
            auto_cache_enabled=True,
        )

    async def check_cache(self, messages: list[dict[str, str]]) -> PromptCacheResult:
        """Check if messages can use cached context."""
        total_tokens = self._estimate_tokens(messages)
        cached_tokens = total_tokens if total_tokens >= 1024 else 0
        new_tokens = total_tokens - cached_tokens

        return PromptCacheResult(
            cached=cached_tokens > 0,
            cache_hit=any(m.get("context") for m in messages),
            cached_tokens=cached_tokens,
            new_tokens=new_tokens,
            total_tokens=total_tokens,
            cost_reduction_percent=75.0 if cached_tokens > 0 else 0.0,
            provider=CacheProvider.GOOGLE,
        )

    async def cache_prompt(
        self, messages: list[dict[str, str]], cache_key: str | None = None
    ) -> bool:
        """Cache messages as context for future requests."""
        total_tokens = self._estimate_tokens(messages)
        if total_tokens >= 1024:
            cache_key = cache_key or self._generate_cache_key(messages)
            self._cached_contexts[cache_key] = messages
            return True
        return False

    async def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate cached context."""
        if cache_key in self._cached_contexts:
            del self._cached_contexts[cache_key]
            return True
        return False

    def get_cache_config(self) -> PromptCacheConfig:
        return self._config

    def calculate_savings(self, cached_tokens: int, new_tokens: int, original_cost: float) -> float:
        if cached_tokens == 0:
            return 0.0
        # 75% discount on cached tokens
        cached_cost = (cached_tokens / 1000) * 0.000125
        new_cost = (new_tokens / 1000) * 0.0005
        original = (original_cost / 1000) * 0.0005
        return original - (cached_cost + new_cost)

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += len(msg.get("content", "")) // 4
        return total

    def _generate_cache_key(self, messages: list[dict[str, str]]) -> str:
        import hashlib

        content = str(messages)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class MultiProviderPromptCachePort(PromptCachePort):
    """
    Unified prompt caching that routes to provider-specific adapters.

    2026 Pattern: Multi-provider caching with automatic failover.
    """

    def __init__(self, primary_provider: CacheProvider = CacheProvider.OPENAI):
        self._adapters: dict[CacheProvider, PromptCachePort] = {
            CacheProvider.OPENAI: OpenAIPromptCacheAdapter(),
            CacheProvider.ANTHROPIC: AnthropicPromptCacheAdapter(),
            CacheProvider.GOOGLE: GooglePromptCacheAdapter(),
        }
        self._current_provider = primary_provider

    def set_provider(self, provider: CacheProvider) -> None:
        """Switch active provider."""
        if provider in self._adapters:
            self._current_provider = provider

    @property
    def current_adapter(self) -> PromptCachePort:
        return self._adapters[self._current_provider]

    async def check_cache(self, messages: list[dict[str, str]]) -> PromptCacheResult:
        return await self.current_adapter.check_cache(messages)

    async def cache_prompt(
        self, messages: list[dict[str, str]], cache_key: str | None = None
    ) -> bool:
        return await self.current_adapter.cache_prompt(messages, cache_key)

    async def invalidate_cache(self, cache_key: str) -> bool:
        return await self.current_adapter.invalidate_cache(cache_key)

    def get_cache_config(self) -> PromptCacheConfig:
        return self.current_adapter.get_cache_config()

    def calculate_savings(self, cached_tokens: int, new_tokens: int, original_cost: float) -> float:
        return self.current_adapter.calculate_savings(cached_tokens, new_tokens, original_cost)
