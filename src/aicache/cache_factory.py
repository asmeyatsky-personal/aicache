"""
AI Cache Factory - Provides appropriate cache implementation based on needs.

This module provides a unified interface to get cache instances, choosing between:
1. LightweightCoreCache - Simple file-based caching without dependencies
2. DomainCache - Full DDD-compliant caching with ports/adapters

The factory automatically selects the appropriate implementation based on
configuration and optional dependencies.
"""

from .config import get_config
from .core.cache import CoreCache


class CacheFactory:
    """Factory for creating appropriate cache instances."""

    _instance: CoreCache | None = None
    _domain_instance = None

    @classmethod
    def get_cache(cls, use_domain: bool = False, use_enhanced: bool = False):
        """
        Get a cache instance.

        Args:
            use_domain: Use domain-compliant storage (requires async)
            use_enhanced: Use enhanced cache with semantic features

        Returns:
            Cache instance (CoreCache by default for backward compatibility)
        """
        config = get_config()

        # For now, always return CoreCache for backward compatibility
        # The domain/enhanced implementations can be phased in
        if cls._instance is None:
            cache_dir = config.get("cache_dir", "~/.cache/aicache")
            cls._instance = CoreCache(cache_dir=cache_dir)

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset cache instances (useful for testing)."""
        cls._instance = None
        cls._domain_instance = None


def get_cache(use_domain: bool = False) -> CoreCache:
    """
    Convenience function to get a cache instance.

    Args:
        use_domain: If True, would return domain-compliant cache (future)

    Returns:
        CoreCache instance
    """
    return CacheFactory.get_cache(use_domain=use_domain)


def create_cache(
    cache_dir: str | None = None,
    ttl: int | None = None,
    max_size_mb: int | None = None,
    enable_semantic: bool = False,
    enable_encryption: bool = True,
) -> CoreCache:
    """
    Create a configured cache instance.

    Args:
        cache_dir: Cache directory path
        ttl: Default TTL in seconds
        max_size_mb: Maximum cache size in MB
        enable_semantic: Enable semantic caching
        enable_encryption: Enable encryption for sensitive data

    Returns:
        Configured CoreCache instance
    """
    config = get_config()

    if cache_dir is None:
        cache_dir = config.get("cache_dir", "~/.cache/aicache")

    cache = CoreCache(cache_dir=cache_dir)

    # Store additional config in cache for later use
    cache._config = {
        "ttl": ttl or config.get("ttl", 0),
        "max_size_mb": max_size_mb or config.get("max_size_mb", 1000),
        "enable_semantic": enable_semantic,
        "enable_encryption": enable_encryption and config.get("security.encrypt_sensitive", True),
    }

    return cache


# Backward compatibility - re-export CoreCache
__all__ = [
    "CacheFactory",
    "CoreCache",
    "create_cache",
    "get_cache",
]
