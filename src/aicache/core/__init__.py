# Core cache module

from .cache import CacheEntry, CoreCache, get_cache

# Backward compatibility - also expose as Cache
Cache = CoreCache

__all__ = ["Cache", "CacheEntry", "CoreCache", "get_cache"]
