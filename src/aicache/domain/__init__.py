"""Domain layer: Core business logic for AI caching."""

from .models import (
    CacheEntry,
    CacheInvalidationEvent,
    CacheMetadata,
    CacheMetrics,
    CachePolicy,
    CacheResult,
    EvictionPolicy,
    InvalidationStrategy,
    SemanticMatch,
    TokenUsageMetrics,
)
from .ports import (
    CacheMetricsPort,
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    RepositoryPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
    TOONRepositoryPort,
)
from .prompt_caching import (
    AnthropicPromptCacheAdapter,
    CacheProvider,
    GooglePromptCacheAdapter,
    MultiProviderPromptCachePort,
    OpenAIPromptCacheAdapter,
    PromptCacheConfig,
    PromptCachePort,
    PromptCacheResult,
)
from .services import (
    CacheEvictionService,
    CacheInvalidationService,
    CacheTTLService,
    QueryNormalizationService,
    SemanticCachingService,
    TokenCountingService,
)

__all__ = [
    "AnthropicPromptCacheAdapter",
    # Models
    "CacheEntry",
    "CacheEvictionService",
    "CacheInvalidationEvent",
    "CacheInvalidationService",
    "CacheMetadata",
    "CacheMetrics",
    "CacheMetricsPort",
    "CachePolicy",
    "CacheProvider",
    "CacheResult",
    "CacheTTLService",
    "EmbeddingGeneratorPort",
    "EventPublisherPort",
    "EvictionPolicy",
    "GooglePromptCacheAdapter",
    "InvalidationStrategy",
    "MultiProviderPromptCachePort",
    "OpenAIPromptCacheAdapter",
    "PromptCacheConfig",
    # Prompt Caching (2026)
    "PromptCachePort",
    "PromptCacheResult",
    # Services
    "QueryNormalizationService",
    "QueryNormalizerPort",
    "RepositoryPort",
    "SemanticCachingService",
    "SemanticIndexPort",
    "SemanticMatch",
    # Ports
    "StoragePort",
    "TOONRepositoryPort",
    "TokenCounterPort",
    "TokenCountingService",
    "TokenUsageMetrics",
]
