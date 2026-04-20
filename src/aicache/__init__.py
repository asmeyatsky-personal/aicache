"""aicache — drop-in caching for Claude and OpenAI SDK calls."""

from .cache_factory import CacheFactory, create_cache
from .config import get_config, get_config_manager
from .core.cache import CoreCache, get_cache
from .domain.models import (
    CacheEntry,
    CacheMetrics,
    CachePolicy,
    CacheResult,
    EvictionPolicy,
    SemanticMatch,
    TokenUsageMetrics,
)
from .domain.ports import (
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
from .modern_cli import main
from .security import (
    SecurityUtils,
    detect_pii,
    is_safe_prompt,
    mask_pii,
    sanitize_input,
    validate_context,
)

__version__ = "0.3.0-dev"

__all__ = [
    "CacheEntry",
    "CacheFactory",
    "CacheMetrics",
    "CacheMetricsPort",
    "CachePolicy",
    "CacheResult",
    "CoreCache",
    "EmbeddingGeneratorPort",
    "EventPublisherPort",
    "EvictionPolicy",
    "QueryNormalizerPort",
    "RepositoryPort",
    "SecurityUtils",
    "SemanticIndexPort",
    "SemanticMatch",
    "StoragePort",
    "TOONRepositoryPort",
    "TokenCounterPort",
    "TokenUsageMetrics",
    "__version__",
    "create_cache",
    "detect_pii",
    "get_cache",
    "get_config",
    "get_config_manager",
    "is_safe_prompt",
    "main",
    "mask_pii",
    "sanitize_input",
    "validate_context",
]
