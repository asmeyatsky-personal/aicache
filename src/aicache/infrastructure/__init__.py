"""Infrastructure layer: Adapter implementations for external dependencies."""

from .adapters import (
    FileSystemStorageAdapter,
    InMemoryCacheMetricsAdapter,
    InMemoryEventPublisherAdapter,
    InMemoryStorageAdapter,
    OpenAITokenCounterAdapter,
    SimpleEmbeddingGeneratorAdapter,
    SimpleQueryNormalizerAdapter,
    SimpleSemanticIndexAdapter,
)

__all__ = [
    "FileSystemStorageAdapter",
    "InMemoryCacheMetricsAdapter",
    "InMemoryEventPublisherAdapter",
    "InMemoryStorageAdapter",
    "OpenAITokenCounterAdapter",
    "SimpleEmbeddingGeneratorAdapter",
    "SimpleQueryNormalizerAdapter",
    "SimpleSemanticIndexAdapter",
]
