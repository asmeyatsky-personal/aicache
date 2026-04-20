"""Infrastructure layer: adapters + composition root."""

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
from .container import Container, build_container, get_container, reset_container

__all__ = [
    "Container",
    "FileSystemStorageAdapter",
    "InMemoryCacheMetricsAdapter",
    "InMemoryEventPublisherAdapter",
    "InMemoryStorageAdapter",
    "OpenAITokenCounterAdapter",
    "SimpleEmbeddingGeneratorAdapter",
    "SimpleQueryNormalizerAdapter",
    "SimpleSemanticIndexAdapter",
    "build_container",
    "get_container",
    "reset_container",
]
