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
from .telemetry import (
    InMemoryTelemetryAdapter,
    JSONLTelemetryAdapter,
    configure_structured_logging,
    summarise,
    traced,
)

__all__ = [
    "Container",
    "FileSystemStorageAdapter",
    "InMemoryCacheMetricsAdapter",
    "InMemoryEventPublisherAdapter",
    "InMemoryStorageAdapter",
    "InMemoryTelemetryAdapter",
    "JSONLTelemetryAdapter",
    "OpenAITokenCounterAdapter",
    "SimpleEmbeddingGeneratorAdapter",
    "SimpleQueryNormalizerAdapter",
    "SimpleSemanticIndexAdapter",
    "build_container",
    "configure_structured_logging",
    "get_container",
    "reset_container",
    "summarise",
    "traced",
]
