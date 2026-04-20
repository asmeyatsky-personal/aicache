"""Real semantic-search adapters — Chroma + sentence-transformers + OpenAI.

Optional dependencies. Each adapter imports its heavy library lazily
at construction so ``aicache`` stays installable with just the base
dependencies. Missing extras raise an informative ``ImportError`` that
points at the correct ``pip install aicache[semantic]`` command.

Ports implemented:
- ``SemanticIndexPort`` — :class:`ChromaSemanticIndexAdapter`
- ``EmbeddingGeneratorPort`` — :class:`SentenceTransformerEmbeddingAdapter`,
  :class:`OpenAIEmbeddingAdapter`
"""

from __future__ import annotations

import logging
from typing import Any

from ..domain.models import SemanticMatch
from ..domain.ports import EmbeddingGeneratorPort, SemanticIndexPort

logger = logging.getLogger(__name__)


def _missing(extra: str, lib: str, err: Exception) -> ImportError:
    return ImportError(
        f"{lib} is required for this adapter. Install with `pip install aicache[{extra}]` "
        f"(underlying error: {err})"
    )


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


class SentenceTransformerEmbeddingAdapter(EmbeddingGeneratorPort):
    """Local CPU embeddings via sentence-transformers.

    The default ``all-MiniLM-L6-v2`` is ~25 MB and produces 384-dim
    embeddings — fast enough for cache-path use (< 50 ms for a typical
    prompt).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise _missing("semantic", "sentence-transformers", err) from err

        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension() or 384)

    async def generate_embedding(self, text: str) -> list[float]:
        # sentence-transformers is synchronous; the port is async for
        # parity with remote embedders. Good enough at cache-path scale.
        vec = self._model.encode(text, convert_to_numpy=False, show_progress_bar=False)
        return [float(x) for x in vec]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]

    def get_embedding_dimension(self) -> int:
        return self._dimension


_OPENAI_EMBED_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingAdapter(EmbeddingGeneratorPort):
    """Embeddings via the OpenAI API. Requires network + API key."""

    def __init__(self, client: Any | None = None, model: str = "text-embedding-3-small") -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as err:
                raise _missing("openai", "openai", err) from err
            client = OpenAI()

        self._client = client
        self._model = model
        self._dimension = _OPENAI_EMBED_DIMS.get(model, 1536)

    async def generate_embedding(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [list(item.embedding) for item in response.data]

    def get_embedding_dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------


class ChromaSemanticIndexAdapter(SemanticIndexPort):
    """ChromaDB-backed vector index.

    Defaults to an in-memory client so tests and short-lived processes
    don't drop files on disk. Pass ``persist_directory`` for durable
    storage. ``collection_name`` lets one cache host several bounded
    contexts if ever needed.
    """

    def __init__(
        self,
        *,
        collection_name: str = "aicache",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as err:
            raise _missing("semantic", "chromadb", err) from err

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        # Don't auto-embed — the port contract says callers pass
        # pre-computed embeddings. We set an Embedding-less collection
        # to be explicit.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def index_embedding(
        self, key: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        # Chroma treats str-valued metadata as query-able; stringify
        # non-primitive values so it doesn't raise.
        safe_meta = {k: _stringify(v) for k, v in (metadata or {}).items()}
        self._collection.upsert(ids=[key], embeddings=[embedding], metadatas=[safe_meta])

    async def find_similar(
        self, embedding: list[float], threshold: float = 0.85
    ) -> list[SemanticMatch]:
        result = self._collection.query(query_embeddings=[embedding], n_results=5)

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if not ids:
            return []

        matches: list[SemanticMatch] = []
        for key, distance in zip(ids, distances, strict=False):
            # Chroma returns cosine *distance* (0 = identical, 2 = opposite)
            # under the hnsw:cosine space. Convert back to similarity.
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            if similarity >= threshold:
                matches.append(
                    SemanticMatch(
                        similarity_score=similarity,
                        matched_entry_key=key,
                        confidence=similarity,
                    )
                )
        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    async def remove_embedding(self, key: str) -> bool:
        try:
            self._collection.delete(ids=[key])
            return True
        except Exception:
            logger.exception("Chroma delete failed for key=%s", key)
            return False

    async def clear(self) -> None:
        # Chroma doesn't have a single "clear" — drop and recreate.
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def _stringify(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else value
    return str(value)
