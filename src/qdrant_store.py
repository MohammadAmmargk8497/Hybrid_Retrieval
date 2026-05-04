"""Qdrant client + helpers.

Production deployment is server-mode (Docker compose service); tests use
the embedded ``QdrantClient(":memory:")`` mode that ships with
``qdrant-client``. Same API — only the constructor differs.

ID strategy
-----------
Qdrant point ids must be int or UUID. Our chunk ids are stable strings
like ``arxiv-paper_p3_c5`` (set in 0008 / 0009). We derive a *deterministic*
UUIDv5 from the chunk id using a fixed namespace, so:
  * the same chunk id always maps to the same point id (idempotent
    upserts work),
  * we never need a side table to look up "what's the UUID for chunk X?",
  * ``payload["chunk_id"]`` carries the original string id for retrieval
    consumers (BM25 fusion, the search response).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.models import Chunk

if TYPE_CHECKING:  # pragma: no cover
    from src.embedding import Embedder

logger = logging.getLogger(__name__)

COLLECTION = "pdf_chunks"
# Stable random UUID4 baked in once. Don't change — that would invalidate
# every point-id mapping in existing Qdrant deployments.
_NS = uuid.UUID("c0a4f9d2-7e0b-4d4e-bf43-3c6d2c9a5ef1")


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def get_qdrant_client(url: str | None = None, timeout: float = 30.0) -> QdrantClient:
    """Connect to Qdrant.

    Pass ``url=":memory:"`` for the in-process embedded mode (used in
    tests). Otherwise pass an HTTP url like ``http://localhost:6333``.
    """
    if url is None or url in (":memory:",):
        return QdrantClient(":memory:")
    return QdrantClient(url=url, timeout=timeout)


def ensure_collection(client: QdrantClient, dim: int, name: str = COLLECTION) -> None:
    """Create the collection if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection %r (dim=%d, distance=cosine)", name, dim)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    collection: str = COLLECTION,
    batch_size: int = 64,
) -> None:
    """Embed and upsert a list of ``Chunk``s.

    Idempotent: re-running with the same chunks overwrites existing
    points (same UUID5). Useful when re-extraction produces a slightly
    different chunking — old chunks remain only if their ids are still
    produced; otherwise they should be cleared via the reindex command.
    """
    if not chunks:
        return
    for start in tqdm(range(0, len(chunks), batch_size), desc="Embed+upsert"):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(
                id=chunk_id_to_point_id(c.id),
                vector=vec.tolist(),
                payload={
                    "chunk_id": c.id,
                    "text": c.text,
                    **c.metadata.model_dump(exclude_none=True),
                },
            )
            for c, vec in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d chunks into %s.", len(chunks), collection)


def scroll_all_chunks(
    client: QdrantClient,
    *,
    collection: str = COLLECTION,
    page_size: int = 512,
) -> Iterable[dict]:
    """Yield every payload in the collection (used to rebuild BM25)."""
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            yield r.payload
        if next_offset is None:
            break
