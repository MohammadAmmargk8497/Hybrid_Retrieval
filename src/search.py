"""Hybrid retrieval: dense (Qdrant + Nomic) + sparse (bm25s), fused with RRF.

RRF reference: Cormack, Clarke & Buettcher,
*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods*, SIGIR 2009.

    score(d) = Σ_r  1 / (k + rank_r(d))

We fuse on stable string **chunk ids** carried in Qdrant payloads. The
dense leg already returns full payloads alongside scores, so for the
common case (top hit came from dense) we don't need a hydration round-trip
back to Qdrant. Only chunk ids that surfaced *only* in the sparse leg
need a follow-up ``client.retrieve`` call.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, TypeVar

from src.models import SearchHit
from src.qdrant_store import COLLECTION, chunk_id_to_point_id

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient

    from src.embedding import Embedder

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 300

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[T]], k: int = 60
) -> list[tuple[T, float]]:
    """Fuse multiple ranked lists of doc-ids with RRF.

    Each input is a sequence of ids ordered best-first. Returns
    ``[(id, fused_score), ...]`` sorted by fused_score desc. Generic over
    id type (int corpus indices or str chunk ids both work).
    """
    fused: dict[T, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):  # 0-indexed
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _payload_to_hit(payload: dict, score: float) -> SearchHit:
    text = payload.get("text", "")
    snippet = (text[:_SNIPPET_CHARS] + "...") if len(text) > _SNIPPET_CHARS else text
    return SearchHit(
        chunk_id=payload.get("chunk_id", ""),
        doc_id=payload.get("doc_id", payload.get("source", "")),
        source=payload.get("source", "Unknown"),
        page=int(payload.get("page_start", 1)),
        score=float(score),
        snippet=snippet,
        text=text,
        section=payload.get("section"),
    )


def hybrid_search(
    qdrant_client: QdrantClient,
    embedder: Embedder,
    bm25_retriever,
    bm25_chunk_ids: Sequence[str],
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    collection: str = COLLECTION,
) -> list[SearchHit]:
    """Run dense + sparse retrieval and fuse with RRF.

    Parameters
    ----------
    qdrant_client
        Connected Qdrant client. Server-mode in production, ``:memory:``
        in tests.
    embedder
        Anything implementing the ``Embedder`` protocol; production uses
        ``NomicEmbedder``.
    bm25_retriever, bm25_chunk_ids
        ``bm25s.BM25`` plus the aligned chunk-id list (sidecar JSON).
    query, top_k, candidate_k, rrf_k
        Standard retrieval knobs. ``candidate_k`` defaults to 4×top_k.
    """
    import bm25s  # local import: heavy dep, not needed for RRF unit tests

    candidate_k = candidate_k or max(top_k * 4, 20)
    candidate_k = min(candidate_k, max(len(bm25_chunk_ids), top_k))
    logger.info("Hybrid search (top_k=%d, candidate_k=%d)", top_k, candidate_k)

    # --- Dense leg: encode query and ask Qdrant.
    qvec = embedder.embed_query(query).tolist()
    dense_resp = qdrant_client.query_points(
        collection_name=collection,
        query=qvec,
        limit=candidate_k,
        with_payload=True,
    )
    dense_points = dense_resp.points if hasattr(dense_resp, "points") else dense_resp

    payloads_by_id: dict[str, dict] = {}
    dense_ranking: list[str] = []
    for p in dense_points:
        payload = p.payload or {}
        cid = payload.get("chunk_id")
        if cid is None:
            continue
        dense_ranking.append(cid)
        payloads_by_id[cid] = payload

    # --- Sparse leg.
    query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    bm25_idx, _scores = bm25_retriever.retrieve(
        query_tokens, k=candidate_k, show_progress=False
    )
    sparse_ranking: list[str] = [bm25_chunk_ids[int(i)] for i in bm25_idx[0]]

    # --- Fuse.
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=rrf_k)
    top = fused[:top_k]
    if not top:
        return []

    # --- Hydrate any chunk_ids that only came from the sparse leg.
    missing = [cid for cid, _ in top if cid not in payloads_by_id]
    if missing:
        retrieved = qdrant_client.retrieve(
            collection_name=collection,
            ids=[chunk_id_to_point_id(cid) for cid in missing],
            with_payload=True,
        )
        for r in retrieved:
            cid = (r.payload or {}).get("chunk_id")
            if cid:
                payloads_by_id[cid] = r.payload

    hits = [_payload_to_hit(payloads_by_id.get(cid, {"chunk_id": cid}), score) for cid, score in top]
    logger.info("Hybrid search returned %d hits.", len(hits))
    return hits
