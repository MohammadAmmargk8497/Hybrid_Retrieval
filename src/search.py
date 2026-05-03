"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank Fusion.

RRF reference: Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods* (SIGIR 2009).

    score(d) = Σ_r  1 / (k + rank_r(d))

where `rank_r(d)` is d's 1-indexed position in ranker r's result list, and k is
a small constant (60 in the original paper) that dampens the influence of very
top-ranked items so multiple rankers can vote.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def search_in_chroma(collection, query: str, top_k: int = 5):
    """Run a Chroma vector search; returns Chroma's raw response dict."""
    logger.info("Chroma search: %r (top_k=%d)", query, top_k)
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def bm25_search(bm25_model, corpus, query: str, top_k: int = 5):
    """Run BM25; returns a list of {corpus_id, score} sorted by score desc."""
    logger.info("BM25 search: %r (top_k=%d)", query, top_k)
    tokenized_query = query.split(" ")
    doc_scores = bm25_model.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:top_k]
    return [{"corpus_id": int(i), "score": float(doc_scores[i])} for i in top_n]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of doc-ids with RRF.

    Each input is a sequence of corpus_ids ordered best-first. Returns
    [(corpus_id, fused_score), ...] sorted by fused_score desc.
    """
    fused: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):  # rank is 0-indexed
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(
    collection,
    bm25_model,
    corpus: Sequence[str],
    metadatas: Sequence[dict],
    query: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    rrf_k: int = 60,
):
    """Run dense + sparse retrieval and fuse with RRF.

    `candidate_k` is the per-ranker depth fed into RRF (defaults to 4×top_k).
    Pulling deeper candidates is cheap and meaningfully improves fusion quality.
    """
    candidate_k = candidate_k or max(top_k * 4, 20)
    logger.info("Hybrid search (top_k=%d, candidate_k=%d)", top_k, candidate_k)

    # --- Dense: map Chroma's returned documents back to corpus indices.
    chroma_results = search_in_chroma(collection, query, top_k=candidate_k)
    chroma_docs = chroma_results["documents"][0]

    # Build doc->id map once. (Cheap relative to a query, and avoids the
    # previous O(N) corpus.index() call per Chroma hit.)
    doc_to_id: dict[str, int] = {}
    for i, doc in enumerate(corpus):
        doc_to_id.setdefault(doc, i)

    chroma_ranking = [doc_to_id[d] for d in chroma_docs if d in doc_to_id]

    # --- Sparse.
    bm25_results = bm25_search(bm25_model, corpus, query, top_k=candidate_k)
    bm25_ranking = [r["corpus_id"] for r in bm25_results]

    # --- Fuse on rank position (the bug-fix: previously BM25's raw score was
    # plugged into 1/(k+score), severely under-weighting BM25 evidence).
    fused = reciprocal_rank_fusion([chroma_ranking, bm25_ranking], k=rrf_k)

    results = []
    for doc_id, score in fused[:top_k]:
        meta = metadatas[doc_id] if doc_id < len(metadatas) else {"source": "Unknown"}
        results.append(
            {
                "document": corpus[doc_id],
                "metadata": {"source": meta.get("source", "Unknown")},
                "score": score,
            }
        )

    logger.info("Hybrid search returned %d results.", len(results))
    return results
