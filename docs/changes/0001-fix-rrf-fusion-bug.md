# 0001 — Fix RRF fusion bug

**Stage:** 1 (foundation fixes) **Files touched:** `src/search.py`

## What was wrong

`hybrid_search` collected per-ranker evidence into a single flat list of
`(doc_id, x)` tuples and passed it to `reciprocal_rank_fusion`, which applied
`1 / (k + x)` to whatever `x` was. The two rankers used incompatible meanings
for `x`:

```python
# Chroma branch — x = positional rank index (0, 1, 2, ...)
for i, doc in enumerate(chroma_docs):
    rrf_results.append((corpus_id, i))            # rank index ✓

# BM25 branch — x = raw BM25 score (often 5–50+)
for result in bm25_results:
    rrf_results.append((result['corpus_id'], result['score']))   # raw score ✗
```

Then:

```python
fused_scores[doc_id] += 1 / (k + score)   # k = 60
```

A Chroma hit at rank 0 contributed `1/(60+0) ≈ 0.0167`. A BM25 hit with a
typical score of 30 contributed `1/(60+30) ≈ 0.0111` — and a strong BM25 hit
with score 100 contributed `1/160 ≈ 0.0062`, *less* than a weak Chroma hit at
rank 19 (`1/79 ≈ 0.0127`). BM25 was systematically suppressed; the higher its
real score, the *worse* its fused contribution.

A second issue: `list(corpus).index(doc)` was an O(N) scan for **every**
Chroma hit on **every** query, where `corpus` was already a list pulled from
the entire collection.

## Fix

Rewrote `reciprocal_rank_fusion` to take **a list of ranked lists** of
corpus-ids — the canonical RRF input — and apply `1 / (k + rank + 1)` using
each item's *position* in its own list. Both rankers now contribute on the
same scale.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    fused = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):  # 0-indexed
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
```

`hybrid_search` now:

1. Pulls `candidate_k = max(4 × top_k, 20)` from each ranker (deeper pools →
   better fusion at negligible cost).
2. Builds a `doc -> corpus_id` dict **once** to map Chroma's returned text
   back to corpus indices in O(1).
3. Passes two clean rank lists (`chroma_ranking`, `bm25_ranking`) to RRF.
4. Returns top-`top_k` after fusion.

`rrf_k` is exposed as a parameter (default 60, per Cormack et al. 2009).

## Why it matters

Hybrid retrieval is the whole product. The fusion math being wrong meant the
"hybrid" results were closer to "vector-only with mild noise from BM25". This
fix restores the intended behavior — keyword evidence (titles, author names,
exact phrases like "FlashAttention") will now actually surface.

## Follow-ups not done here

- **Test the fix.** Will add `tests/test_search.py` once `pytest` is wired up
  in change 0007.
- **Sparse model.** `rank_bm25` + pickled-blob is replaced wholesale in
  change 0006 (`bm25s`). Kept the same surface here so this change is
  isolated.
- **Candidate depth.** `4× top_k` is a defensible default; revisit once we
  add the reranker in Stage 2.
