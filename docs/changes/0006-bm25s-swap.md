# 0006 — Swap `rank_bm25` for `bm25s`

**Stage:** 1 (foundation fixes)
**Files touched:** `src/search.py`, `src/main.py`, `src/config.py`,
`pyproject.toml`

## What was wrong

The previous BM25 path had three real problems on top of the speed gap:

1. **`rank_bm25.BM25Okapi` is pure Python.** Indexing and scoring scale
   linearly in Python with no SIMD; for a few thousand chunks this is
   tolerable, but with the new ~250-token chunks (change 0003) the chunk
   count multiplies ~15× and `rank_bm25` becomes the long pole in every
   query.
2. **Persistence was a `pickle.dump` of the entire model.** Every search
   re-loaded a multi-MB pickle (the existing `bm25_model.pkl` in the repo
   is 7.6 MB) and Python pickle is fragile across version bumps.
3. **Tokenization was `query.split(" ")`.** No lowercasing, no stopwords,
   no punctuation handling. So "Attention" and "attention" were different
   tokens; "BERT," (with the comma) didn't match "BERT".

A subsidiary correctness issue lived in `hybrid_search`: it mapped Chroma
hits back to corpus indices by **document text equality**. If two chunks
happened to have identical text (boilerplate headers, repeated abstracts
across versions of the same paper), they collapsed; if extraction was
non-deterministic between calls, the mapping silently broke.

## Fix

### `bm25s` for sparse retrieval

`bm25s` ([github.com/xhluca/bm25s](https://github.com/xhluca/bm25s)) keeps
the BM25 algorithm but stores everything as a sparse scipy matrix and
runs the hot loop in C/numpy. Reported numbers: 100×–500× faster than
`rank_bm25` on retrieve. It also ships:

- A built-in tokenizer (`bm25s.tokenize`) that does lowercase + regex
  word-splits + stopword removal — much closer to a sane default. We use
  `stopwords="en"` and **no stemming** (deliberate: AI papers contain
  precise terminology like "FlashAttention", "RoPE", "ALiBi" that we do
  not want collapsed by Snowball stemming).
- A **directory-based** save format (separate `.npy` for sparse matrix
  components, `.json` for vocab and params). No pickle, no version-fragile
  blob.

### Fuse on chunk IDs, not document text

The bug-fix opportunity tackled together with the swap: `hybrid_search`
now operates on Chroma chunk-ids throughout. The dense leg returns
`chroma_res["ids"][0]`. The sparse leg returns positional indices from
`bm25s.retrieve`, which we map to chunk-ids via a sidecar
`chunk_ids.json` saved next to the bm25s index at build time. RRF fuses
the two id-lists. The final hydration is one `collection.get(ids=top_ids)`
call — O(1) per query in lookups.

This eliminates two failure modes:

- Identical chunk text no longer collapses.
- Reordering of `collection.get()` between index-build and search no
  longer breaks the mapping.

`reciprocal_rank_fusion` is now generic (`TypeVar("T", bound=Hashable)`),
so int corpus-indices and str chunk-ids both work — kept the door open
for future rerankers that may want to fuse arbitrary id types.

### Storage layout change

| Old | New |
|---|---|
| `<persist>/bm25_model.pkl` | `<persist>/bm25_index/` directory |
| `<persist>/bm25_corpus.json` | `<persist>/bm25_index/chunk_ids.json` |

`config.py` exposes `settings.bm25_index_dir` and
`settings.bm25_chunk_ids_path` (replacing the now-removed
`bm25_model_path` and `bm25_corpus_path`).

The old artifacts left behind from previous runs are harmless (nothing
loads them anymore) but `python -m src reindex` will sweep them as part
of the wipe.

### Dependency change

```toml
- "rank-bm25 (>=0.2.2,<0.3.0)"
+ "bm25s (>=0.2.13,<1.0.0)"
```

`bm25s` has a transitive optional dep on `jax` for its top-k selection;
the package degrades to numpy if `jax` is unavailable, so no extra dep
is forced on us.

## Verification

End-to-end with a 4-doc in-memory corpus (paper A: Attention Is All You
Need; paper B: FlashAttention; C: RLHF; D: LLM survey):

```text
$ query="flashattention GPU memory IO"
[1] paperB.pdf  score=0.0328
[2] paperA.pdf  score=0.0323
[3] paperD.pdf  score=0.0315
```

paperB (the FlashAttention paper) ranks #1 — the sparse leg is now
contributing real signal. The `tests/test_bm25_integration.py` suite
landed in change 0007 covers this end-to-end and asserts on the ranking.

## Re-index required

Anyone with an existing `bm25_model.pkl` in their persist directory must
run `python -m src reindex` once. The new code does not read the old
pickle; running `process` on a directory with no new PDFs would produce
no rebuild trigger, leaving the system in an inconsistent state. The
reindex command (added in 0005) handles this cleanly.

## Follow-ups not done here

- **Incremental BM25 updates.** `bm25s` does not support
  add-document-at-a-time. Today we rebuild the entire index whenever new
  PDFs are processed. Build is fast enough that this is fine for
  thousands of chunks; will revisit if it becomes a problem.
- **Stemming.** Defaults to off. Worth an A/B test once the reranker
  lands — for some queries (plurals, verb forms) Snowball helps recall
  noticeably.
- **Stopwords list.** `"en"` is bm25s's bundled list. ML-specific
  stopwords ("propose", "method", "approach") could be added later; this
  is the kind of tuning that wants real query logs first.
