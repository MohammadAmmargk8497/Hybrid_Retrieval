# 0003 — Shrink chunk size and tighten splitter

**Stage:** 1 (foundation fixes) **Files touched:** `src/config.py`,
`src/pdf_processing.py`

## What was wrong

```python
CHUNK_SIZE = 20000
CHUNK_OVERLAP = 500
```

A typical arXiv paper is 30k–60k chars. With a 20k chunk size most papers
produce **1 to 3 chunks** total, which breaks both retrieval legs:

- **Vector retrieval.** Embedding a 20k-char span (≈ 4–5k tokens) into a
  single 384/768-d vector averages out everything specific. The chunk for
  "the FlashAttention paper" looks roughly like the chunk for "any
  transformer paper". Top-k similarity collapses to coarse topical
  matching; the system can't tell you *which paragraph* is relevant.
- **BM25.** Term-frequency saturation in `BM25Okapi` is parameterized by
  document length. With 3 documents per paper and tens of thousands of
  tokens each, IDF and length-normalization both degrade — rare terms
  ("RoPE", "ALiBi") stop discriminating.
- **Snippets.** The UI shows `doc[:200]` — the first 200 chars of a 20k
  chunk, which is almost always the paper's header / first paragraph
  regardless of what the query was about.

The separator list `["\n\n", "\n", " ", ""]` also has no sentence-boundary
fallback, so when paragraph and line breaks aren't enough, the splitter
falls straight to mid-word splits.

## Fix

```python
CHUNK_SIZE    = 1200   # ≈ 200–300 tokens, paragraph-scale
CHUNK_OVERLAP = 150    # ~12 % overlap, preserves cross-boundary phrases
separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
```

The new separator priority — paragraph → line → sentence → clause → word →
char — keeps chunks landing on meaningful boundaries even when paragraph
breaks are absent (common after PDF extraction concatenates pages with
single newlines).

`1200` was chosen as a defensible default; we'll likely re-tune to a
**token-aware** splitter once the Nomic Embed v1.5 swap lands (Stage 2).
For now the value is paired with downstream embedders that handle ≥512
tokens, so we have headroom.

## Re-index implication

Existing chunks in Chroma were produced at 20k size and are unaffected. A
full re-index is the only way to pick up the new granularity. Documented
re-index path will land alongside change 0005 (config / CLI).

## Why now

Chunk size and tokenization are the two parameters that compound through
*every* downstream change: extraction, embedding choice, reranker
calibration, KG-extraction granularity. Fixing it before swapping
extractor/embedder means we won't have to re-baseline twice.
