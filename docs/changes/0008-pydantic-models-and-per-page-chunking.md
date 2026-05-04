# 0008 — Pydantic data models + per-page chunking

**Stage:** 2 (extraction + embeddings overhaul) **Files touched:**
`src/models.py` (new), `src/pdf_processing.py`, `src/vector_store.py`,
`src/search.py`, `src/main.py`, `src/cli.py`,
`tests/test_bm25_integration.py`, `tests/test_models.py` (new).

## Why

Stage 2 introduces several layers — GROBID extraction, Nomic embeddings,
Qdrant storage, possibly a reranker, then a FastAPI service in Stage 3.
Each layer needs to agree on the shape of a "chunk" and a "search hit".
Before 0008 those were ad-hoc tuples (`(chunk_id, text, {"source": ...})`)
and dicts (`{"document": ..., "metadata": {"source": ...}, "score": ...}`),
typo-prone and unverified at module boundaries.

Pydantic v2 (already a transitive dep via `chromadb`) gives us:

- A single source of truth for field names / types.
- Validation at the boundary — a malformed metadata dict fails loudly at
  the producer instead of silently propagating to the consumer.
- Free JSON serialization with the exact same shape across CLI `--json`
  output, the upcoming FastAPI endpoints, and any IPC with the Darwin
  Research Agent.

We also use this change to fix a foundational gap: **chunks did not know
which page they came from**. The old extraction joined every page into
one string before chunking, so we could never implement
"click-to-open-at-page-N" — the marquee UX you asked for.

## Changes

### `src/models.py`

Three core types, all `frozen=True, extra="forbid"`:

```python
class ChunkMetadata(BaseModel):
    source: str            # "2105.12723v4.pdf"
    doc_id: str            # filename stem (→ arXiv id once Darwin lands)
    page_start: int        # 1-indexed
    page_end: int          # 1-indexed
    section: str | None    # filled by GROBID in 0009
    chunk_index: int

class Chunk(BaseModel):
    id: str
    text: str              # min_length=1
    metadata: ChunkMetadata

class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int              # what page to open the PDF at
    score: float
    snippet: str
    text: str = Field(default="", repr=False)
    section: str | None = None

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]
    took_ms: float
```

`extra="forbid"` is deliberate — typos like `sectionn=...` blow up at
construction time instead of becoming silent extra fields. `frozen=True`
makes accidental mutation impossible (chunks flow through several
modules; immutability is a real defense).

`ChunkMetadata` is intentionally **flat** so `model_dump(exclude_none=True)`
serialises straight into either Chroma or Qdrant payload schemas — no
custom serialiser needed.

`page_start`/`page_end` are 1-indexed because that's what users see in
PDF viewers and what `pypdf` exposes as the "human-facing" page number.
We bumped from 0-indexed via the conversion in `pdf_processing.py`.

### `src/pdf_processing.py` — per-page chunking

`extract_text_from_pdfs` now:

1. Loads each PDF and **iterates pages**, instead of joining them.
2. Splits each page independently, so every chunk is born knowing its
   page number.
3. Returns `list[Chunk]` instead of `list[tuple[str, str, dict]]`.
4. New chunk-id format: `{doc_id}_p{page}_c{chunk_index}` (e.g.
   `2105.12723v4_p4_c2`). Stable across reindexes as long as the
   filename, page, and intra-page chunk index don't change.

Trade-off: chunks **never straddle pages**. We lose the ability to bridge
sentences that wrap end-of-page. At `chunk_size=1200` and typical paper
pages of 3000–5000 chars, this loss is a handful of tokens per page —
fully acceptable in exchange for free click-to-page navigation. GROBID
in 0009 will refine this further with section boundaries (so e.g. the
"Method" section becomes one logical unit).

### `src/vector_store.py`

`store_new_pdfs_in_chroma` accepts `list[Chunk]` and emits
`metadata=[c.metadata.model_dump(exclude_none=True) ...]`. Function
signature change is the only API break for downstream callers.

### `src/search.py`

`hybrid_search` now returns `list[SearchHit]`. The Chroma metadata fetch
populates `page`, `section`, `doc_id`, `source` fields — clients no
longer dig through nested dicts.

A new `run_search_headless` wraps `hybrid_search` with timing and
returns `SearchResponse`, capturing `took_ms` so the GUI/CLI can show
latency and so future SLO instrumentation has a hook.

### `src/main.py` + `src/cli.py`

GUI display updated to show `source · p.N · [section]  (score=...)` per
hit. CLI `--json` mode now uses `response.model_dump_json(indent=2)` —
that's the exact same wire format the Stage-3 FastAPI handler will
return.

### Tests

- `tests/test_models.py` (6 tests) — JSON round-trip, page validators
  (zero rejected), `extra="forbid"` (typos rejected), empty-text reject,
  `model_dump(exclude_none=True)` drops null section,
  `SearchResponse` end-to-end roundtrip.
- `tests/test_bm25_integration.py` updated — fixture now constructs
  `Chunk` objects with deliberate page numbers (paperA=2, B=3, C=4,
  D=5) and the FlashAttention test asserts `hits[0].page == 3`. This
  is the regression guard for the per-page chunking flow.

24 tests total, all passing. Ruff clean.

## Knock-on effects

- Existing Chroma collections store metadata only with `{"source": ...}`
  — they lack `doc_id`, `page_start`, `page_end`, `chunk_index`. They
  will still work (the `meta.get("page_start", 1)` fallback in
  `search.py` defaults to page 1), but the page-aware UX needs a
  `python -m src reindex` to land. Documented in the existing reindex
  story; no new migration step required.
- Backwards compatibility for ad-hoc consumers of the old return types
  is intentionally **not** preserved. The product is pre-1.0 and the
  old shape was undocumented.

## Follow-ups not done here

- **Document-level model.** A `Document` aggregate (arxiv_id, title,
  authors, sections, references) is overkill until GROBID gives us
  structured input — landing in 0009.
- **Section field.** Always `None` until GROBID. The tests already
  cover both populated and empty cases via the optional field.
- **Reranker scores / per-leg ranks.** Could add `dense_rank` and
  `sparse_rank` to `SearchHit` for explainability. Not yet — wait for
  the reranker (0011) so we can design the explainability fields once.
