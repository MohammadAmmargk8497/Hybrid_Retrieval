# 0009 — GROBID extractor

**Stage:** 2 (extraction + embeddings overhaul)
**Files touched:** `src/grobid.py` (new), `src/pdf_processing.py`,
`src/config.py`, `pyproject.toml`, `docker-compose.yml` (new),
`.env.example`, `README.md`, `tests/test_grobid.py` (new).

## Why

PyPDF gave us text and (after change 0008) per-page coordinates, but
nothing else. For arXiv-style ML papers we leave a lot on the table:

- **Two-column layouts** confuse PyPDF — column 1 of page 3 ends up
  interleaved with column 2 of page 3. BM25 still works, but the dense
  embedder is now embedding sentence salad.
- **Header / footer / page-number boilerplate** ("Preprint. Under
  review.", "arXiv:2105.12723", page numbers) becomes high-IDF garbage
  in the BM25 vocabulary.
- **No section information** — we can't tell the user "this hit is in
  Section 4 (Method)", and we can't condition retrieval on section
  later.
- **No structured author / affiliation / reference data** for the
  knowledge-graph layer.

GROBID is a service purpose-built for scholarly PDFs. It produces TEI
XML with sectionised body text, page-level coordinates on every
paragraph, and structured header / reference metadata. The cost is a
running service — we're paying that cost.

## Decisions made

Per project direction this round:

| Question | Choice | Rationale |
|---|---|---|
| Image | `lfoppiano/grobid:0.8.1` (lightweight, ~2 GB) | The `-full` variant adds DL header parsing at ~5 GB more. We can flip the tag in `docker-compose.yml` later if the KG layer needs better author parsing. |
| Behaviour when GROBID is down | **Hard fail** with `GrobidUnavailable` | A half-degraded index masquerading as a real one is worse than a clear error. The user fixes the pipeline; we never silently regress quality. |
| Section information | **Soft hints in metadata** | Sections do *not* dictate chunk boundaries — chunks remain page-scoped (per 0008). Each chunk carries the first section title seen on its page in `metadata.section`. Retrieval doesn't filter on it; the UI surfaces it. |

## What landed

### `src/grobid.py`

Two pieces:

1. `GrobidClient` — thin `requests` wrapper. Two methods: `is_alive()`
   (probes `/api/isalive`) and `process_fulltext(pdf_path)` (POSTs to
   `/api/processFulltextDocument` and returns TEI XML). We pass
   `teiCoordinates=p,head,s` so paragraphs / headings / sentences carry
   page coords; `consolidateHeader=0` and `consolidateCitations=0`
   disable CrossRef enrichment (slow, not needed yet — flip on for the
   KG layer).
2. `parse_tei_to_chunks(tei_xml, *, source, doc_id, text_splitter,
   clean_text)` — pure function from TEI to `list[Chunk]`. Strategy:
   - Pull the abstract from `<teiHeader>/<profileDesc>/<abstract>`,
     attach it to page 1 with `section="Abstract"`.
   - Walk `<text>/<body>/<div>`. For each `<p>`, derive its page from
     the `coords` attribute (falls back to first child `<s>`'s coords,
     then to page 1).
   - Group paragraph text by page, keeping the first non-null section
     title seen on each page.
   - Run the recursive splitter over the per-page text. Emit `Chunk`
     objects with chunk-id `{doc_id}_p{page}_c{chunk_index}` (same
     format as 0008 — stable across reindexing).

The "soft hint" semantic falls out of step (3): if a section spans
pages, only the first page where its `<p>` lands carries the title;
later pages keep whatever earlier section was active. Sections **do
not** split chunks. This keeps page-level chunking intact while still
surfacing useful provenance.

### `src/pdf_processing.py`

- Removed PyPDF dependency from the extraction path.
- `extract_text_from_pdfs` now constructs a `GrobidClient` (or accepts
  one for tests), calls `is_alive()` up front, and raises
  `GrobidUnavailable` if the service isn't there.
- For each PDF: POST → TEI → parse → chunks.
- A `GrobidUnavailable` raised mid-batch is **propagated**, not swallowed
  — if the service dies in the middle of a 200-PDF run we want a clear
  failure, not a half-indexed corpus where some PDFs were silently
  skipped.

### Configuration

Two new env vars (`.env.example` updated):

- `HR_GROBID_URL` (default `http://localhost:8070`)
- `HR_GROBID_TIMEOUT_S` (default `60`)

`Settings` gained `grobid_url: str` and `grobid_timeout_s: float`.

### `docker-compose.yml`

Single-service compose file pinning `lfoppiano/grobid:0.8.1` on port
8070, with a `curl /api/isalive` healthcheck so `docker compose ps`
reflects readiness. `init: true` means GROBID's child Java process gets
reaped properly on container stop.

```bash
docker compose up -d        # start
docker compose logs -f      # watch boot (~20 s for the JVM)
docker compose down         # stop
```

### Dependencies

```toml
- "langchain (==0.1.16)"
- "langchain-community (==0.0.38)"
- "pypdf (>=6.3.0,<7.0.0)"
+ "langchain-text-splitters (>=0.0.1,<1.0.0)"
+ "pydantic (>=2.0,<3.0)"
+ "requests (>=2.31,<3.0)"
```

The langchain umbrella + langchain-community + pypdf are no longer
imported anywhere. `langchain-text-splitters` is the only langchain
piece we still use (the recursive splitter). `pydantic` is now a
direct dep — already a transitive of `chromadb`, but the `0008` models
make it part of our public surface, so it's worth declaring. `requests`
is for the GROBID HTTP client.

## Tests

11 new tests in `tests/test_grobid.py`. The TEI parser is exercised
with a small synthetic XML fixture (abstract + 3 sections across pages
1, 3, 4, plus a section without coords to verify the page-1 fallback):

- `_first_page_from_coords` edge cases (None, empty, malformed).
- Abstract → page 1 with `section="Abstract"`.
- Section title propagates as a soft hint across pages 3–4.
- Pages reflect the coordinate data, not document order.
- Chunk-ids are unique and follow `{doc_id}_p{page}_c{idx}`.
- Invalid XML returns `[]` (logged, doesn't raise).
- `GrobidClient.is_alive` returns False on non-200 / network error.
- `process_fulltext` raises `GrobidUnavailable` on HTTP error.
- `extract_text_from_pdfs` hard-fails when the client says down.

CI / local `pytest` does **not** require a running GROBID — all tests
mock `requests` or call the parser directly.

## Knock-on effects

- Existing Chroma data extracted via PyPDF still works for retrieval;
  reindex (`python -m src reindex`) to pick up section metadata + the
  better extraction. This is the same reindex path 0005 / 0008
  already established.
- `extract_text_from_pdfs` now requires GROBID to be up. The CLI
  `process` and `reindex` commands surface the `GrobidUnavailable`
  exception with a clear message including the docker compose hint.

## Follow-ups not done here

- **Concurrent extraction.** Currently sequential. GROBID is
  thread-safe and the lightweight image handles ~5–10 concurrent
  requests on a laptop. A `ThreadPoolExecutor` would meaningfully
  shorten the first-time index of a `~/Downloads` folder. Hold off
  until 0010 — once Nomic + Qdrant land, we'll know whether GROBID
  or the embedder is the actual bottleneck.
- **Header / reference extraction.** GROBID returns rich author /
  affiliation / reference data; we currently throw it away. Will be
  consumed by the entity-graph layer (Stage 4).
- **README rewrite.** A proper rewrite is owed at the end of Stage 2 /
  start of Stage 3 once the FastAPI surface is stable.
