# Changelog

Newest first. Each entry links to a detailed change note in `docs/changes/`.

## Stage 2 — Extraction + embeddings overhaul

- **0010** — [Qdrant + Nomic Embed v1.5 + LRU cache](changes/0010-qdrant-nomic-embed-cache.md) — Replaced Chroma with Qdrant (Docker service, gRPC + REST). Added `NomicEmbedder` (sentence-transformers, MPS/CUDA/CPU auto, 512d Matryoshka, `search_document:` / `search_query:` prefixes, per-instance LRU on queries). `Embedder` Protocol so tests use a `FakeEmbedder`. Hybrid search now hydrates only chunk_ids that surfaced solely in BM25 — most queries do zero extra Qdrant calls. +7 tests.
- **0009** — [GROBID extractor](changes/0009-grobid-extractor.md) — Replaced PyPDF with GROBID (lightweight Docker image), with hard-fail when service is down. Sections surface as soft metadata hints (`Chunk.metadata.section`); page numbers come from TEI `coords`. Added `docker-compose.yml`, dropped langchain umbrella + pypdf deps, added `requests`. +11 tests against synthetic TEI fixture.
- **0008** — [Pydantic data models + per-page chunking](changes/0008-pydantic-models-and-per-page-chunking.md) — Added `Chunk`, `ChunkMetadata`, `SearchHit`, `SearchResponse` (frozen, `extra="forbid"`). Switched extraction to per-page chunking so every chunk carries 1-indexed `page_start`/`page_end` — foundation for click-to-page UX. Migrated extraction → storage → search → CLI/GUI to the typed flow. +6 model tests.

## Stage 1 — Foundation fixes

- **0007** — [Tooling: ruff + mypy + pytest + Makefile](changes/0007-tooling-and-tests.md) — 18 tests covering RRF math, `clean_text` unicode handling, env config, and end-to-end bm25s+Chroma+RRF. Ruff/mypy configs, `make {install,lint,format,type,test,check}`. Auto-format pass over `src/`.
- **0006** — [Swap `rank_bm25` for `bm25s`](changes/0006-bm25s-swap.md) — ~100× faster sparse retrieval, real tokenizer (lowercase + stopwords), directory-based save (no pickle). Hybrid search now fuses on stable chunk-ids instead of doc-text equality.
- **0005** — [Env / CLI config + `reindex` command](changes/0005-env-cli-config-and-reindex.md) — `HR_*` env vars + tiny `.env` loader, `Settings` dataclass, argparse CLI (`process`/`search`/`gui`/`reindex`/`info`), headless ops shared by CLI and GUI. Drops hard-coded `/Users/ammar/...` paths and unused `sentence-transformers` + `faiss-cpu` deps.
- **0004** — [Delete dead `src/embedding.py`](changes/0004-delete-dead-embedding-module.md) — Removed unused module that pretended to control the embedder; Chroma's default ONNX MiniLM is what's actually running.
- **0003** — [Shrink chunk size and tighten splitter](changes/0003-shrink-chunk-size.md) — `CHUNK_SIZE` 20000→1200, `CHUNK_OVERLAP` 500→150. Old size produced 1–3 chunks per paper, killing retrieval granularity for both legs.
- **0002** — [Preserve unicode in `clean_text`](changes/0002-preserve-unicode-in-clean-text.md) — Stop nuking non-ASCII (Greek letters, math operators, accented author names). NFKC-normalize PDF ligatures (`ﬁ`→`fi`). Strip only control chars.
- **0001** — [Fix RRF fusion bug](changes/0001-fix-rrf-fusion-bug.md) — Hybrid scoring in `src/search.py` mixed rank indices with raw BM25 scores into the same `1/(k+x)` term, badly biasing results. Rewrote `reciprocal_rank_fusion` to consume per-ranker rank lists.
