# Hybrid_Retrieval — Roadmap

Production-grade rebuild of the local hybrid PDF search system. Will also serve as the upstream grounding layer for the **Darwin Research Agent** (which downloads arXiv papers to a folder; this project indexes and serves them).

## Target stack

| Layer | Choice | Replaces |
|---|---|---|
| API | FastAPI | (Tkinter only) |
| Frontend | Web UI (TBD framework) | Tkinter |
| PDF extraction | GROBID | PyPDFLoader |
| Embeddings | Nomic Embed v1.5 | all-MiniLM-L6-v2 |
| Vector store | Qdrant | ChromaDB |
| Sparse retrieval | bm25s (fastest) | rank_bm25 |
| Reranker | TBD (BGE-reranker-v2-m3 candidate) | — |
| Knowledge graph | Entity graph (concept-level) | — |
| Storage of KG | TBD (KuzuDB candidate) | — |

## Stages

### Stage 1 — Foundation fixes (current)
Fix correctness/quality bugs in the existing code before any stack swap. No new deps.

- [x] 0001 — Fix RRF fusion bug in `src/search.py`
- [ ] 0002 — Restrict source folder to `.pdf` only (already done; verify) and stop stripping non-ASCII in `clean_text` (loses math/Greek)
- [ ] 0003 — Reduce `CHUNK_SIZE` to a sane default (~1200 chars / 250 tokens) and tighten splitter
- [ ] 0004 — Remove dead `src/embedding.py` (unused; Chroma uses its own embedder)
- [ ] 0005 — Parameterize config via env / CLI (drop hard-coded `/Users/ammar/...` paths)
- [ ] 0006 — Replace `rank_bm25` + pickle with `bm25s` (incremental, ~100× faster)
- [ ] 0007 — Add `ruff`, `mypy`, `pytest`, basic CI, `Makefile`

### Stage 2 — Extraction + embeddings overhaul
- GROBID service (Docker) for academic PDFs; capture page numbers per chunk so we can deep-link.
- Nomic Embed v1.5 via sentence-transformers (or `nomic` SDK). Persist 768d.
- Optional reranker over top-50 fused.

### Stage 3 — Service + UI
- FastAPI app: `/index`, `/search`, `/document/{id}/page/{n}` endpoints.
- Web UI: query box, ranked snippet list, click → open PDF at the matching page.
- Tkinter retired.

### Stage 4 — Knowledge graph (entities)
- LLM-extracted entities: methods, datasets, tasks, models, authors.
- Stored in graph DB (KuzuDB likely — embedded, no server).
- Surface "concepts in this paper" + "papers using concept X" queries.

### Stage 5 — Darwin integration
- Folder watcher / API hook so Darwin's downloads are auto-indexed.
- Shared identifiers (arXiv id) between the two projects.

## Process

Every code change has a numbered Markdown file under `docs/changes/NNNN-short-name.md` describing **what / why / how**. Index entries land in `docs/CHANGELOG.md`.
