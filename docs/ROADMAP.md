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
- [x] 0002 — Preserve unicode in `clean_text` (NFKC + control-char strip)
- [x] 0003 — Reduce `CHUNK_SIZE` to 1200, tighten splitter
- [x] 0004 — Removed dead `src/embedding.py`
- [x] 0005 — Env / CLI config + `reindex` command
- [x] 0006 — Replaced `rank_bm25` + pickle with `bm25s`; hybrid_search now id-based
- [x] 0007 — `ruff`, `mypy`, `pytest`, `Makefile`; 18 tests landed

Stage 1 complete. Next: **Stage 2** (GROBID + Nomic Embed v1.5 + Qdrant + reranker).

### Stage 2 — Extraction + embeddings overhaul
- [x] 0008 — Pydantic data models (`Chunk` / `SearchHit` / `SearchResponse`) + per-page chunking
- [x] 0009 — GROBID extractor (local Docker), section as soft metadata hint, hard-fail when down
- [x] 0010 — Qdrant + Nomic Embed v1.5 + LRU query cache (combined storage swap)
- [ ] 0011 — Cross-encoder reranker (optional, gated by env var)

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
